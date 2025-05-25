import torch
import torch.nn as nn
from torch.optim import Adam, AdamW # Added AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os
import numpy as np
import shutil

# Assuming dataset.py, model.py, and prototypical_networks.py are accessible
from dataset import SpeechCommandsProcessedDataset, collate_fn, TARGET_SAMPLE_RATE
from model import AudioEmbeddingNet # Your backbone model
from torch.utils.data import DataLoader
import torchaudio.transforms as T

# Use your local, MODIFIED LearntPrototypes
from torch_prototypes.modules.prototypical_network import LearntPrototypes
# Ensure DistortionLoss is correctly imported (from torch_prototypes or your stub)
from torch_prototypes.metrics.distortion import DistortionLoss


# --- GPU-accelerated Preprocessing Module ---
class AudioPreprocessor(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=40, output_size=(40,40)):
        super().__init__()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            center=True, pad_mode="reflect", norm="slaney", onesided=True, mel_scale="htk"
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)
        self.output_size = output_size

    def forward(self, waveform_batch):
        if waveform_batch.ndim == 3 and waveform_batch.size(1) == 1:
            waveform_batch = waveform_batch.squeeze(1)
        
        mel_spec = self.mel_spectrogram(waveform_batch)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        log_mel_spec = log_mel_spec.unsqueeze(1)
        
        resized_log_mel = torch.nn.functional.interpolate(
            log_mel_spec, size=self.output_size, mode='bilinear', align_corners=False
        )
        return resized_log_mel

# --- D_metric (Cost Matrix for DistortionLoss) ---
def create_audio_hierarchical_d_metric(class_list_ordered):
    num_classes = len(class_list_ordered)
    class_to_idx = {name: i for i, name in enumerate(class_list_ordered)}
    DEFAULT_DIST = 2.0
    D_metric = torch.full((num_classes, num_classes), DEFAULT_DIST, dtype=torch.float32)
    D_metric.fill_diagonal_(0)

    def set_distance(word1, word2, dist_val):
        if word1 in class_to_idx and word2 in class_to_idx:
            idx1, idx2 = class_to_idx[word1], class_to_idx[word2]
            D_metric[idx1, idx2] = dist_val
            D_metric[idx2, idx1] = dist_val

    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    num_map_val = {"zero":0, "one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9}
    VERY_CLOSE_NUM, CLOSE_NUM = 0.3, 0.5
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            n1, n2 = numbers[i], numbers[j]
            if abs(num_map_val.get(n1, -100) - num_map_val.get(n2, -100)) == 1:
                set_distance(n1, n2, VERY_CLOSE_NUM)
            elif n1 in num_map_val and n2 in num_map_val:
                set_distance(n1, n2, CLOSE_NUM)

    RELATED_OPP, RELATED_DIR_GROUP = 0.6, 0.8
    set_distance("up", "down", RELATED_OPP); set_distance("left", "right", RELATED_OPP)
    set_distance("forward", "backward", RELATED_OPP); set_distance("on", "off", RELATED_OPP)
    set_distance("go", "stop", RELATED_OPP); set_distance("yes", "no", RELATED_OPP)

    directions_core = ["up", "down", "left", "right", "forward", "backward"]
    for i in range(len(directions_core)):
        for j in range(i + 1, len(directions_core)):
            if D_metric[class_to_idx.get(directions_core[i], -1), class_to_idx.get(directions_core[j], -1)] == DEFAULT_DIST:
                 set_distance(directions_core[i], directions_core[j], RELATED_DIR_GROUP)

    control_words_plus_follow = ["go", "stop", "on", "off", "follow"]
    for i in range(len(control_words_plus_follow)):
        for j in range(i + 1, len(control_words_plus_follow)):
            if D_metric[class_to_idx.get(control_words_plus_follow[i], -1), class_to_idx.get(control_words_plus_follow[j], -1)] == DEFAULT_DIST:
                set_distance(control_words_plus_follow[i], control_words_plus_follow[j], RELATED_DIR_GROUP)
    
    animals = ["bird", "cat", "dog"]; SOMEWHAT_RELATED_ANIMAL = 0.9
    for i in range(len(animals)):
        for j in range(i + 1, len(animals)): set_distance(animals[i], animals[j], SOMEWHAT_RELATED_ANIMAL)

    PHONETIC_CLOSE = 0.6
    set_distance("three", "tree", PHONETIC_CLOSE)
    if "five" in class_to_idx and "nine" in class_to_idx and D_metric[class_to_idx["five"], class_to_idx["nine"]] > PHONETIC_CLOSE:
        set_distance("five", "nine", PHONETIC_CLOSE)
    print("D_metric construction complete.")
    return D_metric

# --- Checkpoint Utilities ---
def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pt", best_filename="best_model.pt"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, best_filename))
        print(f" 🎉 New best model saved to {os.path.join(checkpoint_dir, best_filename)}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device='cuda'):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found at '{checkpoint_path}'! Starting from scratch.")
        return 0, 0.0
        
    print(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle model state dict loading flexibly (e.g. if model is wrapped)
    if hasattr(model, 'module') and not isinstance(model, nn.DataParallel) and not isinstance(model, nn.parallel.DistributedDataParallel):
        # If model was saved with DataParallel wrapper but loaded without
        # Or if it's a custom wrapper like LearntPrototypes containing 'model'
        model_state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
        # Adjust for LearntPrototypes structure if necessary
        if 'prototypes' in model.state_dict() and 'prototypes' not in model_state_dict: # Likely loading into LearntPrototypes
             model.load_state_dict(model_state_dict) # This might be the full LearntPrototypes state
        elif hasattr(model, 'model'): # For LearntPrototypes, load backbone part
            try:
                model.model.load_state_dict(model_state_dict)
            except RuntimeError: # If keys don't match perfectly (e.g. classifier part)
                print("Partial load into model.model due to key mismatch (expected for backbone).")
                model.model.load_state_dict(model_state_dict, strict=False)
        else: # General case
            try:
                model.load_state_dict(model_state_dict)
            except RuntimeError:
                 print("Partial load into model due to key mismatch.")
                 model.load_state_dict(model_state_dict, strict=False)


    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint: # Fallback for other conventions
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("⚠️ No model state_dict found in checkpoint.")
        return 0,0.0
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor): state[k] = v.to(device)

    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    start_epoch = checkpoint.get('epoch', 0)
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    print(f"Resumed from epoch {start_epoch}. Best Val Acc: {best_val_acc:.2f}%")
    return start_epoch, best_val_acc

# --- STAGE 1: PRE-TRAINING THE BACKBONE ---
def pretrain_backbone_with_classifier(
    backbone_model_instance, preprocessor, train_loader, val_loader,
    epochs=20, lr=1e-3, device='cuda',
    checkpoint_dir='checkpoints_pretrain', resume_from_checkpoint=None, save_every_epochs=5
):
    print("\n--- Stage 1: Pre-training Backbone with Auxiliary Classifier ---")
    
    # Ensure the backbone_model_instance has its classifier (AudioEmbeddingNet handles this via num_classes)
    if backbone_model_instance.classifier is None:
        raise ValueError("Backbone model for pre-training must have a classifier head.")

    backbone_model_instance.to(device)
    preprocessor.to(device)

    opt = AdamW(backbone_model_instance.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = CrossEntropyLoss()
    use_amp = (device == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_val_acc = 0.0

    if resume_from_checkpoint:
        start_epoch, best_val_acc = load_checkpoint(resume_from_checkpoint, backbone_model_instance, opt, scaler, device)

    for epoch in range(start_epoch, epochs):
        backbone_model_instance.train()
        total_loss_epoch, correct, total = 0, 0, 0
        for waveforms, labels in tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{epochs} [Train]"):
            waveforms, labels = waveforms.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                processed_audio = preprocessor(waveforms)
                _, out_logits = backbone_model_instance(processed_audio) # emb, logits
                loss = loss_fn(out_logits, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total_loss_epoch += loss.item(); correct += (out_logits.argmax(1) == labels).sum().item(); total += labels.size(0)
        
        train_acc = correct / total * 100 if total > 0 else 0
        avg_loss = total_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Pretrain Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

        backbone_model_instance.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for waveforms_val, labels_val in tqdm(val_loader, desc=f"Pretrain Epoch {epoch+1}/{epochs} [Val]"):
                waveforms_val, labels_val = waveforms_val.to(device, non_blocking=True), labels_val.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    processed_audio_val = preprocessor(waveforms_val)
                    _, out_logits_val = backbone_model_instance(processed_audio_val)
                correct_val += (out_logits_val.argmax(1) == labels_val).sum().item(); total_val += labels_val.size(0)
        val_acc = correct_val / total_val * 100 if total_val > 0 else 0
        print(f"  ↪️ Pretrain Val Acc: {val_acc:.2f}%")

        is_best = val_acc > best_val_acc
        if is_best: best_val_acc = val_acc
        
        ckpt_data = {'epoch': epoch + 1, 'model_state_dict': backbone_model_instance.state_dict(),
                     'optimizer_state_dict': opt.state_dict(), 'scaler_state_dict': scaler.state_dict() if use_amp else None,
                     'best_val_acc': best_val_acc}
        save_checkpoint(ckpt_data, is_best, checkpoint_dir, "checkpoint_pretrain.pt", "best_model_pretrain.pt")
        if (epoch + 1) % save_every_epochs == 0:
            save_checkpoint(ckpt_data, False, checkpoint_dir, f"checkpoint_pretrain_epoch_{epoch+1}.pt")
            print(f"   Saved periodic pretrain checkpoint to {os.path.join(checkpoint_dir, f'checkpoint_pretrain_epoch_{epoch+1}.pt')}")

    print("--- Pre-training Backbone Finished ---")
    best_model_path = os.path.join(checkpoint_dir, "best_model_pretrain.pt")
    if os.path.exists(best_model_path):
        print(f"Loading best pre-trained backbone weights from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        backbone_model_instance.load_state_dict(checkpoint['model_state_dict'])
    else: print("Warning: Best pre-trained model checkpoint not found. Using last epoch's weights.")

# --- UTILITY FOR STAGE 2: GETTING INITIAL PROTOTYPES ---
def get_prototypes_from_backbone(
    backbone_model, preprocessor, data_loader, num_classes, embedding_dim, device='cuda'
):
    print("Calculating initial prototypes from pre-trained backbone...")
    backbone_model.eval(); backbone_model.to(device); preprocessor.to(device)
    all_class_embeddings = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for waveforms, labels in tqdm(data_loader, desc="Extracting Embeddings for Prototypes"):
            waveforms = waveforms.to(device, non_blocking=True); labels_cpu = labels.cpu()
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device=='cuda')):
                processed_audio = preprocessor(waveforms)
                embeddings = backbone_model(processed_audio) # Should return only embeddings
                if isinstance(embeddings, tuple): embeddings = embeddings[0] # If it accidentally returned (emb, None)
            for i in range(embeddings.size(0)):
                all_class_embeddings[labels_cpu[i].item()].append(embeddings[i].detach().cpu().unsqueeze(0))
    initial_prototypes = torch.zeros((num_classes, embedding_dim), dtype=torch.float32)
    for i in range(num_classes):
        if all_class_embeddings[i]:
            initial_prototypes[i] = torch.mean(torch.cat(all_class_embeddings[i], dim=0), dim=0)
        else:
            print(f"Warning: No samples for class {i}. Using random init for this prototype.")
            initial_prototypes[i] = torch.randn(embedding_dim) * 0.01
    return initial_prototypes.to(device)

# --- STAGE 2: TRAINING THE PROTOTYPICAL NETWORK ---
def train_prototypical_network_with_guided_loss(
    prototypical_model, preprocessor, D_metric, train_loader, val_loader,
    epochs=20, lr=1e-4, lambda_metric=1.0, device='cuda',
    checkpoint_dir='checkpoints_proto_guided', resume_from_checkpoint=None, save_every_epochs=5
):
    print("\n--- Stage 2: Training Prototypical Network with Guided Loss ---")
    prototypical_model.to(device); preprocessor.to(device); D_metric = D_metric.to(device)

    opt = AdamW(prototypical_model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn_ce = CrossEntropyLoss()
    loss_fn_dist = DistortionLoss(D_metric)
    use_amp = (device == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch, best_val_acc = 0, 0.0
    if resume_from_checkpoint:
        start_epoch, best_val_acc = load_checkpoint(resume_from_checkpoint, prototypical_model, opt, scaler, device)

    print(f"Training Prototypical Network. Lambda_metric: {lambda_metric}")
    print(f"Starting from epoch {start_epoch + 1}/{epochs}. Current best val_acc: {best_val_acc:.2f}%")

    for epoch in range(start_epoch, epochs):
        prototypical_model.train()
        total_loss_ep, total_ce_ep, total_dist_ep, correct_ep, total_ep = 0,0,0,0,0
        for waveforms, labels in tqdm(train_loader, desc=f"Proto Epoch {epoch+1}/{epochs} [Train]"):
            waveforms, labels = waveforms.to(device,non_blocking=True), labels.to(device,non_blocking=True)
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                processed_audio = preprocessor(waveforms)
                out_logits = prototypical_model(processed_audio)
                classification_loss = loss_fn_ce(out_logits, labels)
                distortion_loss_val = loss_fn_dist(prototypical_model.prototypes)
                loss = classification_loss + lambda_metric * distortion_loss_val
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total_loss_ep+=loss.item(); total_ce_ep+=classification_loss.item(); total_dist_ep+=distortion_loss_val.item()
            correct_ep+=(out_logits.argmax(1)==labels).sum().item(); total_ep+=labels.size(0)

        train_acc = correct_ep/total_ep*100 if total_ep>0 else 0
        avg_loss, avg_ce, avg_dist = (val/len(train_loader) if len(train_loader)>0 else 0 for val in [total_loss_ep, total_ce_ep, total_dist_ep])
        print(f"Proto Epoch {epoch+1}/{epochs}, Avg Total Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, Dist: {avg_dist:.4f}, Train Acc: {train_acc:.2f}%")

        prototypical_model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for waveforms_val, labels_val in tqdm(val_loader, desc=f"Proto Epoch {epoch+1}/{epochs} [Val]"):
                waveforms_val, labels_val = waveforms_val.to(device,non_blocking=True), labels_val.to(device,non_blocking=True)
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    processed_audio_val = preprocessor(waveforms_val)
                    out_logits_val = prototypical_model(processed_audio_val)
                correct_val += (out_logits_val.argmax(1)==labels_val).sum().item(); total_val += labels_val.size(0)
        val_acc = correct_val/total_val*100 if total_val > 0 else 0
        print(f"  ↪️ Proto Val Acc: {val_acc:.2f}%")

        is_best = val_acc > best_val_acc
        if is_best: best_val_acc = val_acc
        ckpt_data = {'epoch': epoch + 1, 'model_state_dict': prototypical_model.state_dict(),
                     'optimizer_state_dict': opt.state_dict(), 'scaler_state_dict': scaler.state_dict() if use_amp else None,
                     'best_val_acc': best_val_acc, 'embedding_dim': prototypical_model.prototypes.shape[1],
                     'num_classes': prototypical_model.n_prototypes, 'lambda_metric': lambda_metric}
        save_checkpoint(ckpt_data, is_best, checkpoint_dir, "checkpoint_proto_guided.pt", "best_model_proto_guided.pt")
        if (epoch + 1) % save_every_epochs == 0:
             save_checkpoint(ckpt_data, False, checkpoint_dir, f"checkpoint_proto_guided_epoch_{epoch+1}.pt")
             print(f"   Saved periodic proto checkpoint to {os.path.join(checkpoint_dir, f'checkpoint_proto_guided_epoch_{epoch+1}.pt')}")
             
    print("Prototypical Network training finished.")
    final_model_path = os.path.join(checkpoint_dir, f"final_proto_guided_model_epoch_{epochs}.pt")
    torch.save(prototypical_model.state_dict(), final_model_path)
    print(f"Final prototypical model state_dict saved to {final_model_path}")

# --- Main Execution Block ---
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Configuration ---
    DATA_DIR = './data/speech_commands_processed_v2'
    EMBEDDING_DIM = 64
    PRETRAIN_EPOCHS = 30      # Epochs for backbone pre-training
    PRETRAIN_LR = 1e-3
    PRETRAIN_SAVE_EVERY = 5
    PRETRAIN_CHECKPOINT_DIR = "checkpoints_audio_pretrain"
    # Set to PRETRAIN_CHECKPOINT_DIR to resume from last best, or specific file
    RESUME_PRETRAIN_FROM = None # Example: PRETRAIN_CHECKPOINT_DIR or os.path.join(PRETRAIN_CHECKPOINT_DIR, "checkpoint_pretrain.pt") 

    PROTO_EPOCHS = 100        # Epochs for prototypical network training
    PROTO_LR = 5e-4           # Usually smaller for fine-tuning
    LAMBDA_METRIC = 0.5       # Weight for the distortion loss
    PROTO_SAVE_EVERY = 10
    PROTO_CHECKPOINT_DIR = "checkpoints_audio_proto_guided_multistage"
    RESUME_PROTO_FROM = None  # Example: PROTO_CHECKPOINT_DIR

    RUN_PRETRAINING = True   # Set to False if you have a pre-trained model and want to skip to Stage 2
    PRETRAINED_BACKBONE_LOAD_PATH = os.path.join(PRETRAIN_CHECKPOINT_DIR, "best_model_pretrain.pt") # Used if RUN_PRETRAINING is False

    # --- Dataset and DataLoader Setup ---
    if not os.path.exists(os.path.join(DATA_DIR, 'metadata.pt')):
        print(f"IMPORTANT: Preprocessed data not found at {DATA_DIR}. Run preprocessing first.")
        exit()

    num_cpus = os.cpu_count()
    num_workers = min(4, num_cpus if num_cpus else 1)
    if device == 'cpu': num_workers = 0
    print(f"Using {num_workers} workers for DataLoader")

    train_set = SpeechCommandsProcessedDataset(subset="training")
    val_set = SpeechCommandsProcessedDataset(subset="validation")
    batch_size = 256 if device == 'cuda' else 32

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=(device == 'cuda'), persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None)
    init_loader = DataLoader(train_set, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=(device == 'cuda'), persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None) # For prototype calculation
    val_loader = DataLoader(val_set, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=(device == 'cuda'), persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None)

    num_classes = train_set.num_classes
    ordered_class_names_from_dataset = [train_set.index_to_label[i] for i in range(num_classes)]
    D_metric_tensor = create_audio_hierarchical_d_metric(ordered_class_names_from_dataset)
    audio_preprocessor = AudioPreprocessor(sample_rate=TARGET_SAMPLE_RATE, n_mels=40, output_size=(40,40)).to(device)

    # --- Stage 1: Pre-training Backbone ---
    backbone_for_pretrain = AudioEmbeddingNet(embedding_dim=EMBEDDING_DIM, num_classes=num_classes)
    if RUN_PRETRAINING:
        pretrain_backbone_with_classifier(
            backbone_model_instance=backbone_for_pretrain,
            preprocessor=audio_preprocessor,
            train_loader=train_loader, val_loader=val_loader,
            epochs=PRETRAIN_EPOCHS, lr=PRETRAIN_LR, device=device,
            checkpoint_dir=PRETRAIN_CHECKPOINT_DIR,
            resume_from_checkpoint=RESUME_PRETRAIN_FROM,
            save_every_epochs=PRETRAIN_SAVE_EVERY
        )
        # backbone_for_pretrain now holds the best pre-trained weights
    elif os.path.exists(PRETRAINED_BACKBONE_LOAD_PATH):
        print(f"Skipping pre-training. Loading backbone from {PRETRAINED_BACKBONE_LOAD_PATH}")
        checkpoint = torch.load(PRETRAINED_BACKBONE_LOAD_PATH, map_location=device)
        backbone_for_pretrain.load_state_dict(checkpoint['model_state_dict'])
        print("Pre-trained backbone loaded successfully.")
    else:
        print(f"RUN_PRETRAINING is False, but no pre-trained model found at {PRETRAINED_BACKBONE_LOAD_PATH}. Exiting.")
        exit()
    
    # --- Prepare Backbone for Stage 2 (Prototypical Network) ---
    # This backbone instance will be part of LearntPrototypes and should NOT have its own classifier head active.
    base_backbone_for_proto = AudioEmbeddingNet(embedding_dim=EMBEDDING_DIM, num_classes=None) # num_classes=None ensures no classifier
    
    # Load weights from the (potentially) pre-trained backbone_for_pretrain
    # We only want to load the feature extractor and embedding layer weights, not the classifier.
    pretrained_state_dict = backbone_for_pretrain.state_dict()
    base_model_state_dict = base_backbone_for_proto.state_dict()
    
    # Filter keys: only load weights for layers that exist in base_backbone_for_proto
    weights_to_load = {k: v for k, v in pretrained_state_dict.items() if k in base_model_state_dict}
    missing_keys, unexpected_keys = base_backbone_for_proto.load_state_dict(weights_to_load, strict=False)
    print(f"Loaded pre-trained weights into base backbone. Missing: {missing_keys}, Unexpected: {unexpected_keys}")
    base_backbone_for_proto.to(device)

    # --- Calculate Initial Prototypes for Stage 2 ---
    initial_prototypes = get_prototypes_from_backbone(
        base_backbone_for_proto, audio_preprocessor, init_loader,
        num_classes, EMBEDDING_DIM, device
    )
    print(f"Initial prototypes calculated. Shape: {initial_prototypes.shape}")

    # --- Stage 2: Training Prototypical Network ---
    # Ensure your LearntPrototypes allows provided prototypes to be trainable!
    prototypical_network = LearntPrototypes(
        model=base_backbone_for_proto, # Pre-trained backbone (only embedding output)
        n_prototypes=num_classes,
        embedding_dim=EMBEDDING_DIM,
        prototypes=initial_prototypes, # Crucial: use the initialized prototypes
        # Make sure your LearntPrototypes class handles this to make them trainable
        # (e.g., by default or via a flag like learn_prototypes_explicitly=True)
        device=device
    )

    train_prototypical_network_with_guided_loss(
        prototypical_model=prototypical_network,
        preprocessor=audio_preprocessor,
        D_metric=D_metric_tensor,
        train_loader=train_loader, val_loader=val_loader,
        epochs=PROTO_EPOCHS, lr=PROTO_LR, lambda_metric=LAMBDA_METRIC, device=device,
        checkpoint_dir=PROTO_CHECKPOINT_DIR,
        resume_from_checkpoint=RESUME_PROTO_FROM,
        save_every_epochs=PROTO_SAVE_EVERY
    )

    print("All stages finished.")
