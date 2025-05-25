# train_audio_prototypes.py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os
import numpy as np
import shutil # For managing best model checkpoint

# Assuming dataset.py, model.py, and the prototype modules are accessible
from dataset import SpeechCommandsProcessedDataset, collate_fn, TARGET_SAMPLE_RATE
from model import AudioEmbeddingNet
from torch.utils.data import DataLoader
import torchaudio.transforms as T

# From your torch_prototypes installation or stubs
from torch_prototypes.modules.prototypical_network import LearntPrototypes
from torch_prototypes.metrics.distortion import DistortionLoss

# GPU-accelerated Preprocessing Module (remains the same)
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

# --- D_metric (Cost Matrix) Creation (remains the same) ---
def create_audio_hierarchical_d_metric(class_list_ordered):
    """
    Creates a D_metric based on presumed acoustic/semantic hierarchies
    for the given list of speech commands.
    Args:
        class_list_ordered (list): List of class names in their index order.
    Returns:
        torch.Tensor: The D_metric matrix.
    """
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
    directions_core = ["up", "down", "left", "right", "forward", "backward"]
    controls_core = ["on", "off", "go", "stop"] 
    responses_core = ["yes", "no"]

    set_distance("up", "down", RELATED_OPP); set_distance("left", "right", RELATED_OPP)
    set_distance("forward", "backward", RELATED_OPP); set_distance("on", "off", RELATED_OPP)
    set_distance("go", "stop", RELATED_OPP); set_distance("yes", "no", RELATED_OPP)

    for i in range(len(directions_core)):
        for j in range(i + 1, len(directions_core)):
            if directions_core[i] in class_to_idx and directions_core[j] in class_to_idx and \
               D_metric[class_to_idx[directions_core[i]], class_to_idx[directions_core[j]]] == DEFAULT_DIST:
                 set_distance(directions_core[i], directions_core[j], RELATED_DIR_GROUP)

    control_words_plus_follow = ["go", "stop", "on", "off", "follow"] 
    for i in range(len(control_words_plus_follow)):
        for j in range(i + 1, len(control_words_plus_follow)):
            if control_words_plus_follow[i] in class_to_idx and control_words_plus_follow[j] in class_to_idx and \
               D_metric[class_to_idx[control_words_plus_follow[i]], class_to_idx[control_words_plus_follow[j]]] == DEFAULT_DIST:
                set_distance(control_words_plus_follow[i], control_words_plus_follow[j], RELATED_DIR_GROUP)

    animals = ["bird", "cat", "dog"]; SOMEWHAT_RELATED_ANIMAL = 0.9
    for i in range(len(animals)):
        for j in range(i + 1, len(animals)): set_distance(animals[i], animals[j], SOMEWHAT_RELATED_ANIMAL)
    
    names = ["marvin", "sheila"]; SOMEWHAT_RELATED_NAME = 1.2 
    if len(names) > 1 and all(n in class_to_idx for n in names): set_distance(names[0], names[1], SOMEWHAT_RELATED_NAME)

    PHONETIC_CLOSE = 0.6
    set_distance("three", "tree", PHONETIC_CLOSE)
    if "five" in class_to_idx and "nine" in class_to_idx and D_metric[class_to_idx["five"], class_to_idx["nine"]] > PHONETIC_CLOSE:
        set_distance("five", "nine", PHONETIC_CLOSE) 
    if "follow" in class_to_idx and "go" in class_to_idx and D_metric[class_to_idx["follow"], class_to_idx["go"]] > RELATED_DIR_GROUP:
        set_distance("follow", "go", RELATED_DIR_GROUP)

    common_nouns_misc = ["bed", "house", "tree"]; RELATED_NOUN_GROUP = 1.0
    for i in range(len(common_nouns_misc)):
        for j in range(i + 1, len(common_nouns_misc)):
            if common_nouns_misc[i] in class_to_idx and common_nouns_misc[j] in class_to_idx and \
               D_metric[class_to_idx[common_nouns_misc[i]], class_to_idx[common_nouns_misc[j]]] >= DEFAULT_DIST:
                 set_distance(common_nouns_misc[i], common_nouns_misc[j], RELATED_NOUN_GROUP)
    
    print("D_metric construction complete based on provided class list.")
    return D_metric


# --- Checkpoint Utilities (remains the same) ---
def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pt"):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, "best_model.pt"))
        print(f" 🎉 New best model saved to {os.path.join(checkpoint_dir, 'best_model.pt')}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device='cuda'):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found at '{checkpoint_path}'! Starting from scratch.")
        return 0, 0.0
        
    print(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state_val in optimizer.state.values(): # Corrected typo: state -> state_val
            for k, v in state_val.items():
                if isinstance(v, torch.Tensor):
                    state_val[k] = v.to(device)

    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    start_epoch = checkpoint.get('epoch', 0)
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    print(f"Resumed from epoch {start_epoch}. Best Val Acc: {best_val_acc:.2f}%")
    return start_epoch, best_val_acc


# --- Modified Training Function ---
def train_prototypes(base_embedding_model, preprocessor,
                     train_loader, val_loader, num_classes, embedding_dim,
                     epochs=20, lr=1e-3, device='cuda',
                     checkpoint_dir='checkpoints_proto',
                     save_every_epochs=5,
                     resume_from_checkpoint=None,
                     prototype_mode='guided',  # Can be 'guided' or 'free'
                     D_metric=None,            # Required if mode is 'guided', optional for logging in 'free'
                     lambda_metric=1.0):       # Used only if mode is 'guided'
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = LearntPrototypes(
        model=base_embedding_model,
        n_prototypes=num_classes,
        embedding_dim=embedding_dim,
        device=device
    )
    model.to(device)
    preprocessor.to(device)

    loss_fn_dist = None
    if D_metric is not None:
        D_metric = D_metric.to(device)
        loss_fn_dist = DistortionLoss(D_metric)

    if prototype_mode == 'guided':
        if loss_fn_dist is None:
            raise ValueError("D_metric (and thus loss_fn_dist) must be provided for 'guided' prototype mode.")
        print(f"Training with Metric-Guided Prototypes. Lambda_metric: {lambda_metric}")
    elif prototype_mode == 'free':
        print("Training with Free (Learnt) Prototypes. Distortion term will not be added to the training objective.")
        if loss_fn_dist is not None:
            print("Distortion (based on D_metric) will be logged if D_metric was provided.")
        # lambda_metric is ignored in 'free' mode for loss calculation
    else:
        raise ValueError(f"Invalid prototype_mode: {prototype_mode}. Choose 'guided' or 'free'.")

    opt = Adam(model.parameters(), lr=lr)
    loss_fn_ce = CrossEntropyLoss()
    
    use_amp = (device == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_val_acc = 0.0

    if resume_from_checkpoint:
        resume_path = resume_from_checkpoint
        if os.path.isdir(resume_from_checkpoint):
            potential_last_ckpt = os.path.join(resume_from_checkpoint, "checkpoint.pt")
            resume_path = potential_last_ckpt if os.path.exists(potential_last_ckpt) else os.path.join(resume_from_checkpoint, "best_model.pt")
        
        if os.path.exists(resume_path):
            start_epoch, best_val_acc = load_checkpoint(resume_path, model, opt, scaler, device)
        else:
            print(f"⚠️ resume_from_checkpoint path '{resume_path}' not found. Starting from scratch.")

    print(f"--- Prototype Mode: {prototype_mode.upper()} ---")
    if prototype_mode == 'guided':
        print(f"Using Lambda_metric: {lambda_metric}")
    print(f"Starting from epoch {start_epoch + 1}/{epochs}. Current best val_acc: {best_val_acc:.2f}%")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss_epoch, total_ce_loss_epoch, total_dist_loss_epoch_logged = 0.0, 0.0, 0.0
        correct, total = 0, 0

        for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            waveforms, labels = waveforms.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                processed_audio = preprocessor(waveforms)
                out_logits = model(processed_audio)
                classification_loss = loss_fn_ce(out_logits, labels)
                
                loss = classification_loss # Base loss
                
                current_prototypes = model.prototypes
                dist_loss_tensor_for_backprop = None

                if loss_fn_dist is not None: # If D_metric was given, calculate distortion
                    dist_loss_tensor_for_backprop = loss_fn_dist(current_prototypes)
                    total_dist_loss_epoch_logged += dist_loss_tensor_for_backprop.item() # Log its value
                    
                    if prototype_mode == 'guided':
                        loss = loss + lambda_metric * dist_loss_tensor_for_backprop

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss_epoch += loss.item()
            total_ce_loss_epoch += classification_loss.item()
            correct += (out_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100 if total > 0 else 0
        avg_loss = total_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0
        avg_ce_loss = total_ce_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0
        avg_dist_loss_logged = total_dist_loss_epoch_logged / len(train_loader) if len(train_loader) > 0 else 0
        
        dist_log_str = f", Avg Dist (Logged): {avg_dist_loss_logged:.4f}" if loss_fn_dist is not None else ""
        print(f"Epoch {epoch+1}/{epochs}, Avg Total Loss: {avg_loss:.4f}, "
              f"Avg CE Loss: {avg_ce_loss:.4f}{dist_log_str}, Train Acc: {train_acc:.2f}%")

        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for waveforms_val, labels_val in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                waveforms_val, labels_val = waveforms_val.to(device, non_blocking=True), labels_val.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    processed_audio_val = preprocessor(waveforms_val)
                    out_logits_val = model(processed_audio_val)
                correct_val += (out_logits_val.argmax(1) == labels_val).sum().item()
                total_val += labels_val.size(0)
        
        val_acc = correct_val / total_val * 100 if total_val > 0 else 0
        print(f"  ↪️ Val Acc: {val_acc:.2f}%")

        is_best = val_acc > best_val_acc
        if is_best: best_val_acc = val_acc
        
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'best_val_acc': best_val_acc,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'prototype_mode': prototype_mode,
            'lambda_metric': lambda_metric if prototype_mode == 'guided' else 0,
            'lr': lr
        }
        save_checkpoint(checkpoint_data, is_best, checkpoint_dir, filename="checkpoint.pt")
        if (epoch + 1) % save_every_epochs == 0:
            save_checkpoint(checkpoint_data, False, checkpoint_dir, filename=f"checkpoint_epoch_{epoch+1}.pt")
            print(f"   Saved periodic checkpoint to {os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')}")

    print("Training finished.")
    final_model_path = os.path.join(checkpoint_dir, f"final_model_epoch_{epochs}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model state_dict saved to {final_model_path}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if not os.path.exists(os.path.join('./data/speech_commands_processed_v2', 'metadata.pt')):
        print("="*50 + "\nIMPORTANT: Preprocessed data not found. Please run 'preprocess_speech_commands.py' first.\n" + "="*50)
        exit()

    num_cpus = os.cpu_count()
    num_workers = min(4, num_cpus if num_cpus else 1)
    if device == 'cpu': num_workers = 0 
    print(f"Using {num_workers} workers for DataLoader")

    print("Initializing training dataset...")
    train_set = SpeechCommandsProcessedDataset(subset="training") 
    print("Initializing validation dataset...")
    val_set = SpeechCommandsProcessedDataset(subset="validation")   

    batch_size = 256 
    if device == 'cpu': batch_size = 32

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=(device == 'cuda'),
        persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=(device == 'cuda'),
        persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
    )

    num_classes = train_set.num_classes
    index_to_label = train_set.index_to_label 
    if not all(i in index_to_label for i in range(num_classes)):
        raise ValueError("index_to_label in metadata is not complete or contiguous from 0.")
    ordered_class_names_from_dataset = [index_to_label[i] for i in range(num_classes)]
    
    print(f"Number of classes from dataset: {num_classes}")
    print(f"First 5 ordered class names for D_metric (from dataset): {ordered_class_names_from_dataset[:5]}")

    D_metric_matrix = create_audio_hierarchical_d_metric(ordered_class_names_from_dataset)
    print("\nShape of D_metric:", D_metric_matrix.shape)
    print(f"Is D_metric symmetric: {torch.all(D_metric_matrix == D_metric_matrix.T).item()}")
    print(f"Is D_metric diagonal zero: {torch.all(torch.diag(D_metric_matrix) == 0).item()}")


    # --- Configuration for Prototype Training ---
    PROTOTYPE_MODE = 'free'  # Changed to 'free' for Learnt Prototypes without metric guidance in loss
    # To run with guided prototypes, change to: PROTOTYPE_MODE = 'guided'

    EMBEDDING_DIM = 64
    LAMBDA_METRIC = 0.5  # This will be ignored if PROTOTYPE_MODE is 'free', but used if 'guided'
    
    audio_embedder = AudioEmbeddingNet(embedding_dim=EMBEDDING_DIM, num_classes=None) # num_classes=None for embedder
    audio_preprocessor = AudioPreprocessor(
        sample_rate=TARGET_SAMPLE_RATE, n_mels=40, output_size=(40,40)
    )
    
    # Adjust checkpoint directory based on mode
    CHECKPOINT_DIR_BASE = "checkpoints_audio_prototypes" # General name for prototype models
    if PROTOTYPE_MODE == 'guided':
        CHECKPOINT_DIR = f"{CHECKPOINT_DIR_BASE}_guided_lambda{LAMBDA_METRIC}"
    elif PROTOTYPE_MODE == 'free':
        CHECKPOINT_DIR = f"{CHECKPOINT_DIR_BASE}_free"
    else: # Should not happen if validated in train_prototypes
        CHECKPOINT_DIR = f"{CHECKPOINT_DIR_BASE}_unknown_config"

    # Example: RESUME_TRAINING_FROM = CHECKPOINT_DIR # to resume from the mode-specific dir
    # Example: RESUME_TRAINING_FROM = f"{CHECKPOINT_DIR}/checkpoint_epoch_10.pt"
    RESUME_TRAINING_FROM = None 

    print(f"\nStarting training run with configuration: {PROTOTYPE_MODE.upper()} Prototypes")
    train_prototypes(
        base_embedding_model=audio_embedder,
        preprocessor=audio_preprocessor,
        D_metric=D_metric_matrix, # D_metric is passed; used for loss if 'guided', for logging if 'free'
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM,
        epochs=200, 
        lr=1e-3,
        lambda_metric=LAMBDA_METRIC, # Passed; its use depends on prototype_mode
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
        save_every_epochs=10,
        resume_from_checkpoint=RESUME_TRAINING_FROM,
        prototype_mode=PROTOTYPE_MODE # Explicitly pass the mode
    )