import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
# Use the new dataset and collate_fn
from dataset import SpeechCommandsProcessedDataset, collate_fn, TARGET_SAMPLE_RATE, TARGET_NUM_SAMPLES
from model import AudioEmbeddingNet
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import os

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
        # waveform_batch: (B, 1, num_samples)
        if waveform_batch.ndim == 3 and waveform_batch.size(1) == 1:
            waveform_batch = waveform_batch.squeeze(1) # (B, num_samples)
        
        mel_spec = self.mel_spectrogram(waveform_batch)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        log_mel_spec = log_mel_spec.unsqueeze(1)
        
        resized_log_mel = torch.nn.functional.interpolate(
            log_mel_spec, size=self.output_size, mode='bilinear', align_corners=False
        )
        return resized_log_mel

def train(model, preprocessor, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda'):
    model.to(device)
    preprocessor.to(device)
    
    opt = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            waveforms, labels = waveforms.to(device, non_blocking=True), labels.to(device, non_blocking=True) # Added non_blocking
            
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device=='cuda')):
                processed_audio = preprocessor(waveforms)
                emb, out = model(processed_audio)
                loss = loss_fn(out, labels)

            opt.zero_grad(set_to_none=True) # Added set_to_none
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for waveforms, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                waveforms, labels = waveforms.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    processed_audio = preprocessor(waveforms)
                    _, out = model(processed_audio)
                
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total * 100
        print(f"  ↪️ Val Acc: {val_acc:.2f}%")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Check if preprocessing has been done, if not, guide the user
    if not os.path.exists(os.path.join('./data/speech_commands_processed_v2', 'metadata.pt')):
        print("="*50)
        print("IMPORTANT: Preprocessed data not found.")
        print("Please run 'python preprocess_speech_commands.py' first.")
        print("This will take some time but is a one-time setup.")
        print("="*50)
        exit()

    # For A100, num_workers can be higher.
    # Optimal num_workers often requires a bit of experimentation.
    # Rule of thumb: 2-4 workers per GPU. For a single A100, 8-16 is often good.
    # Can also use os.cpu_count() as a guide, but don't exceed available cores too much.
    num_cpus = os.cpu_count()
    num_workers = min(16, num_cpus if num_cpus else 1) 
    if device == 'cpu': # If on CPU, don't use multiple workers for simplicity
        num_workers = 0
    print(f"Using {num_workers} workers for DataLoader")
    
    # Instantiate the new dataset
    print("Initializing training dataset...")
    train_set = SpeechCommandsProcessedDataset("training")
    print("Initializing validation dataset...")
    val_set = SpeechCommandsProcessedDataset("validation")
    # test_set = SpeechCommandsProcessedDataset("testing") # If you need it

    # Increase batch_size significantly for A100
    batch_size = 4096  # Start here for A100 80GB, can likely go higher
    if device == 'cpu':
        batch_size = 64


    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None # Added prefetch_factor
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size * 2, # Validation can often use larger batch
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    num_classes = train_set.num_classes
    model = AudioEmbeddingNet(embedding_dim=64, num_classes=num_classes)
    
    # Optional: torch.compile for PyTorch 2.0+
    if hasattr(torch, 'compile') and device == 'cuda':
        print("Compiling model with torch.compile()...")
        try:
            model = torch.compile(model, mode="max-autotune") # or "reduce-overhead"
            print("Model compiled successfully.")
        except Exception as e:
            print(f"torch.compile failed: {e}. Using uncompiled model.")


    audio_preprocessor = AudioPreprocessor(
        sample_rate=TARGET_SAMPLE_RATE, 
        n_mels=40,
        output_size=(40,40)
    )

    print("Starting training...")
    train(model, audio_preprocessor, train_loader, val_loader, epochs=200, lr=1e-3, device=device)
