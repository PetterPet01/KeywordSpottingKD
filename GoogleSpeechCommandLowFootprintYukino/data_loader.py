# File: data_loader.py (Modified)
import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset  # DataLoader removed, usually imported in train script
import torchaudio.transforms as T


class ESC50Dataset(Dataset):
    def __init__(self, base_path, meta_data, target_n_mels=40):  # Added target_n_mels
        self.base_path = base_path
        self.meta_data = meta_data
        self.target_n_mels = target_n_mels  # Store n_mels
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,  # Standard for ESC-50
            hop_length=512,  # Standard for ESC-50
            n_mels=self.target_n_mels  # Use target_n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        # This was already in the original provided code, good for batch indexing by MAML
        if isinstance(idx, (list, tuple)):
            return [self.get_single_item(i) for i in idx]
        else:
            return self.get_single_item(idx)

    def get_single_item(self, idx):
        row = self.meta_data.iloc[idx]
        filename = os.path.join(self.base_path, row.filename)
        label = row.target  # This is numeric class ID
        wav, sr = torchaudio.load(filename, normalize=True)

        # Resample if necessary (ESC-50 is already 44.1kHz, but good practice)
        if sr != 44100:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)(wav)

        mel_spec = self.mel_spectrogram(wav)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        # Output shape: (1, n_mels, n_frames), e.g. (1, 40, 431) for 5s audio at 44.1kHz, hop 512
        return mel_spec_db, label

# dataset = ESC50Dataset('../ESC-50/audio', meta_data)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# mel, label = dataset[21]
#
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 4))
# plt.imshow(mel.squeeze().numpy(), cmap='hot', origin='lower', aspect='auto')
# plt.title(f'Mel-Spectrogram of Example with Label: {label}')
# plt.xlabel('Time')
# plt.ylabel('Mel Filter Bank')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.savefig('log_mel.png')
