# File: data_loader_gsc.py
import os
import torch
import torchaudio
from torchaudio import transforms
from torchaudio.datasets import SPEECHCOMMANDS
import random
import math

EPS = 1e-9
SAMPLE_RATE = 16000
N_MELS = 40  # To match QBcResNetEncoderASM's expectation
N_FFT = 1024  # Common for 16kHz audio
HOP_LENGTH = 160  # 10ms hop for 16kHz
TARGET_AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * 1  # Pad/truncate to 1 second

# Standard GSC V2 labels (35 classes)
DEFAULT_LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]
N_CLASSES_GSC = len(DEFAULT_LABELS)

_label_to_idx_gsc = {label: i for i, label in enumerate(DEFAULT_LABELS)}
_idx_to_label_gsc = {i: label for label, i in _label_to_idx_gsc.items()}


def label_to_idx(label):
    return _label_to_idx_gsc[label]


def idx_to_label(idx):
    return _idx_to_label_gsc[idx]


class SpeechCommandsDatasetMAML(SPEECHCOMMANDS):
    def __init__(self, subset: str, path="./", download=True):
        super().__init__(path, download=download, subset=subset)

        self.to_mel = transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH
        )
        self.amplitude_to_db = transforms.AmplitudeToDB()
        self.subset = subset

        # Pre-filter walker for known labels to avoid issues with unknown words if dataset has more
        self.known_labels_set = set(DEFAULT_LABELS)
        self._walker = [
            s for s in self._walker
            if self._get_label_from_path(s) in self.known_labels_set
        ]

        if subset == "training":
            self._noise = []
            noise_base_path = os.path.join(self._path, "_background_noise_")
            if os.path.exists(noise_base_path):
                noise_paths = [
                    os.path.join(noise_base_path, item)
                    for item in os.listdir(noise_base_path)
                    if item.endswith(".wav")
                ]
                for noise_path in noise_paths:
                    noise_waveform, noise_sr = torchaudio.load(noise_path)
                    if noise_sr != SAMPLE_RATE:
                        noise_waveform = transforms.Resample(orig_freq=noise_sr, new_freq=SAMPLE_RATE)(noise_waveform)
                    if noise_waveform.size(0) > 1:  # If stereo, take mean
                        noise_waveform = torch.mean(noise_waveform, dim=0, keepdim=True)
                    self._noise.append(noise_waveform)

    def _get_label_from_path(self, filepath):
        return os.path.basename(os.path.dirname(filepath))

    def _pad_truncate_waveform(self, waveform):
        if waveform.size(1) < TARGET_AUDIO_LENGTH_SAMPLES:
            padding = TARGET_AUDIO_LENGTH_SAMPLES - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :TARGET_AUDIO_LENGTH_SAMPLES]
        return waveform

    def _noise_augment(self, waveform):
        if not self._noise:
            return waveform
        noise_waveform_orig = random.choice(self._noise)

        # Ensure noise is at least as long as the signal
        while noise_waveform_orig.size(1) < waveform.size(1):
            noise_waveform_orig = torch.cat([noise_waveform_orig, noise_waveform_orig], dim=1)

        noise_sample_start = 0
        if noise_waveform_orig.size(1) > waveform.size(1):
            noise_sample_start = random.randint(0, noise_waveform_orig.size(1) - waveform.size(1))
        noise_waveform = noise_waveform_orig[:, noise_sample_start: noise_sample_start + waveform.size(1)]

        signal_power = waveform.norm(p=2)
        noise_power = noise_waveform.norm(p=2)

        snr_dbs = [20, 10, 3, 0]  # Added 0 dB for more challenging noise
        snr = random.choice(snr_dbs)
        snr_val = math.exp(snr / 10.0)

        if noise_power > EPS:  # Avoid division by zero
            scale = snr_val * noise_power / (signal_power + EPS)
        else:
            scale = 1.0  # No noise power, don't scale signal

        # Mix: ensure signal is not completely drowned out for very low SNR
        # A common way is alpha * signal + (1-alpha) * noise, or scaled addition
        # Here, we scale signal to achieve target SNR relative to noise
        noisy_signal = (scale * waveform + noise_waveform) / (scale + 1.0)
        # Or, more simply:
        # noisy_signal = waveform + (signal_power / (snr_val * noise_power + EPS)) * noise_waveform
        return noisy_signal

    def _shift_augment(self, waveform):
        shift_ms = random.randint(-100, 100)  # Shift up to 100ms
        shift_samples = int(shift_ms * SAMPLE_RATE / 1000)

        if shift_samples == 0:
            return waveform

        shifted_waveform = torch.roll(waveform, shifts=shift_samples, dims=1)
        if shift_samples > 0:
            shifted_waveform[0, :shift_samples] = 0
        elif shift_samples < 0:
            shifted_waveform[0, shift_samples:] = 0
        return shifted_waveform

    def _augment(self, waveform):
        if random.random() < 0.8:  # Apply noise with 80% probability
            waveform = self._noise_augment(waveform)
        if random.random() < 0.5:  # Apply shift with 50% probability
            waveform = self._shift_augment(waveform)
        return waveform

    def get_labels_list(self):
        return DEFAULT_LABELS  # Returns the list of all possible labels

    def __getitem__(self, n):
        filepath = self._walker[n]
        waveform, sample_rate_orig = torchaudio.load(filepath)
        label_str = self._get_label_from_path(filepath)

        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate_orig != SAMPLE_RATE:
            resampler = transforms.Resample(orig_freq=sample_rate_orig, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform = self._pad_truncate_waveform(waveform)  # Ensure fixed length

        if self.subset == "training":
            waveform = self._augment(waveform)

        mel_spec = self.to_mel(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)  # (1, N_MELS, N_FRAMES)

        # Ensure mel_spec_db doesn't have nan/inf from log(0) if EPS wasn't enough
        mel_spec_db = torch.nan_to_num(mel_spec_db, nan=0.0, posinf=0.0, neginf=0.0)

        return mel_spec_db, label_to_idx(label_str)  # Return Mel spec and integer label index