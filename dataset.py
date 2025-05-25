import os
import torch
import torchaudio # Keep for T.Resample if not pre-resampling all, but we are
from torch.utils.data import Dataset

# --- Configuration (must match preprocess_speech_commands.py) ---
PROCESSED_DATA_DIR = './data/speech_commands_processed_v2'
METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'metadata.pt')
TARGET_SAMPLE_RATE = 16000 # From metadata, but good to have here too
TARGET_NUM_SAMPLES = TARGET_SAMPLE_RATE # 1 second of audio
# -------------------------------------------------------------

class SpeechCommandsProcessedDataset(Dataset):
  def __init__(self, subset="training"): # subset can be "training", "validation", "testing"
    if not os.path.exists(METADATA_FILE):
        raise RuntimeError(
            f"Metadata file not found at {METADATA_FILE}. "
            "Please run preprocess_speech_commands.py first."
        )
    
    print(f"Loading preprocessed metadata from {METADATA_FILE} for subset: {subset}...")
    metadata = torch.load(METADATA_FILE)
    print("Metadata loaded.")

    if metadata['target_sample_rate'] != TARGET_SAMPLE_RATE:
        raise ValueError("Mismatch in target sample rate between preprocessed data and dataset config.")

    self.data_list = metadata[subset]
    self.label_to_index = metadata['label_to_index']
    self.index_to_label = metadata['index_to_label']
    self.num_classes = metadata['num_classes']
    self._target_num_samples = TARGET_NUM_SAMPLES # All waveforms will be padded/cut to this

  def __len__(self):
    return len(self.data_list)

  def _cut_if_necessary(self, waveform):
    if waveform.shape[1] > self._target_num_samples:
      waveform = waveform[:, :self._target_num_samples]
    return waveform

  def _right_pad_if_necessary(self, waveform):
    length_waveform = waveform.shape[1]
    if length_waveform < self._target_num_samples:
      num_missing_samples = self._target_num_samples - length_waveform
      last_dim_padding = (0, num_missing_samples) # (pad_left, pad_right) for last dim
      waveform = torch.nn.functional.pad(waveform, last_dim_padding)
    return waveform

  def __getitem__(self, idx):
    item_metadata = self.data_list[idx]
    processed_filepath = item_metadata['path']
    label_index = item_metadata['label_index']
    
    try:
        waveform = torch.load(processed_filepath) # Loads a pre-processed tensor
    except FileNotFoundError:
        print(f"ERROR: Processed file not found: {processed_filepath}")
        print("This might happen if METADATA_FILE is present but actual processed files were deleted.")
        print("Consider deleting METADATA_FILE and re-running preprocess_speech_commands.py.")
        raise
    except Exception as e:
        print(f"ERROR: Could not load or process file: {processed_filepath}")
        raise e


    # Waveform is already mono and at target sample rate.
    # Just need to cut or pad.
    waveform = self._cut_if_necessary(waveform)
    waveform = self._right_pad_if_necessary(waveform)
    
    # Ensure waveform is (1, num_samples)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] != 1: # Should not happen if preprocessing was correct
        waveform = torch.mean(waveform, dim=0, keepdim=True)


    return waveform, label_index


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    # Waveforms should already be fixed length (1, TARGET_NUM_SAMPLES) from __getitem__
    waveforms_batched = torch.stack(waveforms) # (B, 1, TARGET_NUM_SAMPLES)
    labels_batched = torch.tensor(labels)
    return waveforms_batched, labels_batched

