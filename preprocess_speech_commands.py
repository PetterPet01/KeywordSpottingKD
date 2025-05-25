import os
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

# --- Configuration (match dataset.py and train.py) ---
DATA_ROOT = './data'
PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, 'speech_commands_processed_v2') # v2 to avoid old data
METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'metadata.pt')
TARGET_SAMPLE_RATE = 16000
# TARGET_NUM_SAMPLES = TARGET_SAMPLE_RATE # 1 second, will be handled by padding in Dataset's __getitem__
# ----------------------------------------------------

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_subset(subset_name, original_dataset, label_to_index):
    print(f"Processing subset: {subset_name}...")
    subset_data = []
    
    # Create target directory for this subset's processed files
    subset_processed_dir = os.path.join(PROCESSED_DATA_DIR, subset_name)
    ensure_dir(subset_processed_dir)

    resampler_cache = {} # Cache resamplers for different source sample rates

    for i in tqdm(range(len(original_dataset)), desc=f"Preprocessing {subset_name}"):
        waveform, sample_rate, label, speaker_id, utterance_number = original_dataset[i]

        # 1. Mix down to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Resample if necessary
        if sample_rate != TARGET_SAMPLE_RATE:
            if sample_rate not in resampler_cache:
                resampler_cache[sample_rate] = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            resampler = resampler_cache[sample_rate]
            waveform = resampler(waveform)
        
        # We will pad/cut to TARGET_NUM_SAMPLES in the Dataset's __getitem__ dynamically
        # So here we just save the (potentially variable length) processed waveform

        # Create a unique-ish filename (original filename might not be directly available or clean)
        # Recreate a structure similar to original for clarity if desired
        label_dir = os.path.join(subset_processed_dir, label)
        ensure_dir(label_dir)
        # Using index and original label to form a somewhat unique path
        processed_filename = f"{speaker_id}_{utterance_number}_{i}.pt"
        processed_filepath = os.path.join(label_dir, processed_filename)
        
        # 3. Save processed waveform
        torch.save(waveform, processed_filepath)
        
        # 4. Store metadata
        subset_data.append({
            'path': processed_filepath, # Path to the *processed* file
            'label_index': label_to_index[label]
        })
    return subset_data

def main():
    ensure_dir(DATA_ROOT)
    ensure_dir(PROCESSED_DATA_DIR)

    if os.path.exists(METADATA_FILE):
        print(f"Metadata file {METADATA_FILE} already exists. Skipping preprocessing.")
        print("Delete it if you want to re-preprocess.")
        # Check if essential processed directories exist as a sanity check
        if not os.path.exists(os.path.join(PROCESSED_DATA_DIR, "training")) or \
           not os.path.exists(os.path.join(PROCESSED_DATA_DIR, "validation")):
            print("WARNING: Metadata file exists, but processed data directories seem incomplete.")
            print("Consider deleting the metadata file and re-running preprocessing.")
        return

    print("Downloading and preparing original SpeechCommands dataset (if not present)...")
    # Load once to get all labels and establish consistent mapping
    full_train_dataset_for_labels = torchaudio.datasets.SPEECHCOMMANDS(root=DATA_ROOT, download=True, subset="training")
    full_valid_dataset_for_labels = torchaudio.datasets.SPEECHCOMMANDS(root=DATA_ROOT, download=True, subset="validation")
    
    # Using a combined set of labels from both train and val to ensure consistency
    # though for SpeechCommands, they should be the same for these standard subsets.
    all_labels = sorted(list(set(x[2] for x in full_train_dataset_for_labels).union(set(x[2] for x in full_valid_dataset_for_labels))))
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    num_classes = len(label_to_index)
    print(f"Found {num_classes} classes.")

    # Now process each subset
    # We re-instantiate SPEECHCOMMANDS here to ensure we're iterating over the correct subset
    # It will use the already downloaded data.
    training_data_orig = torchaudio.datasets.SPEECHCOMMANDS(root=DATA_ROOT, download=False, subset="training")
    processed_training_data = preprocess_subset("training", training_data_orig, label_to_index)
    
    validation_data_orig = torchaudio.datasets.SPEECHCOMMANDS(root=DATA_ROOT, download=False, subset="validation")
    processed_validation_data = preprocess_subset("validation", validation_data_orig, label_to_index)
    
    testing_data_orig = torchaudio.datasets.SPEECHCOMMANDS(root=DATA_ROOT, download=False, subset="testing")
    processed_testing_data = preprocess_subset("testing", testing_data_orig, label_to_index)


    metadata = {
        'label_to_index': label_to_index,
        'index_to_label': index_to_label,
        'num_classes': num_classes,
        'training': processed_training_data,
        'validation': processed_validation_data,
        'testing': processed_testing_data, # Added testing set
        'target_sample_rate': TARGET_SAMPLE_RATE
    }

    print(f"Saving metadata to {METADATA_FILE}...")
    torch.save(metadata, METADATA_FILE)
    print("Preprocessing complete.")

if __name__ == '__main__':
    main()
