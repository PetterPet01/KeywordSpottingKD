# evaluate_prototypes.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import random

# Assuming dataset.py, model.py, and the prototype modules are accessible
from dataset import SpeechCommandsProcessedDataset, collate_fn, TARGET_SAMPLE_RATE # Removed LABELS
from model import AudioEmbeddingNet # Base embedding model
from train_learnt_audio_prototypes import AudioPreprocessor # Re-use the preprocessor

# From your torch_prototypes installation or stubs
from torch_prototypes.modules.prototypical_network import LearntPrototypes, SKLEARN_AVAILABLE


# Helper function for Few-Shot Data Preparation (Simplified)
def prepare_few_shot_episode(full_dataset, known_base_class_names, index_to_label_full,
                             n_way, k_shot, q_query, device, preprocessor):
    """
    Prepares a few-shot episode (support and query set) from the full_dataset.
    This is a simplified version. It assumes new classes are those not in known_base_class_names.
    Args:
        full_dataset: The complete SpeechCommandsProcessedDataset (e.g., test set).
        known_base_class_names: A list or set of strings representing the class names the base model was trained on.
        index_to_label_full: Mapping from original dataset indices to label strings for the full_dataset.
        n_way: Number of new classes for the few-shot task.
        k_shot: Number of support examples per new class.
        q_query: Number of query examples per new class.
        device: PyTorch device.
        preprocessor: AudioPreprocessor instance.
    Returns:
        A tuple: (support_waveforms, support_labels_mapped,
                  query_waveforms, query_labels_mapped,
                  few_shot_class_names, few_shot_global_indices)
        Returns None if data preparation fails.
    """
    print(f"Preparing {n_way}-way {k_shot}-shot episode with {q_query} query samples per class.")
    
    all_indices = list(range(len(full_dataset)))
    # all_labels_original = [full_dataset.get_label_from_index(full_dataset[i][1].item()) for i in all_indices] # Get string labels
    
    # Create a mapping from label string to list of sample indices
    label_to_indices = {}
    for idx, (_, original_label_idx_tensor) in enumerate(full_dataset): # Iterate through dataset to get actual sample indices
        original_label_idx = original_label_idx_tensor.item()
        # Use index_to_label_full which is full_test_set.index_to_label
        label_str = index_to_label_full.get(original_label_idx, f"unknown_label_{original_label_idx}")
        if label_str not in label_to_indices:
            label_to_indices[label_str] = []
        label_to_indices[label_str].append(idx) # Store dataset index (0 to len(full_dataset)-1)

    # Get unique labels from the dataset being used for few-shot (e.g., test set)
    available_labels_in_set = sorted(list(label_to_indices.keys()))
    
    # Exclude classes known by the base model
    base_model_known_labels_set = set(known_base_class_names)

    candidate_new_labels = [l for l in available_labels_in_set if l not in base_model_known_labels_set and len(label_to_indices.get(l, [])) >= (k_shot + q_query)]

    if len(candidate_new_labels) < n_way:
        print(f"Error: Not enough new classes with sufficient samples. Found {len(candidate_new_labels)}, need {n_way}.")
        print(f"Known base labels: {base_model_known_labels_set}")
        print(f"Candidate new labels: {candidate_new_labels}")
        return None

    selected_new_class_names = random.sample(candidate_new_labels, n_way)
    print(f"Selected new classes for few-shot: {selected_new_class_names}")

    support_waveforms_list = []
    support_labels_list = []
    query_waveforms_list = []
    query_labels_list = []
    
    few_shot_global_indices = [] # Store original global indices of these new classes

    for i, class_name in enumerate(selected_new_class_names):
        try:
            global_idx = next(k for k, v in index_to_label_full.items() if v == class_name)
            few_shot_global_indices.append(global_idx)
        except StopIteration:
            # Fallback: This might occur if class_name was somehow not in index_to_label_full,
            # or if we want to assign a new range of indices for few-shot specific tasks.
            # For now, let's assume this won't happen if selected_new_class_names come from available_labels_in_set
            # which are derived from index_to_label_full.
            # If it does, assign a placeholder or handle error.
            # A simple placeholder, assuming base classes occupy 0..N-1:
            # few_shot_global_indices.append(len(known_base_class_names) + i)
            print(f"Warning: Could not find global index for new class '{class_name}' in index_to_label_full. This might affect reporting if global indices are crucial elsewhere.")
            # Assigning a pseudo-index for consistency if needed, but it might not be a "true" global index
            few_shot_global_indices.append(-1 - i) # Negative to indicate it's not a real global index


        class_sample_indices = label_to_indices[class_name] # These are indices within full_dataset
        random.shuffle(class_sample_indices)
        
        support_indices_in_dataset = class_sample_indices[:k_shot]
        query_indices_in_dataset = class_sample_indices[k_shot : k_shot + q_query]

        if len(support_indices_in_dataset) < k_shot or len(query_indices_in_dataset) < q_query:
            print(f"Error: Class {class_name} does not have enough samples for {k_shot}-shot, {q_query}-query.")
            return None

        for s_idx in support_indices_in_dataset:
            waveform, _ = full_dataset[s_idx] # Get item from dataset using its index
            support_waveforms_list.append(waveform)
            support_labels_list.append(i) # Mapped label: 0 to n_way-1

        for q_idx in query_indices_in_dataset:
            waveform, _ = full_dataset[q_idx] # Get item from dataset using its index
            query_waveforms_list.append(waveform)
            query_labels_list.append(i) # Mapped label: 0 to n_way-1
            
    support_waveforms_batched, _ = collate_fn([(w.unsqueeze(0), torch.tensor(l)) for w, l in zip(support_waveforms_list, support_labels_list)])
    support_labels_mapped = torch.tensor(support_labels_list, dtype=torch.long)
    
    query_waveforms_batched, _ = collate_fn([(w.unsqueeze(0), torch.tensor(l)) for w, l in zip(query_waveforms_list, query_labels_list)])
    query_labels_mapped = torch.tensor(query_labels_list, dtype=torch.long)

    return (support_waveforms_batched.to(device), support_labels_mapped.to(device),
            query_waveforms_batched.to(device), query_labels_mapped.to(device),
            selected_new_class_names, few_shot_global_indices)


def evaluate_model(checkpoint_path, data_dir='./data/speech_commands_processed_v2',
                   batch_size=256, device_str='cuda',
                   use_kdtree=False,
                   few_shot_eval=False, n_way=0, k_shot=0, num_query_per_class=0):
    # ... (rest of the setup: device, checkpoint loading, model instantiation) ...
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if use_kdtree and not SKLEARN_AVAILABLE:
        print("Error: --use_kdtree requires scikit-learn. Please install it. Disabling KD-Tree.")
        use_kdtree = False

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path '{checkpoint_path}' does not exist.")
        return

    print(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    embedding_dim = checkpoint.get('embedding_dim')
    num_classes_from_checkpoint = checkpoint.get('num_classes')
    saved_class_names = checkpoint.get('class_names')


    if embedding_dim is None or num_classes_from_checkpoint is None:
        print("Error: Checkpoint missing 'embedding_dim' or 'num_classes'. Cannot re-instantiate model.")
        return
    
    print(f"Model parameters from checkpoint: embedding_dim={embedding_dim}, num_classes={num_classes_from_checkpoint}")

    preprocessor = AudioPreprocessor(sample_rate=TARGET_SAMPLE_RATE, n_mels=40, output_size=(40,40))
    preprocessor.to(device)
    preprocessor.eval()

    base_embedding_model = AudioEmbeddingNet(embedding_dim=embedding_dim, num_classes=None)

    model = LearntPrototypes(
        model=base_embedding_model,
        n_prototypes=num_classes_from_checkpoint, 
        embedding_dim=embedding_dim,
        dist=checkpoint.get('dist_metric', 'euclidean'), 
        squared=checkpoint.get('squared_dist', False), 
        ph=checkpoint.get('ph_delta'), 
        device=device 
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded 'model_state_dict' from checkpoint dictionary.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded state_dict directly from checkpoint file.")

    model.to(device)
    model.eval()

    if use_kdtree:
        print("Building KD-Tree for inference...")
        model.build_kdtree()
        model.use_kdtree_for_inference = True

    print(f"Initializing dataset from: {data_dir}")
    try:
        full_test_set = SpeechCommandsProcessedDataset(subset="testing")
    except FileNotFoundError as e:
        print(f"Error: Could not load test dataset. {e}")
        return

    # Determine original_class_names (base model's classes)
    if saved_class_names and len(saved_class_names) == num_classes_from_checkpoint:
        original_class_names = saved_class_names
    else:
        original_class_names = [full_test_set.index_to_label.get(i, f"Class {i}") for i in range(num_classes_from_checkpoint)]

    if num_classes_from_checkpoint != full_test_set.num_classes and not few_shot_eval:
        print(f"Warning: Num classes in checkpoint ({num_classes_from_checkpoint}) "
              f"differs from test dataset's total classes ({full_test_set.num_classes}).")

    if few_shot_eval:
        if not (n_way > 0 and k_shot > 0 and num_query_per_class > 0):
            print("Error: For few-shot evaluation, --n_way, --k_shot, and --num_query_per_class must be positive.")
            return

        print("\n--- Starting Few-Shot Evaluation ---")
        fs_data = prepare_few_shot_episode(
            full_dataset=full_test_set, 
            known_base_class_names=original_class_names, # Pass the derived original_class_names
            index_to_label_full=full_test_set.index_to_label, # Pass the dataset's full label mapping
            n_way=n_way, 
            k_shot=k_shot, 
            q_query=num_query_per_class, 
            device=device, 
            preprocessor=preprocessor
        )

        if fs_data is None:
            print("Failed to prepare few-shot episode. Aborting few-shot evaluation.")
            # return # Optionally return or continue to standard evaluation
        else:
            support_waveforms, support_labels_mapped, \
            query_waveforms, query_labels_mapped, \
            few_shot_class_names, _ = fs_data # few_shot_global_indices is not used later for now

            with torch.no_grad():
                processed_support_audio = preprocessor(support_waveforms)
                support_embeddings = model.model(processed_support_audio)

            model.augment_with_few_shot_prototypes(support_embeddings, support_labels_mapped)
            
            all_preds_fs = []
            all_labels_fs = query_labels_mapped.cpu().numpy()

            with torch.no_grad():
                processed_query_audio = preprocessor(query_waveforms)
                logits_fs = model(processed_query_audio) 
                
                # Logits for the new classes are at the end
                logits_for_new_classes = logits_fs[:, num_classes_from_checkpoint : num_classes_from_checkpoint + n_way]
                preds_fs_local = torch.argmax(logits_for_new_classes, dim=1)
                all_preds_fs.extend(preds_fs_local.cpu().numpy())

            accuracy_fs = accuracy_score(all_labels_fs, all_preds_fs)
            print(f"\n--- Few-Shot Evaluation Results ({n_way}-way, {k_shot}-shot) ---")
            print(f"Query Set Accuracy (on new classes only): {accuracy_fs * 100:.2f}%")
            
            report_fs = classification_report(
                all_labels_fs, all_preds_fs, 
                target_names=few_shot_class_names, 
                digits=3, zero_division=0
            )
            print("\nFew-Shot Classification Report (on new classes):")
            print(report_fs)

            model.revert_to_original_prototypes()
            print("--- Finished Few-Shot Evaluation ---\n")
    
    # --- Standard Evaluation on Test Set ---
    # (The rest of the standard evaluation code remains the same)
    print("--- Starting Standard Evaluation on Full Test Set ---")
    
    num_cpus = os.cpu_count()
    num_workers = min(4, num_cpus if num_cpus else 1)
    if device == torch.device('cpu'): num_workers = 0
    
    test_loader = DataLoader(
        full_test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=(device != torch.device('cpu')),
        persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
    )
    
    all_preds_std = []
    all_labels_std = []
    
    use_amp = (device != torch.device('cpu'))

    print(f"Running standard evaluation (AMP enabled: {use_amp})...")
    with torch.no_grad():
        for waveforms, labels in tqdm(test_loader, desc="Standard Evaluating"):
            waveforms, labels = waveforms.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                processed_audio = preprocessor(waveforms)
                logits = model(processed_audio) 
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds_std.extend(preds.cpu().numpy())
            all_labels_std.extend(labels.cpu().numpy())

    accuracy_std = accuracy_score(all_labels_std, all_preds_std)
    print(f"\n--- Standard Evaluation Results for {os.path.basename(checkpoint_path)} ---")
    print(f"Overall Test Accuracy: {accuracy_std * 100:.2f}%")

    print("\nStandard Classification Report:")
    unique_labels_in_data = np.unique(np.concatenate((all_labels_std, all_preds_std)))
    valid_labels_for_report = [l for l in unique_labels_in_data if l < len(original_class_names)]
    
    target_names_for_report = [original_class_names[i] for i in valid_labels_for_report]

    if not target_names_for_report and valid_labels_for_report:
        target_names_for_report = [f"Class {i}" for i in valid_labels_for_report]
    elif not valid_labels_for_report:
        print("No valid labels found to generate classification report for standard evaluation.")
    else:
        report_std = classification_report(
            all_labels_std, 
            all_preds_std, 
            labels=valid_labels_for_report, 
            target_names=target_names_for_report, 
            digits=3,
            zero_division=0
        )
        print(report_std)
    print("--- Finished Standard Evaluation ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained audio prototype model.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the model checkpoint (.pt file).")
    parser.add_argument('--data_dir', type=str, default='./data/speech_commands_processed_v2',
                        help="Directory containing the preprocessed SpeechCommands dataset.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for evaluation.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help="Device to use for evaluation ('cuda' or 'cpu').")
    
    parser.add_argument('--use_kdtree', action='store_true',
                        help="Use KD-Tree for faster inference (requires scikit-learn).")
    
    parser.add_argument('--few_shot_eval', action='store_true',
                        help="Enable N-way K-shot evaluation on new classes.")
    parser.add_argument('--n_way', type=int, default=5,
                        help="N: Number of new classes for few-shot task.")
    parser.add_argument('--k_shot', type=int, default=1,
                        help="K: Number of support examples per new class for few-shot task.")
    parser.add_argument('--num_query_per_class', type=int, default=15,
                        help="Number of query examples per new class for few-shot task.")
    
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device_str=args.device,
        use_kdtree=args.use_kdtree,
        few_shot_eval=args.few_shot_eval,
        n_way=args.n_way,
        k_shot=args.k_shot,
        num_query_per_class=args.num_query_per_class
    )
