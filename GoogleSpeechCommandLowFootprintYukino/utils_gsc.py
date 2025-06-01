# File: utils_gsc.py (or can be part of the main training script)
import random
import torch


def build_label_to_indices_map(dataset):
    """
    Builds a map from label_idx to a list of dataset indices for that label.
    Assumes dataset __getitem__ returns (data, label_idx).
    """
    label_to_indices = {}
    print(f"Building label_to_indices map for dataset with {len(dataset)} items...")
    for i in range(len(dataset)):
        try:
            _, label_idx = dataset[i]  # Get the integer label index
            if label_idx not in label_to_indices:
                label_to_indices[label_idx] = []
            label_to_indices[label_idx].append(i)
        except Exception as e:
            print(f"Warning: Could not process item {i} from dataset: {e}")
            # Potentially skip problematic files or handle more gracefully
            # For GSC, this should be rare if filtering by known labels is done
            continue
    return label_to_indices


def episodic_sampling_gsc(c_way, k_shot_support, k_shot_query,
                          available_label_indices, label_to_indices_map, dataset_instance):
    """
    Samples a task for few-shot learning.
    Args:
        c_way (int): Number of classes per task (N-way).
        k_shot_support (int): Number of support examples per class.
        k_shot_query (int): Number of query examples per class.
        available_label_indices (list): List of integer label indices to sample classes from.
        label_to_indices_map (dict): Maps label_idx -> list of dataset sample indices.
        dataset_instance: The dataset object to fetch items from.
    Returns:
        torch.Tensor: support_set (c_way, k_shot_support, C, H, W)
        torch.Tensor: query_set (c_way, k_shot_query, C, H, W)
    """
    chosen_label_indices = random.sample(available_label_indices, c_way)

    support_items_per_class = []
    query_items_per_class = []

    for label_idx in chosen_label_indices:
        all_dataset_indices_for_label = label_to_indices_map.get(label_idx, [])

        if len(all_dataset_indices_for_label) < k_shot_support + k_shot_query:
            # This should be rare if class distribution is somewhat even and k_shots are small
            # Fallback: sample with replacement or skip class (can unbalance task)
            # For now, print a warning and sample with replacement if needed
            print(f"Warning: Class {idx_to_label(label_idx)} (idx {label_idx}) has only "
                  f"{len(all_dataset_indices_for_label)} samples. "
                  f"Needing {k_shot_support + k_shot_query}. Sampling with replacement.")

            current_support_for_class = []
            current_query_for_class = []

            # Support set with replacement
            for _ in range(k_shot_support):
                ds_idx = random.choice(all_dataset_indices_for_label)
                mel_spec, _ = dataset_instance[ds_idx]
                current_support_for_class.append(mel_spec)

            # Query set with replacement
            for _ in range(k_shot_query):
                ds_idx = random.choice(all_dataset_indices_for_label)
                mel_spec, _ = dataset_instance[ds_idx]
                current_query_for_class.append(mel_spec)

        else:  # Enough samples, sample without replacement
            sampled_dataset_indices = random.sample(all_dataset_indices_for_label,
                                                    k_shot_support + k_shot_query)

            current_support_for_class = []
            for i in range(k_shot_support):
                mel_spec, _ = dataset_instance[sampled_dataset_indices[i]]
                current_support_for_class.append(mel_spec)

            current_query_for_class = []
            for i in range(k_shot_support, k_shot_support + k_shot_query):
                mel_spec, _ = dataset_instance[sampled_dataset_indices[i]]
                current_query_for_class.append(mel_spec)

        if k_shot_support > 0:
            support_items_per_class.append(torch.stack(current_support_for_class))
        else:  # Handle 0-shot support if ever needed
            dummy_shape = current_query_for_class[0].shape if k_shot_query > 0 else (1, N_MELS, 100)  # Dummy
            support_items_per_class.append(torch.empty(0, *dummy_shape))

        if k_shot_query > 0:
            query_items_per_class.append(torch.stack(current_query_for_class))
        else:  # Handle 0-shot query
            dummy_shape = current_support_for_class[0].shape if k_shot_support > 0 else (1, N_MELS, 100)  # Dummy
            query_items_per_class.append(torch.empty(0, *dummy_shape))

    return torch.stack(support_items_per_class), torch.stack(query_items_per_class)