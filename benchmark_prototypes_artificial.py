import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import time
from tqdm import trange

# Assuming dataset.py and model.py are in the same directory or accessible via PYTHONPATH
# Adjust these imports if your project structure is different.
try:
    from dataset import SpeechCommandsProcessedDataset, collate_fn, TARGET_SAMPLE_RATE
    from model import AudioEmbeddingNet
    from train_learnt_audio_prototypes import AudioPreprocessor
except ImportError as e:
    print(f"Error importing local modules (dataset, model, train_learnt_audio_prototypes): {e}")
    print("Please ensure these files are in the correct path or adjust imports.")
    print("Falling back to dummy implementations for critical components if possible for script execution.")
    # Provide dummy implementations if essential for the script to run at all for structure checking
    TARGET_SAMPLE_RATE = 16000


    class AudioPreprocessor(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.dummy = nn.Linear(1, 1)

        def forward(self, x): return torch.randn(x.size(0), 1, 40, 40, device=x.device)  # Dummy spectrogram


    class AudioEmbeddingNet(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.dummy = nn.Linear(40 * 40,
                                                                                        kwargs.get('embedding_dim', 64))

        def forward(self, x): return self.dummy(x.view(x.size(0), -1))


    class SpeechCommandsProcessedDataset(torch.utils.data.Dataset):
        def __init__(self, subset="testing", data_dir=None): self.len = 100  # Dummy length

        def __len__(self): return self.len

        def __getitem__(self, idx): return torch.randn(TARGET_SAMPLE_RATE), 0  # Dummy audio, dummy label


    def collate_fn(batch):
        waveforms = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return torch.stack(waveforms), torch.tensor(labels)

# Import from the provided prototypical_network.py
try:
    from torch_prototypes.modules.prototypical_network import LearntPrototypes, SKLEARN_AVAILABLE
except ImportError:
    print("Error: Could not import LearntPrototypes from prototypical_network.py.")
    print("Please ensure prototypical_network.py is in the same directory or accessible via PYTHONPATH.")
    # Define a dummy LearntPrototypes if not found, to allow the script to be parsed
    SKLEARN_AVAILABLE = False


    class LearntPrototypes(nn.Module):
        def __init__(self, model, n_prototypes, embedding_dim, **kwargs):
            super().__init__()
            self.model = model
            self.prototypes = nn.Parameter(torch.randn(n_prototypes, embedding_dim))
            self.embedding_dim = embedding_dim
            self.n_prototypes = n_prototypes  # Added for consistency
            self.use_kdtree_for_inference = False  # Mock attribute
            self.dist = kwargs.get('dist', 'euclidean')
            self.squared = kwargs.get('squared', False)
            self.ph = None
            self.use_manual_distance = kwargs.get('use_manual_distance', False)

        def forward(self, *input_data, **kwargs):
            embeddings = self.model(*input_data, **kwargs)
            # Simplified dummy forward for placeholder
            if len(embeddings.shape) == 4:  # Case for pixel-wise prototypes (not used here)
                b, _, h, w = embeddings.shape
                return torch.randn(b, self.prototypes.shape[0], h, w, device=embeddings.device)
            else:  # Standard prototype comparison
                # embeddings: (batch_size, embedding_dim)
                # self.prototypes: (n_prototypes, embedding_dim)
                # A very simple placeholder for distance calculation
                dists = torch.randn(embeddings.shape[0], self.prototypes.shape[0], device=embeddings.device)
                return dists

        def build_kdtree(self):
            print("Mock build_kdtree called")
            if not SKLEARN_AVAILABLE:
                print("SKLEARN_AVAILABLE is False. KD-Tree cannot be built.")
                # In a real scenario, this might raise an error or handle gracefully
                # raise ImportError("scikit-learn is required for KD-Tree functionality but not found.")

        def enable_kdtree_inference(self):
            self.use_kdtree_for_inference = True
            if SKLEARN_AVAILABLE:
                self.build_kdtree()
            else:
                print(
                    "Warning: KD-Tree inference enabled, but scikit-learn is not available. Tree will not be built/used effectively.")

        def disable_kdtree_inference(self):
            self.use_kdtree_for_inference = False


def run_inference_benchmark_cpu(model, preprocessor, loader, device, num_benchmark_batches):
    """
    Runs inference for a few batches on CPU and collects timings.
    """
    model.eval()
    all_timings = []

    # Warm-up run (one batch)
    try:
        waveforms, _ = next(iter(loader))
        waveforms = waveforms.to(device)
        with torch.no_grad():
            processed_audio_warmup = preprocessor(waveforms)
            _ = model(processed_audio_warmup)
            if hasattr(model, 'model') and isinstance(model.model, nn.Module):
                _ = model.model(processed_audio_warmup)
    except Exception as e:
        print(f"Warm-up failed: {e}")
        return []

    batch_iterator = iter(loader)
    print(f"Running CPU benchmark for {num_benchmark_batches} batches with {model.prototypes.shape[0]} prototypes...")
    for batch_idx in trange(num_benchmark_batches):
        try:
            t_batch_start = time.perf_counter()

            t_data_load_start = time.perf_counter()
            waveforms, _ = next(batch_iterator)
            waveforms = waveforms.to(device)
            t_data_load_end = time.perf_counter()
            current_batch_timings = {"data_load": (t_data_load_end - t_data_load_start) * 1000}

            with torch.no_grad():
                t_preprocess_start = time.perf_counter()
                processed_audio = preprocessor(waveforms)
                t_preprocess_end = time.perf_counter()
                current_batch_timings["preprocess"] = (t_preprocess_end - t_preprocess_start) * 1000

                t_full_model_call_start = time.perf_counter()
                _ = model(processed_audio)
                t_full_model_call_end = time.perf_counter()
                current_batch_timings["full_model_call_external"] = (
                                                                            t_full_model_call_end - t_full_model_call_start) * 1000

                t_backbone_internal_start = time.perf_counter()
                _embeddings = model.model(processed_audio)
                t_backbone_internal_end = time.perf_counter()
                current_batch_timings["backbone_approx"] = (t_backbone_internal_end - t_backbone_internal_start) * 1000

                current_batch_timings["prototype_operations_approx"] = \
                    current_batch_timings["full_model_call_external"] - current_batch_timings["backbone_approx"]
                current_batch_timings["prototype_operations_approx"] = max(0, current_batch_timings[
                    "prototype_operations_approx"])

            t_batch_end = time.perf_counter()
            current_batch_timings["total_batch_time"] = (t_batch_end - t_batch_start) * 1000
            all_timings.append(current_batch_timings)
        except StopIteration:
            print("DataLoader exhausted.")
            break
        except Exception as e:
            print(f"Error during CPU batch {batch_idx} processing: {e}")
            import traceback
            traceback.print_exc()
            break
    return all_timings


def main_cpu(checkpoint_path, data_dir, batch_size, use_kdtree_arg, use_manual_dist_arg, num_benchmark_batches,
             num_prototypes_override):
    device = torch.device('cpu')
    print(f"Forcing CPU device for benchmark: {device}")

    if use_kdtree_arg and not SKLEARN_AVAILABLE:
        print(
            "Warning: --use_kdtree requires scikit-learn. KD-Tree mode will be disabled for actual operation, though benchmark may proceed with mock.")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path '{checkpoint_path}' does not exist.")
        return

    print(f"Loading checkpoint from '{checkpoint_path}' onto CPU...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    embedding_dim_chk = checkpoint.get('embedding_dim')
    num_classes_from_checkpoint = checkpoint.get('num_classes')
    # Handles cases where checkpoint is state_dict or a dict containing model_state_dict
    model_state_dict = checkpoint.get('model_state_dict', checkpoint if isinstance(checkpoint, dict) else {})

    # Infer dimensions if not directly in checkpoint
    # This logic prioritizes 'prototypes' tensor within the state_dict for dimension inference
    # as it's the most direct source for prototype information.
    if embedding_dim_chk is None or num_classes_from_checkpoint is None:
        found_dims_from_state_dict = False
        if isinstance(model_state_dict, dict):
            proto_tensor_key = None
            if 'prototypes' in model_state_dict:  # Direct key in LearntPrototypes state_dict
                proto_tensor_key = 'prototypes'
            elif 'model.prototypes' in model_state_dict:  # If LearntPrototypes is wrapped
                proto_tensor_key = 'model.prototypes'

            if proto_tensor_key and isinstance(model_state_dict[proto_tensor_key], torch.Tensor):
                proto_shape = model_state_dict[proto_tensor_key].shape
                num_classes_from_checkpoint = proto_shape[0]
                embedding_dim_chk = proto_shape[1]
                found_dims_from_state_dict = True
                print(
                    f"Inferred num_prototypes ({num_classes_from_checkpoint}) and embedding_dim ({embedding_dim_chk}) "
                    f"from '{proto_tensor_key}' tensor in state_dict.")

        if not found_dims_from_state_dict and 'prototypes' in checkpoint and isinstance(checkpoint['prototypes'],
                                                                                        torch.Tensor):
            # Fallback: Check if 'prototypes' is a top-level key in the checkpoint file itself
            proto_shape = checkpoint['prototypes'].shape
            num_classes_from_checkpoint = proto_shape[0]
            embedding_dim_chk = proto_shape[1]
            print(
                f"Inferred num_prototypes ({num_classes_from_checkpoint}) and embedding_dim ({embedding_dim_chk}) "
                f"from 'prototypes' tensor in main checkpoint dictionary.")
        elif not found_dims_from_state_dict:
            print("Error: Checkpoint missing 'embedding_dim' or 'num_classes', and cannot infer from prototype tensor.")
            print("Ensure your checkpoint contains 'embedding_dim' and 'num_classes', "
                  "or a 'prototypes' (or 'model.prototypes') tensor in its state_dict, "
                  "or a top-level 'prototypes' tensor.")
            print(
                f"Available keys in checkpoint: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            if isinstance(model_state_dict, dict):
                print(f"Available keys in model_state_dict: {list(model_state_dict.keys())}")
            return

    if embedding_dim_chk is None:  # Should be caught above, but as a safeguard
        print("Error: embedding_dim could not be determined. Exiting.")
        return

    preprocessor = AudioPreprocessor(sample_rate=TARGET_SAMPLE_RATE, n_mels=40, output_size=(40, 40))
    preprocessor.to(device).eval()

    base_embedding_model = AudioEmbeddingNet(embedding_dim=embedding_dim_chk, num_classes=None)

    # Initialize LearntPrototypes with inferred/loaded num_classes and embedding_dim
    # num_classes_from_checkpoint here is the initial number of prototypes.
    # It will be updated if num_prototypes_override is used.
    model = LearntPrototypes(
        model=base_embedding_model,
        n_prototypes=num_classes_from_checkpoint if num_classes_from_checkpoint is not None else 1,
        # Fallback for n_prototypes
        embedding_dim=embedding_dim_chk,
        dist=checkpoint.get('dist_metric', 'euclidean'),
        squared=checkpoint.get('squared_dist', False),
        ph=checkpoint.get('ph_delta'),
        device=device,
        use_manual_distance=use_manual_dist_arg
    )

    model.load_state_dict(model_state_dict, strict=False)
    model.to(device).eval()

    actual_num_prototypes_before_override = model.prototypes.shape[0]
    current_embedding_dim = model.prototypes.shape[1]  # Use this for consistency

    if num_prototypes_override > 0 and num_prototypes_override > actual_num_prototypes_before_override:
        print(
            f"\nArtificially increasing prototypes from {actual_num_prototypes_before_override} to {num_prototypes_override} "
            f"using a structured (cluster-based) method favorable for KD-Trees...")
        num_to_add = num_prototypes_override - actual_num_prototypes_before_override

        original_prototypes_data = model.prototypes.data.clone()

        # Determine num_artificial_clusters: aim for clusters of at least 5, max 50 clusters
        num_artificial_clusters = 0
        if num_to_add > 0:
            num_artificial_clusters = max(1, min(num_to_add // 5, 50))

        prototypes_per_artificial_cluster = 0
        remainder_prototypes = 0
        if num_artificial_clusters > 0:
            prototypes_per_artificial_cluster = num_to_add // num_artificial_clusters
            remainder_prototypes = num_to_add % num_artificial_clusters

        print(
            f"Generating {num_to_add} new prototypes structured into {num_artificial_clusters} clusters (approx {prototypes_per_artificial_cluster} per cluster).")

        added_prototypes_list = []

        if num_to_add > 0 and num_artificial_clusters > 0:
            # Determine characteristic scale and location of original prototypes
            if actual_num_prototypes_before_override > 0:
                original_proto_min_vals = original_prototypes_data.min(dim=0).values
                original_proto_max_vals = original_prototypes_data.max(dim=0).values
                original_proto_mean = original_prototypes_data.mean(dim=0)

                if actual_num_prototypes_before_override > 1:
                    original_proto_std_for_centers = original_prototypes_data.std(dim=0)
                    # Ensure std is not too small, to allow for spread of new cluster centers
                    original_proto_std_for_centers[original_proto_std_for_centers < 1e-3] = 1e-3
                else:  # Only one original prototype
                    original_proto_std_for_centers = torch.ones_like(
                        original_proto_mean) * 0.5  # Arbitrary reasonable spread

                max_dim_range = (original_proto_max_vals - original_proto_min_vals).max()
                if max_dim_range < 1e-3: max_dim_range = 1.0  # Fallback if original points are collapsed
                within_cluster_noise_scale = max_dim_range * 0.01  # very tight clusters
                if within_cluster_noise_scale < 1e-4: within_cluster_noise_scale = 1e-4  # ensure not excessively small

            else:  # No original prototypes, generate from scratch
                original_proto_mean = torch.zeros(current_embedding_dim, device=original_prototypes_data.device,
                                                  dtype=original_prototypes_data.dtype)
                original_proto_std_for_centers = torch.ones_like(original_proto_mean)  # Scale for new centers
                within_cluster_noise_scale = 0.01  # Small absolute noise for points within clusters

            artificial_cluster_centers_list = []
            for _ in range(num_artificial_clusters):
                # Generate center noise in U[-1, 1] for each dimension
                center_noise = (torch.rand_like(original_proto_mean) * 2.0 - 1.0)
                # New centers are shifted from original_proto_mean by this noise scaled by original_proto_std_for_centers.
                # Factor 2.0 helps spread new cluster centers further.
                center = original_proto_mean + center_noise * original_proto_std_for_centers * 2.0
                artificial_cluster_centers_list.append(center)

            if artificial_cluster_centers_list:
                artificial_cluster_centers = torch.stack(artificial_cluster_centers_list)

                for i in range(num_artificial_clusters):
                    num_in_this_cluster = prototypes_per_artificial_cluster + (1 if i < remainder_prototypes else 0)
                    if num_in_this_cluster == 0:
                        continue

                    cluster_points = artificial_cluster_centers[i] + \
                                     torch.randn(num_in_this_cluster, current_embedding_dim,
                                                 device=original_prototypes_data.device,
                                                 dtype=original_prototypes_data.dtype) * within_cluster_noise_scale
                    added_prototypes_list.append(cluster_points)

        if added_prototypes_list:
            added_prototypes = torch.cat(added_prototypes_list, dim=0)
            if added_prototypes.shape[0] != num_to_add:
                print(
                    f"Note: Generated {added_prototypes.shape[0]} artificial prototypes, target was {num_to_add}. Using generated count.")
            new_prototypes_data = torch.cat([original_prototypes_data, added_prototypes], dim=0)
        elif num_to_add > 0:  # num_to_add > 0 but failed to generate structured ones
            print(
                f"Warning: Failed to generate structured prototypes (num_to_add={num_to_add}, num_artificial_clusters={num_artificial_clusters}). Falling back to random.")
            fallback_std_scale = 1.0
            if actual_num_prototypes_before_override > 0 and 'original_proto_std_for_centers' in locals():
                fallback_std_scale = original_proto_std_for_centers.mean()

            dummy_prototypes = torch.randn(
                (num_to_add, current_embedding_dim),
                device=original_prototypes_data.device,
                dtype=original_prototypes_data.dtype
            ) * fallback_std_scale
            new_prototypes_data = torch.cat([original_prototypes_data, dummy_prototypes], dim=0)
        else:  # num_to_add == 0 or no new prototypes needed/generated
            new_prototypes_data = original_prototypes_data

        model.prototypes = nn.Parameter(new_prototypes_data, requires_grad=model.prototypes.requires_grad)
        model.n_prototypes = new_prototypes_data.shape[0]  # CRITICAL: Update n_prototypes attribute

        print(f"Prototypes shape after artificial increase: {model.prototypes.shape}")
    else:
        if num_prototypes_override > 0:
            print(
                f"\n--num_prototypes_override ({num_prototypes_override}) is not greater than actual prototypes "
                f"({actual_num_prototypes_before_override}). Using original number of prototypes.")
        print(f"\nUsing {model.prototypes.shape[0]} prototypes for benchmark.")

    try:
        effective_data_dir = data_dir
        if not os.path.isdir(effective_data_dir) and type(SpeechCommandsProcessedDataset) is not object:
            print(f"Warning: Data directory '{effective_data_dir}' not found. Using a placeholder for dataset loading.")

        if type(SpeechCommandsProcessedDataset) is not object and hasattr(SpeechCommandsProcessedDataset,
                                                                          '__init__') and \
                'data_dir' in SpeechCommandsProcessedDataset.__init__.__code__.co_varnames:
            test_set = SpeechCommandsProcessedDataset(subset="testing", data_dir=effective_data_dir)
        else:
            test_set = SpeechCommandsProcessedDataset(subset="testing")
    except Exception as e:
        print(f"Error: Could not load test dataset from '{effective_data_dir}'. {e}")
        return

    num_workers = 0  # CPU benchmark, keep data loading simple
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=False  # No CUDA
    )

    print(f"\n--- Benchmarking Standard O(N) Inference on CPU (Manual Dist: {use_manual_dist_arg}) ---")
    model.disable_kdtree_inference()
    timings_standard_cpu = run_inference_benchmark_cpu(model, preprocessor, test_loader, device, num_benchmark_batches)

    timings_kdtree_cpu = []
    if use_kdtree_arg:
        if not SKLEARN_AVAILABLE:
            print("\nSKLEARN_AVAILABLE is False. Skipping KD-Tree benchmark as scikit-learn is not installed.")
        else:
            print(f"\n--- Benchmarking KD-Tree Inference on CPU (Manual Dist settings primarily affect O(N) mode) ---")
            t_kdtree_build_start = time.perf_counter()
            try:
                model.enable_kdtree_inference()  # This builds/rebuilds the tree with current prototypes
                t_kdtree_build_end = time.perf_counter()
                print(
                    f"KD-Tree build time (CPU): {(t_kdtree_build_end - t_kdtree_build_start) * 1000:.2f} ms for {model.prototypes.shape[0]} prototypes.")
                timings_kdtree_cpu = run_inference_benchmark_cpu(model, preprocessor, test_loader, device,
                                                                 num_benchmark_batches)
            except ImportError:
                print(
                    "Skipping KD-Tree benchmark: scikit-learn not found during build_kdtree call (should be caught by SKLEARN_AVAILABLE).")
            except Exception as e:
                print(f"Skipping KD-Tree benchmark due to error during build or run: {e}")
                import traceback
                traceback.print_exc()

    def aggregate_and_print_timings(label, timings_list, num_prototypes_in_run):
        if not timings_list:
            print(f"\nNo timings recorded for {label}.")
            return None

        print(
            f"\n--- Aggregated CPU Timings for {label} (avg over {len(timings_list)} batches, {num_prototypes_in_run} prototypes) ---")
        aggregated = {}
        # Initialize keys to ensure they are present even if some batches fail for certain metrics
        all_stat_keys = {
            "data_load", "preprocess", "backbone_approx",
            "prototype_operations_approx", "full_model_call_external", "total_batch_time"
        }
        for key_to_init in all_stat_keys:
            aggregated[key_to_init] = 0.0

        num_valid_batches_for_key = {key: 0 for key in all_stat_keys}

        if timings_list:
            for t_batch in timings_list:
                for key, value in t_batch.items():
                    if value is not None:
                        aggregated[key] = aggregated.get(key, 0.0) + value
                        num_valid_batches_for_key[key] = num_valid_batches_for_key.get(key, 0) + 1

            for key in aggregated:
                if num_valid_batches_for_key.get(key, 0) > 0:
                    aggregated[key] /= num_valid_batches_for_key[key]
                else:
                    aggregated[key] = 0.0  # Or np.nan, or skip printing

        preferred_order = [
            "data_load", "preprocess", "backbone_approx",
            "prototype_operations_approx",
            "full_model_call_external", "total_batch_time"
        ]

        for key in preferred_order:
            if key in aggregated:  # num_valid_batches_for_key.get(key,0) > 0: # Only print if data was collected
                print(f"  {key:<30}: {aggregated[key]:.3f} ms")

        # Print any other keys not in preferred_order
        other_keys = sorted(list(set(aggregated.keys()) - set(preferred_order)))
        for key in other_keys:
            if key in aggregated:  # num_valid_batches_for_key.get(key,0) > 0:
                print(f"  {key:<30}: {aggregated[key]:.3f} ms")
        return aggregated

    print("\n" + "=" * 60)
    print("CPU Benchmark Summary")
    print(f"Manual Distance Calculation Used for O(N): {use_manual_dist_arg}")
    print(f"Final number of prototypes in model: {model.prototypes.shape[0]}")
    print("=" * 60)

    num_prototypes_in_model = model.prototypes.shape[0]

    agg_standard_cpu = aggregate_and_print_timings(f"Standard O(N) [CPU, ManualDist={use_manual_dist_arg}]",
                                                   timings_standard_cpu, num_prototypes_in_model)

    if use_kdtree_arg and SKLEARN_AVAILABLE and timings_kdtree_cpu:
        agg_kdtree_cpu = aggregate_and_print_timings(f"KD-Tree [CPU]",
                                                     timings_kdtree_cpu, num_prototypes_in_model)
        if agg_standard_cpu and agg_kdtree_cpu:
            std_total_time = agg_standard_cpu.get("total_batch_time", 0)
            kdt_total_time = agg_kdtree_cpu.get("total_batch_time", 0)

            if kdt_total_time > 0 and std_total_time > 0:  # Avoid division by zero
                total_speedup = std_total_time / kdt_total_time
                print(f"\n  Speedup (Total Batch Time, CPU): {total_speedup:.2f}x")
            else:
                print("\n  Speedup (Total Batch Time, CPU): N/A (one or both timings are zero/missing)")

            std_proto_ops_time = agg_standard_cpu.get("prototype_operations_approx", 0)
            kdt_proto_ops_time = agg_kdtree_cpu.get("prototype_operations_approx", 0)

            if std_proto_ops_time > 0 and kdt_proto_ops_time > 0:  # Avoid division by zero
                proto_ops_speedup = std_proto_ops_time / kdt_proto_ops_time
                print(f"  Speedup (Prototype Operations Approx, CPU): {proto_ops_speedup:.2f}x "
                      f"({std_proto_ops_time:.2f}ms vs {kdt_proto_ops_time:.2f}ms for {num_prototypes_in_model} prototypes)")
            else:
                print(f"  Speedup (Prototype Operations Approx, CPU): N/A (one or both timings are zero/missing)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark CPU inference speed of LearntPrototypes.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the model checkpoint (.pt file).")
    parser.add_argument('--data_dir', type=str, default='./data/speech_commands_processed_v2',
                        help="Directory containing the preprocessed SpeechCommands dataset.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for CPU evaluation.")
    parser.add_argument('--use_kdtree', action='store_true',
                        help="Enable KD-Tree mode for comparison on CPU.")
    parser.add_argument('--use_manual_distance', action='store_true',
                        help="Enable manual distance calculation mode (for standard O(N) method, not KD-Tree).")
    parser.add_argument('--num_batches', type=int, default=50,
                        help="Number of batches from the test set to average timings over.")
    parser.add_argument('--num_prototypes_override', type=int, default=0,
                        help="Artificially set the number of prototypes for benchmarking. "
                             "If > 0 and larger than checkpoint's, new structured prototypes are added. "
                             "Set to 0 to use prototypes from checkpoint (default).")

    args = parser.parse_args()

    # Optional: For more deterministic behavior in prototype generation if needed for debugging,
    # but not essential for the benchmark's goal of favorable conditions.
    # torch.manual_seed(42)
    # np.random.seed(42)

    main_cpu(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_kdtree_arg=args.use_kdtree,
        use_manual_dist_arg=args.use_manual_distance,
        num_benchmark_batches=args.num_batches,
        num_prototypes_override=args.num_prototypes_override
    )

    # Example Usage:
    # First, create a dummy checkpoint if you don't have one:
    # import torch
    # checkpoint_content = {
    #    'embedding_dim': 64,
    #    'num_classes': 10, # Corresponds to n_prototypes if 'prototypes' tensor isn't in state_dict
    #    'dist_metric': 'euclidean',
    #    'squared_dist': False,
    #    'ph_delta': None,
    #    'model_state_dict': {
    #        'prototypes': torch.randn(10, 64) + torch.arange(10).unsqueeze(1).float() * 0.1, # Initial prototypes with some variation
    #        # For dummy AudioEmbeddingNet (if strict=True for load_state_dict on base_embedding_model)
    #        # 'model.dummy.weight': torch.randn(64, 40*40),
    #        # 'model.dummy.bias': torch.randn(64)
    #    }
    # }
    # torch.save(checkpoint_content, 'dummy_checkpoint.pt')
    #
    # Then run:
    # python benchmark_prototypes_artificial.py --checkpoint_path dummy_checkpoint.pt --num_batches 10
    # python benchmark_prototypes_artificial.py --checkpoint_path dummy_checkpoint.pt --num_batches 10 --use_kdtree
    # python benchmark_prototypes_artificial.py --checkpoint_path dummy_checkpoint.pt --num_batches 10 --use_manual_distance
    # python benchmark_prototypes_artificial.py --checkpoint_path dummy_checkpoint.pt --num_batches 10 --num_prototypes_override 5000 --use_kdtree
    # python benchmark_prototypes_artificial.py --checkpoint_path dummy_checkpoint.pt --num_batches 10 --num_prototypes_override 20 --use_kdtree # Test with small addition