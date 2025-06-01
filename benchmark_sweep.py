import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import time
from tqdm import trange
from tqdm import tqdm
import pandas as pd  # For results table

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not found. Plotting will be disabled. Results will be printed as a table.")
    print("You can install it with: pip install matplotlib pandas")

# Assuming dataset.py and model.py are in the same directory or accessible via PYTHONPATH
# Adjust these imports if your project structure is different.
try:
    from dataset import SpeechCommandsProcessedDataset, collate_fn, TARGET_SAMPLE_RATE
    from model import AudioEmbeddingNet
    from train_learnt_audio_prototypes import AudioPreprocessor  # Assuming this is where it is
except ImportError as e:
    print(f"Error importing local modules (dataset, model, train_learnt_audio_prototypes): {e}")
    print("Please ensure these files are in the correct path or adjust imports.")
    print("Falling back to dummy implementations for critical components if possible for script execution.")
    TARGET_SAMPLE_RATE = 16000


    class AudioPreprocessor(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.dummy = nn.Linear(1, 1)

        def forward(self, x): return torch.randn(x.size(0), 1, 40, 40, device=x.device)


    class AudioEmbeddingNet(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.dummy = nn.Linear(40 * 40,
                                                                                        kwargs.get('embedding_dim', 64))

        def forward(self, x): return self.dummy(x.view(x.size(0), -1))


    class SpeechCommandsProcessedDataset(torch.utils.data.Dataset):
        def __init__(self, subset="testing", data_dir=None): self.len = 1000  # Increased dummy length

        def __len__(self): return self.len

        def __getitem__(self, idx): return torch.randn(TARGET_SAMPLE_RATE), 0


    def collate_fn(batch):
        waveforms = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return torch.stack(waveforms), torch.tensor(labels)

try:
    from torch_prototypes.modules.prototypical_network import LearntPrototypes, SKLEARN_AVAILABLE
except ImportError:
    print("Error: Could not import LearntPrototypes from prototypical_network.py.")
    print("Please ensure prototypical_network.py is in the same directory or accessible via PYTHONPATH.")
    SKLEARN_AVAILABLE = False


    class LearntPrototypes(nn.Module):
        def __init__(self, model, n_prototypes, embedding_dim, **kwargs):
            super().__init__()
            self.model = model
            self.prototypes = nn.Parameter(torch.randn(n_prototypes, embedding_dim))
            self.embedding_dim = embedding_dim
            self.n_prototypes = n_prototypes
            self.use_kdtree_for_inference = False
            self.dist = kwargs.get('dist', 'euclidean')
            self.squared = kwargs.get('squared', False)
            self.ph = None
            self.use_manual_distance = kwargs.get('use_manual_distance', False)
            self._kdtree = None  # Mock attribute

        def forward(self, *input_data, **kwargs):
            embeddings = self.model(*input_data, **kwargs)
            if len(embeddings.shape) == 4:
                b, _, h, w = embeddings.shape
                return torch.randn(b, self.prototypes.shape[0], h, w, device=embeddings.device)
            else:
                # A very simple placeholder for distance calculation
                # embeddings: (batch_size, embedding_dim)
                # self.prototypes: (n_prototypes, embedding_dim)
                if self.use_kdtree_for_inference and self._kdtree is not None and SKLEARN_AVAILABLE:
                    # Mock KD-Tree query, actual query happens in real class
                    # For benchmark purposes, the time taken here should reflect kdtree query
                    # We can simulate a small delay or assume it's part of the "prototype_operations_approx"
                    # In the real class, this path would use self._kdtree.query
                    pass  # Actual distance calculation would be different

                # Fallback / Standard distance calculation (simplified)
                # This dummy version doesn't actually calculate distances
                # but is okay for structural testing of the benchmark script.
                # The timing "prototype_operations_approx" relies on subtracting backbone time.
                dists = torch.randn(embeddings.shape[0], self.prototypes.shape[0], device=embeddings.device)
                return dists

        def build_kdtree(self):
            if not SKLEARN_AVAILABLE:
                print("Mock build_kdtree: SKLEARN_AVAILABLE is False. KD-Tree cannot be built.")
                return
            print(f"Mock build_kdtree called for {self.prototypes.shape[0]} prototypes.")
            # In a real implementation, this would build the tree:
            # from sklearn.neighbors import KDTree
            # self._kdtree = KDTree(self.prototypes.data.cpu().numpy())
            self._kdtree = "mock_kdtree_object"  # Simulate tree presence

        def enable_kdtree_inference(self):
            if SKLEARN_AVAILABLE:
                self.build_kdtree()
                if self._kdtree is not None:
                    self.use_kdtree_for_inference = True
                else:
                    print("Warning: KD-Tree build failed/skipped. KD-Tree inference will not be effective.")
                    self.use_kdtree_for_inference = False
            else:
                print(
                    "Warning: KD-Tree inference enabled, but scikit-learn is not available. Tree will not be built/used.")
                self.use_kdtree_for_inference = False

        def disable_kdtree_inference(self):
            self.use_kdtree_for_inference = False
            # self._kdtree = None # Optionally clear the tree


def run_inference_benchmark_cpu(model, preprocessor, loader, device, num_benchmark_batches, current_num_prototypes):
    model.eval()
    all_timings = []
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
    desc = f"Benchmarking {current_num_prototypes} protos ({'KD-Tree' if model.use_kdtree_for_inference and SKLEARN_AVAILABLE else 'Standard'})"
    for _ in trange(num_benchmark_batches, desc=desc, leave=False):
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
                if hasattr(model, 'model') and model.model is not None:
                    _embeddings = model.model(processed_audio)
                else:  # Should not happen with AudioEmbeddingNet
                    _embeddings = processed_audio  # Or some other placeholder if model.model is the entire thing
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
            print(f"Error during CPU batch processing: {e}")
            import traceback
            traceback.print_exc()
            break
    return all_timings


def aggregate_timings(timings_list):
    if not timings_list: return None
    aggregated = {}
    all_stat_keys = {"data_load", "preprocess", "backbone_approx",
                     "prototype_operations_approx", "full_model_call_external", "total_batch_time"}
    for key_to_init in all_stat_keys: aggregated[key_to_init] = 0.0
    num_valid_batches_for_key = {key: 0 for key in all_stat_keys}

    for t_batch in timings_list:
        for key, value in t_batch.items():
            if value is not None and key in aggregated:
                aggregated[key] += value
                num_valid_batches_for_key[key] += 1

    for key in aggregated:
        if num_valid_batches_for_key.get(key, 0) > 0:
            aggregated[key] /= num_valid_batches_for_key[key]
        else:
            aggregated[key] = np.nan  # Use NaN for missing data
    return aggregated


def print_aggregated_timings_summary(label, aggregated_results, num_prototypes_in_run, num_batches):
    if not aggregated_results:
        print(f"\nNo timings recorded for {label}.")
        return
    print(
        f"\n--- Aggregated CPU Timings for {label} (avg over {num_batches} batches, {num_prototypes_in_run} prototypes) ---")
    preferred_order = ["data_load", "preprocess", "backbone_approx",
                       "prototype_operations_approx", "full_model_call_external", "total_batch_time"]
    for key in preferred_order:
        if key in aggregated_results:
            print(f"  {key:<30}: {aggregated_results[key]:.3f} ms")
    other_keys = sorted(list(set(aggregated_results.keys()) - set(preferred_order)))
    for key in other_keys:
        if key in aggregated_results:
            print(f"  {key:<30}: {aggregated_results[key]:.3f} ms")


def main_cpu_for_sweep(checkpoint_path, data_dir, batch_size, use_manual_dist_arg,
                       num_benchmark_batches, num_prototypes_override, attempt_kdtree):
    """
    Loads model, adjusts prototypes, runs benchmarks, and returns key metrics.
    `attempt_kdtree` means the KD-Tree part of the benchmark will be run if SKLEARN_AVAILABLE.
    """
    device = torch.device('cpu')
    # print(f"Running benchmark for override: {num_prototypes_override}, KD-Tree attempt: {attempt_kdtree}")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path '{checkpoint_path}' does not exist.")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    embedding_dim_chk = checkpoint.get('embedding_dim')
    num_classes_from_checkpoint = checkpoint.get('num_classes')
    model_state_dict = checkpoint.get('model_state_dict', checkpoint if isinstance(checkpoint, dict) else {})

    if embedding_dim_chk is None or num_classes_from_checkpoint is None:
        found_dims_from_state_dict = False
        if isinstance(model_state_dict, dict):
            proto_tensor_key = None
            if 'prototypes' in model_state_dict:
                proto_tensor_key = 'prototypes'
            elif 'model.prototypes' in model_state_dict:
                proto_tensor_key = 'model.prototypes'

            if proto_tensor_key and isinstance(model_state_dict[proto_tensor_key], torch.Tensor):
                proto_shape = model_state_dict[proto_tensor_key].shape
                num_classes_from_checkpoint = proto_shape[0];
                embedding_dim_chk = proto_shape[1]
                found_dims_from_state_dict = True
        if not found_dims_from_state_dict and 'prototypes' in checkpoint and isinstance(checkpoint['prototypes'],
                                                                                        torch.Tensor):
            proto_shape = checkpoint['prototypes'].shape
            num_classes_from_checkpoint = proto_shape[0];
            embedding_dim_chk = proto_shape[1]
        elif not found_dims_from_state_dict:
            print("Error: Critical dimensions missing from checkpoint.")
            return None
    if embedding_dim_chk is None: return None

    preprocessor = AudioPreprocessor(sample_rate=TARGET_SAMPLE_RATE, n_mels=40, output_size=(40, 40))
    preprocessor.to(device).eval()
    base_embedding_model = AudioEmbeddingNet(embedding_dim=embedding_dim_chk, num_classes=None)

    model = LearntPrototypes(
        model=base_embedding_model,
        n_prototypes=num_classes_from_checkpoint if num_classes_from_checkpoint is not None else 1,
        embedding_dim=embedding_dim_chk,
        dist=checkpoint.get('dist_metric', 'euclidean'), squared=checkpoint.get('squared_dist', False),
        ph=checkpoint.get('ph_delta'), device=device, use_manual_distance=use_manual_dist_arg
    )
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device).eval()

    actual_num_prototypes_before_override = model.prototypes.shape[0]
    current_embedding_dim = model.prototypes.shape[1]

    if num_prototypes_override > 0 and num_prototypes_override > actual_num_prototypes_before_override:
        num_to_add = num_prototypes_override - actual_num_prototypes_before_override
        original_prototypes_data = model.prototypes.data.clone()
        num_artificial_clusters = 0
        if num_to_add > 0: num_artificial_clusters = max(1, min(num_to_add // 5, 50))
        prototypes_per_artificial_cluster, remainder_prototypes = (0, 0)
        if num_artificial_clusters > 0:
            prototypes_per_artificial_cluster = num_to_add // num_artificial_clusters
            remainder_prototypes = num_to_add % num_artificial_clusters

        added_prototypes_list = []
        if num_to_add > 0 and num_artificial_clusters > 0:
            if actual_num_prototypes_before_override > 0:
                original_proto_min_vals = original_prototypes_data.min(dim=0).values
                original_proto_max_vals = original_prototypes_data.max(dim=0).values
                original_proto_mean = original_prototypes_data.mean(dim=0)
                original_proto_std_for_centers = original_prototypes_data.std(
                    dim=0) if actual_num_prototypes_before_override > 1 else torch.ones_like(original_proto_mean) * 0.5
                original_proto_std_for_centers[original_proto_std_for_centers < 1e-3] = 1e-3
                max_dim_range = (original_proto_max_vals - original_proto_min_vals).max()
                if max_dim_range < 1e-3: max_dim_range = 1.0
                within_cluster_noise_scale = max(max_dim_range * 0.01, 1e-4)
            else:
                original_proto_mean = torch.zeros(current_embedding_dim, device=original_prototypes_data.device,
                                                  dtype=original_prototypes_data.dtype)
                original_proto_std_for_centers = torch.ones_like(original_proto_mean)
                within_cluster_noise_scale = 0.01

            artificial_cluster_centers_list = []
            for _ in range(num_artificial_clusters):
                center_noise = (torch.rand_like(original_proto_mean) * 2.0 - 1.0)
                center = original_proto_mean + center_noise * original_proto_std_for_centers * 2.0
                artificial_cluster_centers_list.append(center)

            if artificial_cluster_centers_list:
                artificial_cluster_centers = torch.stack(artificial_cluster_centers_list)
                for i in range(num_artificial_clusters):
                    num_in_this_cluster = prototypes_per_artificial_cluster + (1 if i < remainder_prototypes else 0)
                    if num_in_this_cluster == 0: continue
                    cluster_points = artificial_cluster_centers[i] + \
                                     torch.randn(num_in_this_cluster, current_embedding_dim,
                                                 device=original_prototypes_data.device,
                                                 dtype=original_prototypes_data.dtype) * within_cluster_noise_scale
                    added_prototypes_list.append(cluster_points)

        if added_prototypes_list:
            added_prototypes = torch.cat(added_prototypes_list, dim=0)
            new_prototypes_data = torch.cat([original_prototypes_data, added_prototypes], dim=0)
        elif num_to_add > 0:  # Fallback to random if structured generation failed but was intended
            fallback_std_scale = original_proto_std_for_centers.mean() if actual_num_prototypes_before_override > 0 and 'original_proto_std_for_centers' in locals() else 1.0
            dummy_prototypes = torch.randn((num_to_add, current_embedding_dim), device=original_prototypes_data.device,
                                           dtype=original_prototypes_data.dtype) * fallback_std_scale
            new_prototypes_data = torch.cat([original_prototypes_data, dummy_prototypes], dim=0)
        else:
            new_prototypes_data = original_prototypes_data

        model.prototypes = nn.Parameter(new_prototypes_data, requires_grad=model.prototypes.requires_grad)
        model.n_prototypes = new_prototypes_data.shape[0]
    # If num_prototypes_override is not > actual, model.prototypes remains as loaded.
    # model.n_prototypes should reflect this.
    elif num_prototypes_override > 0 and num_prototypes_override <= actual_num_prototypes_before_override:
        # User specified a number of prototypes that is not an increase.
        # We should either use the original number or truncate/select.
        # For simplicity in sweep, if override is not an increase, use original or what's loaded.
        # The current logic ensures model.n_prototypes and model.prototypes.shape[0] are consistent.
        # If we want to *reduce* prototypes, that's a different logic.
        # For now, this path means "use loaded prototypes, or fewer if num_prototypes_override < original"
        # This part might need refinement if the goal is to strictly set to num_prototypes_override
        # even if it's smaller than original. Current logic focuses on *increasing*.
        # Let's assume for the sweep, num_prototypes_override will generally be >= original.
        # If num_prototypes_override is specified but not > original, it effectively uses original.
        print(
            f"Note: num_prototypes_override ({num_prototypes_override}) not > original ({actual_num_prototypes_before_override}). Using {model.prototypes.shape[0]} protos.")
        # Ensure n_prototypes attribute matches current state after any potential no-op override
        model.n_prototypes = model.prototypes.shape[0]

    actual_num_prototypes_for_run = model.prototypes.shape[0]

    try:
        if type(SpeechCommandsProcessedDataset) is not object and hasattr(SpeechCommandsProcessedDataset,
                                                                          '__init__') and \
                'data_dir' in SpeechCommandsProcessedDataset.__init__.__code__.co_varnames:
            test_set = SpeechCommandsProcessedDataset(subset="testing", data_dir=data_dir)
        else:
            test_set = SpeechCommandsProcessedDataset(subset="testing")
    except Exception as e:
        print(f"Error: Could not load test dataset. {e}")
        return None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # --- Standard O(N) Benchmark ---
    model.disable_kdtree_inference()
    timings_standard_raw = run_inference_benchmark_cpu(model, preprocessor, test_loader, device, num_benchmark_batches,
                                                       actual_num_prototypes_for_run)
    agg_standard = aggregate_timings(timings_standard_raw)
    # print_aggregated_timings_summary(f"Standard O(N) [ManualDist={use_manual_dist_arg}]", agg_standard, actual_num_prototypes_for_run, num_benchmark_batches)

    # --- KD-Tree Benchmark ---
    kdtree_build_time_ms = np.nan
    agg_kdtree = None
    if attempt_kdtree:
        if not SKLEARN_AVAILABLE:
            print(f"KD-Tree benchmark for {actual_num_prototypes_for_run} protos skipped: scikit-learn not available.")
        else:
            t_kdtree_build_start = time.perf_counter()
            try:
                model.enable_kdtree_inference()  # Builds/rebuilds tree
                t_kdtree_build_end = time.perf_counter()
                kdtree_build_time_ms = (t_kdtree_build_end - t_kdtree_build_start) * 1000

                if model.use_kdtree_for_inference:  # Check if enable was successful
                    timings_kdtree_raw = run_inference_benchmark_cpu(model, preprocessor, test_loader, device,
                                                                     num_benchmark_batches,
                                                                     actual_num_prototypes_for_run)
                    agg_kdtree = aggregate_timings(timings_kdtree_raw)
                    # print_aggregated_timings_summary(f"KD-Tree", agg_kdtree, actual_num_prototypes_for_run, num_benchmark_batches)
                else:
                    print(
                        f"KD-Tree benchmark for {actual_num_prototypes_for_run} protos skipped: enable_kdtree_inference did not set use_kdtree_for_inference flag.")

            except Exception as e:
                print(f"KD-Tree benchmark for {actual_num_prototypes_for_run} protos failed: {e}")
                import traceback;
                traceback.print_exc()
    else:
        print(f"KD-Tree benchmark for {actual_num_prototypes_for_run} protos skipped: not attempted.")

    results = {
        "num_prototypes_actual": actual_num_prototypes_for_run,
        "standard_proto_ops_ms": agg_standard['prototype_operations_approx'] if agg_standard else np.nan,
        "standard_total_ms": agg_standard['total_batch_time'] if agg_standard else np.nan,
        "kdtree_proto_ops_ms": agg_kdtree['prototype_operations_approx'] if agg_kdtree else np.nan,
        "kdtree_total_ms": agg_kdtree['total_batch_time'] if agg_kdtree else np.nan,
        "kdtree_build_ms": kdtree_build_time_ms
    }
    return results


def perform_sweep_and_plot(base_args, prototype_counts_to_test, attempt_kdtree_sweep):
    all_results = []

    print("\nStarting benchmark sweep across prototype counts...")
    print(f"Prototype counts to test: {prototype_counts_to_test}")
    print(f"Attempting KD-Tree benchmarks: {attempt_kdtree_sweep} (requires scikit-learn)\n")

    for count_override in tqdm(prototype_counts_to_test, desc="Prototype Sweep"):
        print(f"\n----- Running for num_prototypes_override = {count_override} -----")
        current_run_results = main_cpu_for_sweep(
            checkpoint_path=base_args.checkpoint_path,
            data_dir=base_args.data_dir,
            batch_size=base_args.batch_size,
            use_manual_dist_arg=base_args.use_manual_distance,
            num_benchmark_batches=base_args.num_batches,
            num_prototypes_override=count_override,
            attempt_kdtree=attempt_kdtree_sweep  # Pass the sweep-level kdtree flag
        )
        if current_run_results:
            all_results.append(current_run_results)
        else:
            # Add a placeholder if a run fails, to maintain row correspondence if desired
            all_results.append({
                "num_prototypes_actual": count_override,  # Best guess
                "standard_proto_ops_ms": np.nan, "standard_total_ms": np.nan,
                "kdtree_proto_ops_ms": np.nan, "kdtree_total_ms": np.nan,
                "kdtree_build_ms": np.nan
            })

    print("\n\n" + "=" * 30 + " SWEEP SUMMARY " + "=" * 30)
    if not all_results:
        print("No results collected from the sweep.")
        return

    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values(by="num_prototypes_actual").reset_index(drop=True)

    print("\nAggregated Timings (ms):")
    # Select columns for display to make it cleaner
    display_cols = ["num_prototypes_actual", "standard_proto_ops_ms", "kdtree_proto_ops_ms",
                    "standard_total_ms", "kdtree_total_ms", "kdtree_build_ms"]
    # Ensure all display_cols exist, add if not (with NaN)
    for col in display_cols:
        if col not in df_results.columns:
            df_results[col] = np.nan

    print(df_results[display_cols].to_string(float_format="%.2f"))

    if MATPLOTLIB_AVAILABLE:
        actual_prototypes = df_results["num_prototypes_actual"].values

        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # Plot 1: Prototype Operations Time
        axs[0].plot(actual_prototypes, df_results["standard_proto_ops_ms"], marker='o', linestyle='-',
                    label='Standard O(N) Proto Ops')
        if attempt_kdtree_sweep and SKLEARN_AVAILABLE:
            axs[0].plot(actual_prototypes, df_results["kdtree_proto_ops_ms"], marker='x', linestyle='--',
                        label='KD-Tree Proto Ops')
        axs[0].set_ylabel('Time (ms)')
        axs[0].set_title('Prototype Operations Time vs. Number of Prototypes')
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Total Batch Time
        axs[1].plot(actual_prototypes, df_results["standard_total_ms"], marker='o', linestyle='-',
                    label='Standard O(N) Total Batch')
        if attempt_kdtree_sweep and SKLEARN_AVAILABLE:
            axs[1].plot(actual_prototypes, df_results["kdtree_total_ms"], marker='x', linestyle='--',
                        label='KD-Tree Total Batch')
        axs[1].set_ylabel('Time (ms)')
        axs[1].set_title('Total Batch Time vs. Number of Prototypes')
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: KD-Tree Build Time (if applicable)
        if attempt_kdtree_sweep and SKLEARN_AVAILABLE:
            axs[2].plot(actual_prototypes, df_results["kdtree_build_ms"], marker='s', linestyle=':',
                        label='KD-Tree Build Time')
            axs[2].set_ylabel('Time (ms)')
            axs[2].set_title('KD-Tree Build Time vs. Number of Prototypes')
            axs[2].legend()
            axs[2].grid(True)
        else:
            axs[2].text(0.5, 0.5, 'KD-Tree build times not applicable or not run.', horizontalalignment='center',
                        verticalalignment='center', transform=axs[2].transAxes)

        axs[-1].set_xlabel('Number of Prototypes')
        plt.tight_layout()

        plot_filename = f"benchmark_sweep_{time.strftime('%Y%m%d-%H%M%S')}.png"
        plt.savefig(plot_filename)
        print(f"\nPlot saved to {plot_filename}")
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot interactively: {e}. Plot saved to {plot_filename}.")

    else:
        print("\nMatplotlib not available. Skipping plot generation.")
        print("To generate plots, install matplotlib and pandas: pip install matplotlib pandas")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark CPU inference speed of LearntPrototypes across multiple prototype counts.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the model checkpoint (.pt file).")
    parser.add_argument('--data_dir', type=str, default='./data/speech_commands_processed_v2',
                        help="Directory for dataset.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for CPU evaluation.")
    parser.add_argument('--use_manual_distance', action='store_true',
                        help="Enable manual distance calculation (for standard O(N) mode).")
    parser.add_argument('--num_batches', type=int, default=20,
                        help="Number of batches for timing each run (reduced for faster sweep).")
    # New argument to control if KD-Tree part of the sweep is run
    parser.add_argument('--enable_kdtree_sweep', action='store_true',
                        help="Enable KD-Tree benchmarking during the sweep (if scikit-learn is available).")

    parser.add_argument('--prototype_counts', type=str, default="10,100,500,1000,2000,5000,10000",
                        help="Comma-separated list of prototype counts to test for num_prototypes_override.")

    args = parser.parse_args()

    try:
        prototype_counts_to_test = [int(x.strip()) for x in args.prototype_counts.split(',')]
        if not prototype_counts_to_test or any(c <= 0 for c in prototype_counts_to_test):
            raise ValueError("Prototype counts must be positive integers.")
    except ValueError as e:
        print(f"Error parsing --prototype_counts: {e}. Please provide a comma-separated list of positive integers.")
        exit(1)

    # Ensure dummy checkpoint exists for testing if a real one isn't provided
    if not os.path.exists(args.checkpoint_path) and "dummy_checkpoint.pt" in args.checkpoint_path:
        print(f"Creating a dummy checkpoint at {args.checkpoint_path} for testing...")
        dummy_emb_dim = 64
        dummy_num_classes = 10  # Initial number of prototypes
        # Create a state_dict that AudioEmbeddingNet would produce if it had a 'model' sub-module
        # and LearntPrototypes has 'prototypes'
        dummy_state_dict = {
            'prototypes': torch.randn(dummy_num_classes, dummy_emb_dim) + torch.arange(dummy_num_classes).unsqueeze(
                1).float() * 0.1,
            # For the dummy AudioEmbeddingNet (model.dummy...)
            'model.dummy.weight': torch.randn(dummy_emb_dim, 40 * 40),  # Matches dummy AudioEmbeddingNet
            'model.dummy.bias': torch.randn(dummy_emb_dim)
        }
        # Checkpoint structure expected by loading logic
        checkpoint_content = {
            'embedding_dim': dummy_emb_dim,
            'num_classes': dummy_num_classes,
            'dist_metric': 'euclidean',
            'squared_dist': False,
            'ph_delta': None,
            'model_state_dict': dummy_state_dict
        }
        torch.save(checkpoint_content, args.checkpoint_path)
        print(f"Dummy checkpoint '{args.checkpoint_path}' created.")

    perform_sweep_and_plot(
        base_args=args,
        prototype_counts_to_test=sorted(list(set(prototype_counts_to_test))),  # Ensure sorted unique
        attempt_kdtree_sweep=args.enable_kdtree_sweep  # Control KD-Tree attempts globally for sweep
    )

    # Example Usage:
    # 1. Ensure you have a checkpoint (or let it create 'dummy_checkpoint.pt'):
    #    (The script now creates 'dummy_checkpoint.pt' if specified and not found)
    #
    # 2. Run the sweep:
    #    python your_script_name.py --checkpoint_path dummy_checkpoint.pt --num_batches 5 --enable_kdtree_sweep --prototype_counts "50,100,500,1000"
    #
    #    To run without KD-Tree attempts (even if sklearn is available):
    #    python your_script_name.py --checkpoint_path dummy_checkpoint.pt --num_batches 5 --prototype_counts "50,100,500,1000"
    #
    #    To use your real checkpoint:
    #    python your_script_name.py --checkpoint_path path/to/your/model.pt --num_batches 20 --enable_kdtree_sweep --prototype_counts "100,500,1000,5000,10000,20000"