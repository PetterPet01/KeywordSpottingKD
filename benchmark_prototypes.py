import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import time
from tqdm import trange

# Assuming dataset.py, model.py, and the prototype modules are accessible
from dataset import SpeechCommandsProcessedDataset, collate_fn, TARGET_SAMPLE_RATE
from model import AudioEmbeddingNet
from train_learnt_audio_prototypes import AudioPreprocessor

from torch_prototypes.modules.prototypical_network import LearntPrototypes, SKLEARN_AVAILABLE

# Using the LearntPrototypesWithTiming class from the previous example for detailed breakdown
class LearntPrototypesWithTiming(LearntPrototypes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_details = {}

    def _forward_kdtree(self, embeddings): # embeddings are already on CPU here
        self.timing_details['kdtree_query_prep_start'] = time.perf_counter()
        if self.kdtree is None:
            raise RuntimeError("KD-Tree not built.")

        # Embeddings are already CPU tensors, .numpy() is efficient
        query_embeddings_np = embeddings.detach().numpy() # No .cpu() needed

        if self.dist == "cosine" and self._kdtree_prototypes_normalized:
            norm = np.linalg.norm(query_embeddings_np, axis=1, keepdims=True)
            query_embeddings_np = query_embeddings_np / (norm + 1e-8)
        self.timing_details['kdtree_query_prep_end'] = time.perf_counter()
        
        self.timing_details['kdtree_lookup_start'] = time.perf_counter()
        kdtree_euclidean_dists_np, kdtree_indices_np = self.kdtree.query(query_embeddings_np, k=1)
        self.timing_details['kdtree_lookup_end'] = time.perf_counter()

        self.timing_details['kdtree_result_proc_start'] = time.perf_counter()
        # Results are CPU tensors
        kdtree_euclidean_dists = torch.from_numpy(kdtree_euclidean_dists_np.squeeze(-1)).to(embeddings.device, dtype=torch.float32)
        kdtree_indices = torch.from_numpy(kdtree_indices_np.squeeze(-1)).to(embeddings.device).long()

        scores = torch.full((embeddings.size(0), self.prototypes.shape[0]), float('-inf'), 
                            device=embeddings.device, dtype=embeddings.dtype) # dtype will be float32 on CPU

        final_dists = kdtree_euclidean_dists
        if self.dist == "cosine":
            cosine_dists = (kdtree_euclidean_dists.pow(2)) / 2.0
            final_dists = cosine_dists
        
        if self.ph is not None:
            final_dists = self.ph(final_dists) # Assuming ph handles float32
        
        if self.squared:
            final_dists = final_dists.pow(2)
        
        source_values = -final_dists.to(scores.dtype) # dtypes should match (both float32)
        scores[torch.arange(embeddings.size(0), device=embeddings.device), kdtree_indices] = source_values
        self.timing_details['kdtree_result_proc_end'] = time.perf_counter()
        return scores

    def forward(self, *input_data, **kwargs):
        self.timing_details = {}
        
        self.timing_details['backbone_start'] = time.perf_counter()
        embeddings = self.model(*input_data, **kwargs)
        self.timing_details['backbone_end'] = time.perf_counter()
        
        original_shape = embeddings.shape
        two_dim_data = False
        b, c_dim, h, w = -1,-1,-1,-1

        self.timing_details['reshape_start'] = time.perf_counter()
        if len(embeddings.shape) == 4:  
            two_dim_data = True
            b, c_dim, h, w = embeddings.shape
            embeddings_reshaped = (
                embeddings.view(b, c_dim, h * w)
                .transpose(1, 2)
                .contiguous()
                .view(b * h * w, c_dim)
            )
        else:
            embeddings_reshaped = embeddings
        self.timing_details['reshape_end'] = time.perf_counter()
        
        # Prototypes are on CPU if model is on CPU
        current_prototypes = self.prototypes.to(embeddings_reshaped.device) 

        self.timing_details['dist_calc_start'] = time.perf_counter()
        if not self.training and self.use_kdtree_for_inference:
            scores = self._forward_kdtree(embeddings_reshaped)
        else:
            current_prototypes_casted = current_prototypes.to(embeddings_reshaped.dtype) # Should be float32
            if self.dist == "cosine":
                sim = nn.CosineSimilarity(dim=-1)(
                    embeddings_reshaped[:, None, :], current_prototypes_casted[None, :, :]
                )
                dists = 1 - sim 
            else: 
                dists = torch.cdist(embeddings_reshaped, current_prototypes_casted, p=2)

            if self.ph is not None:
                dists = self.ph(dists)
            if self.squared: 
                dists = dists.pow(2)
            scores = -dists.to(embeddings_reshaped.dtype)
        self.timing_details['dist_calc_end'] = time.perf_counter()

        self.timing_details['unreshape_start'] = time.perf_counter()
        if two_dim_data:
            scores = (
                scores.view(b, h * w, self.prototypes.shape[0]) 
                .transpose(1, 2)
                .contiguous()
                .view(b, self.prototypes.shape[0], h, w)
            )
        self.timing_details['unreshape_end'] = time.perf_counter()
        return scores


def run_inference_benchmark_cpu(model, preprocessor, loader, device, num_benchmark_batches):
    """
    Runs inference for a few batches on CPU and collects detailed timings.
    AMP is disabled.
    """
    model.eval() # Ensure model is in evaluation mode
    all_timings = []

    # Warm-up run (one batch)
    try:
        waveforms, _ = next(iter(loader))
        waveforms = waveforms.to(device) # non_blocking=True not relevant for CPU
        with torch.no_grad(): # No AMP context for CPU
            _processed_audio = preprocessor(waveforms)
            _ = model(_processed_audio)
    except Exception as e:
        print(f"Warm-up failed: {e}")
        return []

    batch_iterator = iter(loader)
    print(f"Running CPU benchmark for {num_benchmark_batches} batches...")
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
                
                model_internal_timings = model.timing_details
                current_batch_timings["backbone"] = (model_internal_timings.get('backbone_end',0) - model_internal_timings.get('backbone_start',0)) * 1000
                current_batch_timings["reshape_pre"] = (model_internal_timings.get('reshape_end',0) - model_internal_timings.get('reshape_start',0)) * 1000
                
                if model.use_kdtree_for_inference:
                    current_batch_timings["kdtree_query_prep"] = (model_internal_timings.get('kdtree_query_prep_end',0) - model_internal_timings.get('kdtree_query_prep_start',0)) * 1000
                    current_batch_timings["kdtree_lookup"] = (model_internal_timings.get('kdtree_lookup_end',0) - model_internal_timings.get('kdtree_lookup_start',0)) * 1000
                    current_batch_timings["kdtree_result_proc"] = (model_internal_timings.get('kdtree_result_proc_end',0) - model_internal_timings.get('kdtree_result_proc_start',0)) * 1000
                    current_batch_timings["dist_calc_total_kdtree"] = (model_internal_timings.get('dist_calc_end',0) - model_internal_timings.get('dist_calc_start',0)) * 1000
                else:
                    current_batch_timings["dist_calc_standard"] = (model_internal_timings.get('dist_calc_end',0) - model_internal_timings.get('dist_calc_start',0)) * 1000
                
                current_batch_timings["unreshape_post"] = (model_internal_timings.get('unreshape_end',0) - model_internal_timings.get('unreshape_start',0)) * 1000
                current_batch_timings["full_model_call_external"] = (t_full_model_call_end - t_full_model_call_start) * 1000

            t_batch_end = time.perf_counter()
            current_batch_timings["total_batch_time"] = (t_batch_end - t_batch_start) * 1000
            all_timings.append(current_batch_timings)
        except StopIteration:
            print("DataLoader exhausted.")
            break
        except Exception as e:
            print(f"Error during CPU batch {batch_idx} processing: {e}")
            break
    return all_timings

def main_cpu(checkpoint_path, data_dir, batch_size, use_kdtree_arg, num_benchmark_batches):
    
    device = torch.device('cpu') # Force CPU
    print(f"Forcing CPU device for benchmark: {device}")

    if use_kdtree_arg and not SKLEARN_AVAILABLE:
        print("Error: --use_kdtree requires scikit-learn. KD-Tree mode disabled.")
        use_kdtree_arg = False
    
    # 1. Load Checkpoint and Instantiate Model
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path '{checkpoint_path}' does not exist.")
        return

    print(f"Loading checkpoint from '{checkpoint_path}' onto CPU...")
    checkpoint = torch.load(checkpoint_path, map_location=device) # Load directly to CPU
    embedding_dim = checkpoint.get('embedding_dim')
    num_classes_from_checkpoint = checkpoint.get('num_classes')

    if embedding_dim is None or num_classes_from_checkpoint is None:
        print("Error: Checkpoint missing 'embedding_dim' or 'num_classes'.")
        return

    preprocessor = AudioPreprocessor(sample_rate=TARGET_SAMPLE_RATE, n_mels=40, output_size=(40,40))
    preprocessor.to(device).eval() # Move preprocessor to CPU

    base_embedding_model = AudioEmbeddingNet(embedding_dim=embedding_dim, num_classes=None)
    # base_embedding_model will be moved to CPU when model is moved.
    
    model = LearntPrototypesWithTiming(
        model=base_embedding_model,
        n_prototypes=num_classes_from_checkpoint,
        embedding_dim=embedding_dim,
        dist=checkpoint.get('dist_metric', 'euclidean'),
        squared=checkpoint.get('squared_dist', False),
        ph=checkpoint.get('ph_delta'),
        device=device # Explicitly tell model its prototypes are on CPU
    )

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval() # Ensure entire model is on CPU

    # 2. Prepare DataLoader
    try:
        test_set = SpeechCommandsProcessedDataset(subset="testing")
    except FileNotFoundError as e:
        print(f"Error: Could not load test dataset. {e}")
        return

    # For CPU, num_workers=0 is often best to avoid IPC overhead, unless data loading is very heavy
    num_workers = 0 
    
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn # pin_memory, persistent_workers, prefetch_factor less relevant for CPU
    )

    # --- Run Benchmark for Standard O(N) on CPU ---
    print("\n--- Benchmarking Standard O(N) Inference on CPU ---")
    model.use_kdtree_for_inference = False
    timings_standard_cpu = run_inference_benchmark_cpu(model, preprocessor, test_loader, device, num_benchmark_batches)
    
    # --- Run Benchmark for KD-Tree on CPU (if enabled and possible) ---
    timings_kdtree_cpu = []
    if use_kdtree_arg:
        print("\n--- Benchmarking KD-Tree Inference on CPU ---")
        t_kdtree_build_start = time.perf_counter()
        try:
            # Prototypes are already on CPU when model is on CPU
            model.build_kdtree() 
            t_kdtree_build_end = time.perf_counter()
            print(f"KD-Tree build time (CPU): {(t_kdtree_build_end - t_kdtree_build_start) * 1000:.2f} ms")
            model.use_kdtree_for_inference = True
            timings_kdtree_cpu = run_inference_benchmark_cpu(model, preprocessor, test_loader, device, num_benchmark_batches)
        except ImportError:
            print("Skipping KD-Tree benchmark: scikit-learn not found during build_kdtree.")
        except Exception as e:
            print(f"Skipping KD-Tree benchmark due to error: {e}")

    # --- Report Results ---
    def aggregate_and_print_timings(label, timings_list):
        if not timings_list:
            print(f"\nNo timings recorded for {label}.")
            return None
        
        print(f"\n--- Aggregated CPU Timings for {label} (avg over {len(timings_list)} batches) ---")
        aggregated = {}
        for key in timings_list[0].keys(): # Assumes all dicts have same keys after first successful run
            valid_timings_for_key = [t[key] for t in timings_list if key in t]
            if valid_timings_for_key:
                 aggregated[key] = np.mean(valid_timings_for_key)
            else:
                aggregated[key] = 0 # Or some other placeholder if a key is missing in some runs

        sorted_keys = sorted([k for k in aggregated.keys() if k != "total_batch_time"]) + ["total_batch_time"]
        for key in sorted_keys:
            print(f"  {key:<25}: {aggregated[key]:.3f} ms")
        return aggregated

    print("\n" + "="*50)
    print("CPU Benchmark Summary")
    print("="*50)
    agg_standard_cpu = aggregate_and_print_timings("Standard O(N) [CPU]", timings_standard_cpu)
    
    if use_kdtree_arg and SKLEARN_AVAILABLE:
        agg_kdtree_cpu = aggregate_and_print_timings("KD-Tree [CPU]", timings_kdtree_cpu)
        if agg_standard_cpu and agg_kdtree_cpu and agg_kdtree_cpu.get("total_batch_time", 0) > 0:
            total_speedup = agg_standard_cpu.get("total_batch_time", 0) / agg_kdtree_cpu.get("total_batch_time", 0)
            dist_calc_key_standard = "dist_calc_standard"
            dist_calc_key_kdtree = "dist_calc_total_kdtree"
            
            if dist_calc_key_standard in agg_standard_cpu and dist_calc_key_kdtree in agg_kdtree_cpu and agg_kdtree_cpu[dist_calc_key_kdtree] > 0:
                 dist_calc_speedup = agg_standard_cpu[dist_calc_key_standard] / agg_kdtree_cpu[dist_calc_key_kdtree]
                 print(f"\n  Speedup (Distance Calc Only, CPU): {dist_calc_speedup:.2f}x")
            
            print(f"  Speedup (Total Batch Time, CPU): {total_speedup:.2f}x")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark CPU inference speed of LearntPrototypes.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the model checkpoint (.pt file).")
    parser.add_argument('--data_dir', type=str, default='./data/speech_commands_processed_v2',
                        help="Directory containing the preprocessed SpeechCommands dataset.")
    parser.add_argument('--batch_size', type=int, default=32, # Potentially smaller batch for CPU
                        help="Batch size for CPU evaluation.")
    parser.add_argument('--use_kdtree', action='store_true',
                        help="Enable KD-Tree mode for comparison on CPU.")
    parser.add_argument('--num_batches', type=int, default=50,
                        help="Number of batches from the test set to average timings over.")
    
    args = parser.parse_args()

    # For embedded, you might also want to set PyTorch threads:
    # torch.set_num_threads(1) # Or whatever is appropriate for your target
    # print(f"PyTorch using {torch.get_num_threads()} threads for CPU operations.")


    main_cpu(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_kdtree_arg=args.use_kdtree,
        num_benchmark_batches=args.num_batches
    )

    # Example Usage:
    # python benchmark_inference_cpu.py --checkpoint_path path/to/best_model.pt --num_batches 100
    # python benchmark_inference_cpu.py --checkpoint_path path/to/best_model.pt --num_batches 100 --use_kdtree