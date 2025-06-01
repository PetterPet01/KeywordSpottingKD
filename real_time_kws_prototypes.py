import os
import sys
import time
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import sounddevice as sd

# Attempt to import scipy.signal
try:
    import scipy.signal
except ImportError:
    print("ERROR: scipy is not installed. Please install it for FIR filter functionality.")
    sys.exit(1)

# Assuming dataset.py, model.py, etc., are accessible
try:
    from dataset import SpeechCommandsProcessedDataset
    from model import AudioEmbeddingNet
    from train_learnt_audio_prototypes import AudioPreprocessor
except ImportError as e:
    print(f"Error importing local modules (dataset, model, train_learnt_audio_prototypes): {e}")
    print("Please ensure these files are in the correct path or adjust imports.")
    print("Falling back to dummy implementations for critical components.")


    class SpeechCommandsProcessedDataset(torch.utils.data.Dataset):  # Dummy
        def __init__(self, subset="training", data_dir=None):
            print_once("Using DUMMY SpeechCommandsProcessedDataset.")
            self.data_dir = data_dir
            self.sample_rate = 16000
            if data_dir and os.path.exists(os.path.join(data_dir, "dummy_metadata.pt")):
                try:
                    dummy_meta = torch.load(os.path.join(data_dir, "dummy_metadata.pt"))
                    self.index_to_label = dummy_meta.get('index_to_label', {0: "_unknown_", 1: "yes", 2: "no"})
                    self.num_classes = len(self.index_to_label)
                    self.sample_rate = dummy_meta.get('sample_rate', 16000)
                    print_once(
                        f"Dummy dataset loaded dummy metadata: {self.num_classes} classes, SR {self.sample_rate}")
                except Exception as ex_meta:
                    print_once(f"Dummy dataset failed to load dummy_metadata.pt: {ex_meta}. Using hardcoded defaults.")
                    self.index_to_label = {0: "_unknown_", 1: "yes", 2: "no", 3: "up", 4: "down"}
                    self.num_classes = 5;
                    self.sample_rate = 16000
            else:
                print_once("Dummy dataset using hardcoded defaults (no dummy_metadata.pt or data_dir).")
                self.index_to_label = {0: "_unknown_", 1: "yes", 2: "no", 3: "up", 4: "down"}
                self.num_classes = 5;
                self.sample_rate = 16000
            self.len = 10

        def __len__(self):
            return self.len

        def __getitem__(self, idx):
            return torch.randn(self.sample_rate), 0


    class AudioPreprocessor(torch.nn.Module):  # Dummy
        def __init__(self, sample_rate=16000, n_mels=40, output_size=(40, 40), **kwargs):
            super().__init__();
            self.sample_rate = sample_rate;
            self.n_mels = n_mels;
            self.output_size = output_size
            print_once(f"Using DUMMY AudioPreprocessor (SR={sample_rate}, Mels={n_mels}, Size={output_size})")
            self.mel_spec_transform = T.MelSpectrogram(sample_rate, n_mels=n_mels, n_fft=400, hop_length=160)

        def forward(self, x_wav):
            mel_out = self.mel_spec_transform(x_wav).unsqueeze(1)
            b, c, h_in, w_current = mel_out.shape;
            h_target, w_target = self.output_size
            if w_current < w_target:
                mel_out = torch.nn.functional.pad(mel_out, (0, w_target - w_current))
            elif w_current > w_target:
                mel_out = mel_out[:, :, :, :w_target]
            if h_in != h_target: return torch.randn(b, 1, h_target, w_target, device=x_wav.device)
            return mel_out


    class AudioEmbeddingNet(torch.nn.Module):  # Dummy
        def __init__(self, embedding_dim=64, **kwargs):
            super().__init__();
            self.embedding_dim = embedding_dim
            print_once(f"Using DUMMY AudioEmbeddingNet (emb_dim={embedding_dim})")
            self.conv1 = torch.nn.Conv2d(1, 16, 3, 1, 1);
            self.relu = torch.nn.ReLU();
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1));
            self.fc = torch.nn.Linear(16, embedding_dim)

        def forward(self, x_spec): return self.fc(self.pool(self.relu(self.conv1(x_spec))).view(x_spec.size(0), -1))

try:
    from torch_prototypes.modules.prototypical_network import LearntPrototypes, SKLEARN_AVAILABLE
except ImportError:
    print("Error: Could not import LearntPrototypes. Using DUMMY.")
    SKLEARN_AVAILABLE = False


    class LearntPrototypes(torch.nn.Module):  # Dummy
        def __init__(self, model, n_prototypes, embedding_dim, dist="euclidean", squared=False, ph=None,
                     use_manual_distance=False, device="cpu", **kwargs):
            super().__init__();
            self.model = model;
            self.prototypes_param = torch.nn.Parameter(torch.randn(n_prototypes, embedding_dim, device=torch.device(
                device)))  # Use Parameter for device transfer
            self.embedding_dim = embedding_dim;
            self.n_prototypes = n_prototypes;
            self.use_kdtree_for_inference = False
            self.dist = dist;
            self.squared = squared;
            self.use_manual_distance = use_manual_distance
            print_once(
                f"Using DUMMY LearntPrototypes (dist={dist}, squared={squared}, manual_dist={use_manual_distance}). Prototypes on {self.prototypes_param.device}. KDTree: {SKLEARN_AVAILABLE}")

        @property  # Make prototypes behave like the real one for to(device) calls on parent
        def prototypes(self):
            return self.prototypes_param

        def forward(self, *input_data, **kwargs):  # input_data is preprocessor_output
            embeddings = self.model(*input_data, **kwargs)  # (B, embedding_dim)
            current_prototypes_casted = self.prototypes.to(embeddings.device,
                                                           embeddings.dtype)  # (N_proto, embedding_dim)

            if self.dist == "cosine":
                norm_embeddings = torch.linalg.norm(embeddings, dim=1, keepdim=True)
                norm_prototypes = torch.linalg.norm(current_prototypes_casted, dim=1, keepdim=True)
                normalized_embeddings = embeddings / (norm_embeddings + 1e-8)
                normalized_prototypes = current_prototypes_casted / (norm_prototypes + 1e-8)
                sim = torch.matmul(normalized_embeddings, normalized_prototypes.t())  # (B, N_proto)
                dists = 1 - sim
            else:
                sum_sq_embeddings = torch.sum(embeddings.pow(2), dim=1, keepdim=True)
                sum_sq_prototypes = torch.sum(current_prototypes_casted.pow(2), dim=1, keepdim=True).t()
                dot_product = torch.matmul(embeddings, current_prototypes_casted.t())
                dists_sq = sum_sq_embeddings - 2 * dot_product + sum_sq_prototypes
                dists_sq = torch.clamp(dists_sq, min=0.0)
                dists = torch.sqrt(dists_sq)

            final_dists_for_scores = dists.pow(2) if self.squared else dists
            scores = -final_dists_for_scores.to(embeddings.dtype)  # (B, N_proto)
            return scores

        def build_kdtree(self):
            print_once("Mock build_kdtree (Dummy LearntPrototypes)")

        def enable_kdtree_inference(self):
            if SKLEARN_AVAILABLE:
                self.build_kdtree(); self.use_kdtree_for_inference = True
            else:
                print_once("KD-Tree cannot be enabled: scikit-learn not available (Dummy LearntPrototypes).")

        def disable_kdtree_inference(self):
            self.use_kdtree_for_inference = False

# --- Configuration ---
WINDOW_DURATION_S = 1.0
STRIDE_DURATION_S = 0.25
DEFAULT_PROBABILITY_THRESHOLD = 0.7  # Changed from distance_threshold
DEFAULT_SOFTMAX_TEMPERATURE = 1.0
DEFAULT_TARGET_WAKE_WORDS = ["marvin", "sheila"]
DEFAULT_COOLDOWN_S = 1.5

audio_buffer_g = np.array([], dtype=np.float32)
base_embedding_model_g = None
learnt_prototypes_model_g = None
audio_preprocessor_g = None
device_g = None
labels_g = []
model_target_sr_g = 16000
model_window_samples_g = int(model_target_sr_g * WINDOW_DURATION_S)

_printed_messages_g = set()


def print_once(message):
    global _printed_messages_g
    if message not in _printed_messages_g: print(message); _printed_messages_g.add(message)


def load_prototype_kws_model_and_labels_from_data(
        checkpoint_path: str,
        data_dir: str,  # Still needed for fallback or if SR not in checkpoint
        device_str: str,
        use_kdtree: bool,
        use_manual_dist_flag: bool
):
    global model_target_sr_g, model_window_samples_g

    print(f"Loading model from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}");
        sys.exit(1)

    device = torch.device(device_str)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_labels = None
    num_classes_from_source = 0  # Can be from checkpoint or data_dir

    # --- Attempt to load labels and sample rate from CHECKPOINT first ---
    if 'index_to_label' in checkpoint and isinstance(checkpoint['index_to_label'], dict):
        index_to_label_from_ckpt = checkpoint['index_to_label']
        num_classes_from_source = len(index_to_label_from_ckpt)
        # Validate the loaded index_to_label
        if all(i in index_to_label_from_ckpt for i in range(num_classes_from_source)):
            class_labels = [index_to_label_from_ckpt[i] for i in range(num_classes_from_source)]
            print_once(f"SUCCESS: Loaded labels directly from checkpoint (first 5): {class_labels[:5]}...")
        else:
            print_once("WARNING: 'index_to_label' in checkpoint is malformed. Will attempt fallback.")
            class_labels = None  # Force fallback
    elif 'class_names_ordered' in checkpoint and isinstance(checkpoint['class_names_ordered'], list):
        class_labels = checkpoint['class_names_ordered']
        num_classes_from_source = len(class_labels)
        print_once(f"SUCCESS: Loaded 'class_names_ordered' directly from checkpoint (first 5): {class_labels[:5]}...")

    if 'sample_rate' in checkpoint:  # Also try to load sample_rate from checkpoint
        model_target_sr_g = checkpoint['sample_rate']
        model_window_samples_g = int(model_target_sr_g * WINDOW_DURATION_S)
        print_once(f"Model Target Sample Rate set from CHECKPOINT: {model_target_sr_g} Hz")
        print_once(f"Model Window Samples set to: {model_window_samples_g}")
    else:
        print_once("WARNING: 'sample_rate' not found in checkpoint. Will attempt to get from data_dir if needed.")

    # --- FALLBACK: If labels not in checkpoint, use data_dir ---
    if class_labels is None:
        print_once("Labels not successfully loaded from checkpoint. Falling back to data_dir method.")
        print_once(f"Ensure data_dir ('{data_dir}') metadata matches the training conditions of the checkpoint.")
        try:
            print(f"Attempting to load dataset metadata from: {data_dir} for labels/SR...")
            # We need to ensure SpeechCommandsProcessedDataset can be initialized without a subset if used only for metadata
            dataset_for_metadata = SpeechCommandsProcessedDataset(subset="validation")  # Or just data_dir=data_dir

            if model_target_sr_g == 16000 and 'sample_rate' not in checkpoint:  # Only update SR if not from checkpoint
                model_target_sr_g = 16000
                model_window_samples_g = int(model_target_sr_g * WINDOW_DURATION_S)
                print_once(f"Model Target Sample Rate set from DATA_DIR: {model_target_sr_g} Hz")
                print_once(f"Model Window Samples set to: {model_window_samples_g}")
            elif 'sample_rate' not in checkpoint:  # SR was not default and not in checkpoint
                print_once(f"Using model target SR previously set or defaulted: {model_target_sr_g} Hz")

            num_classes_from_source = dataset_for_metadata.num_classes
            index_to_label_from_data = dataset_for_metadata.index_to_label

            if not all(i in index_to_label_from_data for i in range(num_classes_from_source)):
                print(f"ERROR: index_to_label from data_dir ('{data_dir}') is incomplete. Cannot proceed.");
                sys.exit(1)

            class_labels = [index_to_label_from_data[i] for i in range(num_classes_from_source)]
            print(f"Loaded labels via data_dir fallback (first 5): {class_labels[:5]}...")
            del dataset_for_metadata
        except Exception as e:
            print(f"ERROR: Could not reconstruct labels/SR from data_dir ('{data_dir}') fallback. {e}");
            sys.exit(1)

    # --- Infer Model Dimensions from checkpoint (num_prototypes and embedding_dim) ---
    embedding_dim = checkpoint.get('embedding_dim')
    # num_classes in checkpoint's top-level can be n_prototypes
    num_prototypes_from_checkpoint_meta = checkpoint.get('num_classes')
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)

    if embedding_dim is None or num_prototypes_from_checkpoint_meta is None:
        proto_key_options = ['prototypes', 'prototypes_param']  # Check both for real and dummy state_dict
        found_proto_key = None
        for key_opt in proto_key_options:  # Try to find the actual prototype tensor
            # The prototype tensor might be nested under 'model.' if the base_embedding_model was saved as part of LearntPrototypes
            # and LearntPrototypes itself was saved.
            # Or it could be directly 'prototypes' if LearntPrototypes.state_dict() was saved.
            potential_keys_in_state_dict = [key_opt, f"model.{key_opt}"]  # Simplified search for now
            for pk in potential_keys_in_state_dict:
                if isinstance(model_state_dict, dict) and pk in model_state_dict and \
                        isinstance(model_state_dict[pk], torch.Tensor):
                    found_proto_key = pk
                    break
            if found_proto_key: break

        if found_proto_key:
            proto_shape = model_state_dict[found_proto_key].shape
            num_prototypes_from_checkpoint_meta = proto_shape[0]
            embedding_dim = proto_shape[1]
            print_once(
                f"Inferred num_prototypes ({num_prototypes_from_checkpoint_meta}) and embedding_dim ({embedding_dim}) from '{found_proto_key}' tensor in state_dict.")
        else:
            print(
                "ERROR: Checkpoint missing 'embedding_dim' or 'num_classes', and cannot infer from 'prototypes' tensor in state_dict.");
            sys.exit(1)

    # --- Validate Consistency between loaded labels and model structure ---
    if num_prototypes_from_checkpoint_meta != len(class_labels):
        print(f"CRITICAL ERROR: Number of prototypes in checkpoint's structure ({num_prototypes_from_checkpoint_meta}) "
              f"does not match number of classes derived from label source ({len(class_labels)}).")
        print("This indicates a fundamental mismatch. The KWS will likely misinterpret predictions.")
        print(
            f"Prototypes in model: {num_prototypes_from_checkpoint_meta}. Labels loaded: {len(class_labels)} derived from {'checkpoint' if 'index_to_label' in checkpoint or 'class_names_ordered' in checkpoint else 'data_dir'}.")
        sys.exit(1)

    # --- Initialize Model Components ---
    expected_spec_height = 40
    expected_spec_width = 40
    # Try to get preprocessor params from checkpoint if saved
    pp_config = checkpoint.get('preprocessor_config', {})
    expected_spec_height = pp_config.get('n_mels', expected_spec_height)
    pp_output_size_h = pp_config.get('output_size_h', expected_spec_height)
    pp_output_size_w = pp_config.get('output_size_w', expected_spec_width)

    print_once(
        f"AudioPreprocessor params (ckpt/default): n_mels={expected_spec_height}, output_size=({pp_output_size_h},{pp_output_size_w})")
    audio_preprocessor = AudioPreprocessor(
        sample_rate=model_target_sr_g,
        n_mels=expected_spec_height,
        output_size=(pp_output_size_h, pp_output_size_w)  # Use potentially loaded params
    ).to(device).eval()

    base_embedding_model = AudioEmbeddingNet(embedding_dim=embedding_dim).to(device)

    model_config = checkpoint.get('model_config', {})  # For LearntPrototypes params
    dist_param = model_config.get('dist', 'euclidean')
    squared_param = model_config.get('squared', False)
    ph_delta_param = model_config.get('ph', None)
    print_once(
        f"LearntPrototypes config (ckpt/default): dist='{dist_param}', squared={squared_param}, ph={ph_delta_param}, manual_dist_cli={use_manual_dist_flag}")

    learnt_prototypes_model = LearntPrototypes(
        model=base_embedding_model,
        n_prototypes=num_prototypes_from_checkpoint_meta,  # Use count from model structure
        embedding_dim=embedding_dim,
        dist=dist_param, squared=squared_param, ph=ph_delta_param,
        use_manual_distance=use_manual_dist_flag, device=device_str
    )

    try:
        learnt_prototypes_model.load_state_dict(model_state_dict, strict=True)
    except RuntimeError as e_strict:
        print_once(f"Warning: load_state_dict with strict=True failed ('{e_strict}'). Trying strict=False.")
        try:
            learnt_prototypes_model.load_state_dict(model_state_dict, strict=False)
        except RuntimeError as e_non_strict:
            print(f"Error loading state_dict even with strict=False: {e_non_strict}");
            sys.exit(1)
    learnt_prototypes_model.to(device).eval()

    # KD-Tree setup (remains the same logic)
    if use_kdtree:
        if SKLEARN_AVAILABLE and hasattr(learnt_prototypes_model, 'enable_kdtree_inference'):
            print_once("Enabling KD-Tree for inference as requested.")
            learnt_prototypes_model.enable_kdtree_inference()
            if not learnt_prototypes_model.use_kdtree_for_inference:
                print_once("Warning: KD-Tree was requested but could not be enabled by model.")
        else:
            print_once("Warning: KD-Tree requested, but scikit-learn unavailable or model doesn't support it.")
    elif hasattr(learnt_prototypes_model, 'disable_kdtree_inference'):
        learnt_prototypes_model.disable_kdtree_inference()
        print_once("KD-Tree is NOT enabled for inference.")

    kdtree_active = hasattr(learnt_prototypes_model,
                            'use_kdtree_for_inference') and learnt_prototypes_model.use_kdtree_for_inference
    if hasattr(learnt_prototypes_model,
               'use_manual_distance') and learnt_prototypes_model.use_manual_distance and not kdtree_active:
        print_once("Note: Manual distance calculation will be used in LearntPrototypes (KD-Tree is off).")
    elif kdtree_active:
        print_once(
            "Note: KD-Tree is active; 'use_manual_distance' flag is ignored by LearntPrototypes when KD-Tree is on.")

    print("Model and labels loaded successfully.")
    return audio_preprocessor, base_embedding_model, learnt_prototypes_model, device, class_labels

def preprocess_audio_chunk_rt(
        waveform_np: np.ndarray, current_sample_rate: int,
        audio_preprocessor_model: torch.nn.Module, device_tensor: torch.device
):
    global model_target_sr_g, model_window_samples_g
    waveform_torch_resampled: torch.Tensor
    if current_sample_rate == 44100 and model_target_sr_g == 16000:
        print_once(f"Input SR {current_sample_rate}Hz->{model_target_sr_g}Hz: FIR LPF+resample.")
        cutoff_hz = 8000.0;
        numtaps = 121
        fir_coeffs = scipy.signal.firwin(numtaps, cutoff_hz, fs=current_sample_rate, pass_zero='lowpass')
        waveform_filtered_np = scipy.signal.filtfilt(fir_coeffs, 1.0, waveform_np.astype(np.float64))
        w_filt_torch = torch.from_numpy(waveform_filtered_np.copy()).float().unsqueeze(0)
        resampler = T.Resample(orig_freq=current_sample_rate, new_freq=model_target_sr_g,
                               resampling_method="sinc_interp_hann")
        waveform_torch_resampled = resampler(w_filt_torch)
    elif current_sample_rate != model_target_sr_g:
        print_once(f"Input SR {current_sample_rate}Hz->{model_target_sr_g}Hz: torchaudio resample.")
        w_orig_torch = torch.from_numpy(waveform_np).float().unsqueeze(0)
        resampler = T.Resample(orig_freq=current_sample_rate, new_freq=model_target_sr_g)
        waveform_torch_resampled = resampler(w_orig_torch)
    else:
        print_once(f"Input SR {current_sample_rate}Hz matches target. No resampling.")
        waveform_torch_resampled = torch.from_numpy(waveform_np).float().unsqueeze(0)

    num_samples_after_resample = waveform_torch_resampled.shape[1]
    if num_samples_after_resample < model_window_samples_g:
        pad_amount = model_window_samples_g - num_samples_after_resample
        waveform_torch_windowed = torch.nn.functional.pad(waveform_torch_resampled, (0, pad_amount))
    elif num_samples_after_resample > model_window_samples_g:
        waveform_torch_windowed = waveform_torch_resampled[:, :model_window_samples_g]
    else:
        waveform_torch_windowed = waveform_torch_resampled
    return audio_preprocessor_model(waveform_torch_windowed.to(device_tensor))


def audio_callback(indata, frames, time_info, status):
    global audio_buffer_g
    if status: print(status, file=sys.stderr)
    audio_buffer_g = np.concatenate((audio_buffer_g, indata[:, 0]))


def main_loop(args):
    global audio_buffer_g, learnt_prototypes_model_g, audio_preprocessor_g, device_g, labels_g
    # base_embedding_model_g is part of learnt_prototypes_model_g

    audio_preprocessor_g, _, learnt_prototypes_model_g, device_g, labels_g = \
        load_prototype_kws_model_and_labels_from_data(
            args.checkpoint_path, args.data_dir, args.device,
            args.use_kdtree, args.use_manual_distance
        )

    target_wakeword_indices = []
    target_wakewords_lower = [w.lower() for w in args.wakewords]
    for wakeword_target in target_wakewords_lower:
        try:
            idx = labels_g.index(wakeword_target);
            target_wakeword_indices.append(idx)
        except ValueError:
            print(f"Warn: Target '{wakeword_target}' not in labels. Ignoring.")
    if not target_wakeword_indices: print("ERROR: No valid target wake words. Exiting."); sys.exit(1)
    print(
        f"Listening for (indices): {target_wakeword_indices} (labels: {[labels_g[i] for i in target_wakeword_indices]})")

    mic_actual_samplerate = args.mic_samplerate
    mic_window_duration_samples = int(mic_actual_samplerate * WINDOW_DURATION_S)
    mic_stride_duration_samples = int(mic_actual_samplerate * STRIDE_DURATION_S)

    print(f"Mic SR: {mic_actual_samplerate}Hz. Model Target SR: {model_target_sr_g}Hz.")
    print(f"Processing {WINDOW_DURATION_S}s chunks ({model_window_samples_g} samples post-resample).")
    print(f"Stride: {STRIDE_DURATION_S}s ({mic_stride_duration_samples} samples at mic SR).")
    print(
        f"Probability thresh: {args.probability_threshold}. Softmax Temp: {args.softmax_temperature}. Cooldown: {args.cooldown_duration}s.")
    kdtree_stat = "Unknown"
    if hasattr(learnt_prototypes_model_g, 'use_kdtree_for_inference'):
        kdtree_stat = "Enabled" if learnt_prototypes_model_g.use_kdtree_for_inference else "Disabled"
    print(f"KD-Tree: {kdtree_stat}")
    print("Press Ctrl+C to stop.")

    last_detection_time = 0.0
    try:
        with sd.InputStream(device=args.mic_id, channels=1, samplerate=mic_actual_samplerate,
                            callback=audio_callback, blocksize=mic_stride_duration_samples):
            while True:
                if len(audio_buffer_g) >= mic_window_duration_samples:
                    current_window_audio_mic_sr = audio_buffer_g[:mic_window_duration_samples].copy()
                    audio_buffer_g = audio_buffer_g[mic_stride_duration_samples:]

                    model_input_spectrogram = preprocess_audio_chunk_rt(
                        current_window_audio_mic_sr, mic_actual_samplerate,
                        audio_preprocessor_g, device_g
                    )
                    with torch.no_grad():
                        scores = learnt_prototypes_model_g(model_input_spectrogram)  # Shape (1, N_prototypes)

                        # Apply softmax to get probabilities
                        # Scores are logits; higher scores mean "closer" / more likely
                        probabilities = torch.softmax(scores.squeeze(0) / args.softmax_temperature,
                                                      dim=-1)  # Shape (N_prototypes)

                    max_prob_val, pred_proto_idx_tensor = torch.max(probabilities, dim=0)

                    pred_proto_idx = pred_proto_idx_tensor.item()
                    pred_label = labels_g[pred_proto_idx]
                    current_max_prob = max_prob_val.item()

                    # print(pred_label)

                    if current_max_prob >= args.probability_threshold:
                        curr_time = time.time()
                        if curr_time - last_detection_time >= args.cooldown_duration:
                            print(f"**** DETECTED '{pred_label}' (Proto Idx {pred_proto_idx}) "
                                  f"Prob: {current_max_prob:.3f} at {time.strftime('%Y-%m-%d %H:%M:%S')} ****")
                            last_detection_time = curr_time
                        elif args.verbose:
                            print_once(f"Detected '{pred_label}' (Prob: {current_max_prob:.3f}) but in cooldown.")
                    elif args.verbose:
                        print(f"  Max prob for '{pred_label}' (Idx {pred_proto_idx}): {current_max_prob:.2f}", end='\r')

                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}"); import traceback; traceback.print_exc()
    finally:
        print("Exiting KWS application.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time KWS with LearntPrototypes using probability thresholding.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to .pt model checkpoint.")
    parser.add_argument("--data_dir", type=str, default="./data/speech_commands_processed_v2",
                        help="Path to processed SpeechCommands dataset directory for labels and target SR.")
    parser.add_argument("--wakewords", type=str, nargs='+', default=DEFAULT_TARGET_WAKE_WORDS,
                        help="Target wake words.")
    # Changed from distance_threshold to probability_threshold
    parser.add_argument("--probability_threshold", type=float, default=DEFAULT_PROBABILITY_THRESHOLD,
                        help="Minimum probability for a positive detection.")
    parser.add_argument("--softmax_temperature", type=float, default=DEFAULT_SOFTMAX_TEMPERATURE,
                        help="Temperature for softmax scaling of scores.")
    parser.add_argument("--cooldown_duration", type=float, default=DEFAULT_COOLDOWN_S, help="Detection cooldown (s).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device.")
    parser.add_argument("--mic_id", type=int, default=None, help="Mic ID ('python -m sounddevice').")
    parser.add_argument("--mic_samplerate", type=int, default=16000, help="Mic sample rate.")
    parser.add_argument("--use_kdtree", action="store_true", help="Enable KD-Tree (if available and supported).")
    parser.add_argument("--use_manual_distance", action="store_true",
                        help="Force manual distance in LearntPrototypes (ignored if KD-Tree is active).")
    parser.add_argument("--verbose", action="store_true", help="Print all max probabilities, not just wake words.")
    cli_args = parser.parse_args()

    try:
        sd.query_devices()
    except Exception as e:
        print(f"Sounddevice issue: {e}. Check install & mic."); sys.exit(1)

    main_loop(cli_args)