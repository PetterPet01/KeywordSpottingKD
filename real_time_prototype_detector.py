# real_time_prototype_detector.py
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import sounddevice as sd

# Attempt to import scipy.signal and exit if not found
try:
    import scipy.signal
except ImportError:
    print(
        "ERROR: scipy is not installed. Please install it using 'pip install scipy' to use the FIR filter functionality.")
    sys.exit(1)

# --- Import project-specific modules ---
try:
    # Assuming get_data.py is in a 'Utils' subdirectory.
    # If get_data.py is in the same directory as this script, change to:
    # import get_data as get_data_utils
    # If your project structure is different, adjust this import path.
    import get_data as get_data_utils

    MODEL_TARGET_SAMPLE_RATE = get_data_utils.SAMPLE_RATE
    N_CLASS_FROM_GET_DATA = get_data_utils.N_CLASS
    DEFAULT_LABELS_FROM_GET_DATA = get_data_utils.DEFAULT_LABELS
    # We'll use the list DEFAULT_LABELS_FROM_GET_DATA for indexing,
    # but having idx_to_label can be a fallback or alternative.
    # IDX_TO_LABEL_FROM_GET_DATA = get_data_utils.idx_to_label

    from model import AudioEmbeddingNet  # Base embedding model
    from train_learnt_audio_prototypes import AudioPreprocessor  # Re-use the preprocessor
    from torch_prototypes.modules.prototypical_network import LearntPrototypes, SKLEARN_AVAILABLE
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure:")
    print(
        "1. 'get_data.py' is accessible (e.g., in a 'Utils' subdirectory or current directory, adjust import path if needed).")
    print(
        "2. 'model.py', 'train_learnt_audio_prototypes.py' are accessible (e.g., in the current directory or PYTHONPATH).")
    print("3. 'torch_prototypes' library is installed.")
    sys.exit(1)

# --- Configuration ---
# MODEL_TARGET_SAMPLE_RATE is now set from get_data_utils.SAMPLE_RATE
WINDOW_DURATION_S = 1.0
MODEL_WINDOW_SAMPLES = int(MODEL_TARGET_SAMPLE_RATE * WINDOW_DURATION_S)

STRIDE_DURATION_S = 0.25
PROBABILITY_THRESHOLD = 0.7
DEFAULT_TARGET_WAKE_WORDS = ["marvin", "sheila"]
DEFAULT_COOLDOWN_S = 1.0

# Global buffer for audio
audio_buffer = np.array([], dtype=np.float32)
model_loaded = None
preprocessor_loaded = None
device_g = None
class_names_loaded = None  # Will be populated with DEFAULT_LABELS_FROM_GET_DATA

_printed_messages = set()


def print_once(message):
    global _printed_messages
    if message not in _printed_messages:
        print(message)
        _printed_messages.add(message)


def load_prototype_model_and_preprocessor(checkpoint_path, device_str, use_kdtree_if_available=False):
    """Loads the LearntPrototypes model and its AudioPreprocessor."""
    global class_names_loaded  # Allow modification of this global
    print(f"Loading checkpoint from: {checkpoint_path}")
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if use_kdtree_if_available and not SKLEARN_AVAILABLE:
        print("Warning: --use_kdtree requires scikit-learn, but it's not installed. Disabling KD-Tree.")
        use_kdtree_if_available = False

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path '{checkpoint_path}' does not exist.")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    embedding_dim = checkpoint.get('embedding_dim')
    num_classes_from_checkpoint = checkpoint.get('num_classes')
    # `saved_class_names` is no longer read from checkpoint

    if embedding_dim is None or num_classes_from_checkpoint is None:
        print("Error: Checkpoint missing 'embedding_dim' or 'num_classes'. Cannot re-instantiate model.")
        sys.exit(1)

    # Validate num_classes from checkpoint against N_CLASS from get_data.py
    if num_classes_from_checkpoint != N_CLASS_FROM_GET_DATA:
        print(f"Warning: Number of classes in checkpoint ({num_classes_from_checkpoint}) "
              f"does not match N_CLASS from get_data.py ({N_CLASS_FROM_GET_DATA}).")
        print("The model's output logits will be mapped to labels based on the order in get_data.DEFAULT_LABELS.")
        print(
            "Ensure the model was trained with these exact classes in this specific order for correct interpretation.")
        # If a strict match is required, you could sys.exit(1) here.

    print(f"Model parameters from checkpoint: embedding_dim={embedding_dim}, num_classes={num_classes_from_checkpoint}")
    print(
        f"Using class names and order from get_data.py (Total: {N_CLASS_FROM_GET_DATA} classes). First few: {DEFAULT_LABELS_FROM_GET_DATA[:5]}")

    preprocessor = AudioPreprocessor(
        sample_rate=MODEL_TARGET_SAMPLE_RATE,
        n_mels=40,
        output_size=(40, 40)
    )
    preprocessor.to(device)
    preprocessor.eval()

    base_embedding_model = AudioEmbeddingNet(embedding_dim=embedding_dim, num_classes=None)

    model = LearntPrototypes(
        model=base_embedding_model,
        n_prototypes=num_classes_from_checkpoint,  # Use num_classes from checkpoint for model structure
        embedding_dim=embedding_dim,
        dist=checkpoint.get('dist_metric', 'euclidean'),
        squared=checkpoint.get('squared_dist', False),
        ph=checkpoint.get('ph_delta'),
        device=device
    )

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    if use_kdtree_if_available:
        print("Building KD-Tree for inference...")
        model.build_kdtree()
        model.use_kdtree_for_inference = True

    print("Model and preprocessor loaded successfully.")
    # Set the global class_names_loaded to the list from get_data.py
    class_names_loaded = DEFAULT_LABELS_FROM_GET_DATA
    return model, preprocessor, device  # Removed class_names from return as it's global now


def resample_and_prepare_waveform(
        waveform_np: np.ndarray,
        current_sample_rate: int,
        target_sample_rate: int,
        target_num_samples: int
) -> torch.Tensor:
    # ... (This function remains unchanged from the previous version) ...
    waveform_torch_resampled: torch.Tensor

    if current_sample_rate == 44100 and target_sample_rate == 16000:
        msg = (f"Input SR {current_sample_rate}Hz: Applying custom FIR LPF (8kHz cutoff) "
               f"then resampling to {target_sample_rate}Hz.")
        print_once(msg)
        cutoff_hz = 8000.0
        numtaps = 121
        fir_coeffs = scipy.signal.firwin(numtaps, cutoff_hz, fs=current_sample_rate, pass_zero='lowpass')
        waveform_filtered_np = scipy.signal.filtfilt(fir_coeffs, 1.0, waveform_np.astype(np.float64))
        waveform_torch_filtered_orig_sr = torch.from_numpy(waveform_filtered_np.copy()).float().unsqueeze(0)

        resampler = T.Resample(orig_freq=current_sample_rate, new_freq=target_sample_rate,
                               resampling_method="sinc_interp_hann")
        waveform_torch_resampled = resampler(waveform_torch_filtered_orig_sr)

    elif current_sample_rate != target_sample_rate:
        msg = (f"Input SR {current_sample_rate}Hz: Applying standard torchaudio resampling "
               f"to {target_sample_rate}Hz.")
        print_once(msg)
        waveform_torch_orig_sr = torch.from_numpy(waveform_np).float().unsqueeze(0)
        resampler = T.Resample(orig_freq=current_sample_rate, new_freq=target_sample_rate)
        waveform_torch_resampled = resampler(waveform_torch_orig_sr)
    else:
        msg = (f"Input SR {current_sample_rate}Hz matches target {target_sample_rate}Hz. "
               f"No resampling needed.")
        print_once(msg)
        waveform_torch_resampled = torch.from_numpy(waveform_np).float().unsqueeze(0)

    num_samples_after_resample = waveform_torch_resampled.shape[1]
    if num_samples_after_resample < target_num_samples:
        padding = target_num_samples - num_samples_after_resample
        waveform_final = torch.nn.functional.pad(waveform_torch_resampled, (0, padding))
    elif num_samples_after_resample > target_num_samples:
        waveform_final = waveform_torch_resampled[:, :target_num_samples]
    else:
        waveform_final = waveform_torch_resampled

    return waveform_final


def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    if status:
        print(status, file=sys.stderr)
    audio_buffer = np.concatenate((audio_buffer, indata[:, 0]))


def main(args):
    global audio_buffer, model_loaded, preprocessor_loaded, class_names_loaded, device_g

    # class_names_loaded will be set by load_prototype_model_and_preprocessor
    model_loaded, preprocessor_loaded, device_g = \
        load_prototype_model_and_preprocessor(args.checkpoint_path, args.device, args.use_kdtree)

    target_labels = [word.lower() for word in args.wakewords]
    print(f"Target wake words: {target_labels}")

    # Validate wakewords against the class_names_loaded (which are DEFAULT_LABELS_FROM_GET_DATA)
    if class_names_loaded is None:  # Should not happen if loading is successful
        print("Error: Class names not loaded. Exiting.")
        sys.exit(1)

    for label in target_labels:
        if label not in class_names_loaded:
            print(
                f"Warning: Target wake word '{label}' is not in the model's known class list derived from get_data.py.")
            print(f"Available classes are: {class_names_loaded}")

    mic_actual_samplerate = args.mic_samplerate
    mic_window_duration_samples = int(mic_actual_samplerate * WINDOW_DURATION_S)
    mic_stride_duration_samples = int(mic_actual_samplerate * STRIDE_DURATION_S)

    print(f"Listening for wake words: {target_labels} with threshold {args.threshold}...")
    print(f"Detection cooldown: {args.cooldown_duration} seconds.")
    print(f"Using microphone: {args.mic_id if args.mic_id is not None else 'default'}")
    print(f"Mic sample rate: {mic_actual_samplerate} Hz. Model target SR: {MODEL_TARGET_SAMPLE_RATE} Hz.")
    print(f"Processing {WINDOW_DURATION_S}s audio chunks, striding by {STRIDE_DURATION_S}s.")
    print(f"Mic window: {mic_window_duration_samples} samples. Mic stride: {mic_stride_duration_samples} samples.")
    print(f"Model expects {MODEL_WINDOW_SAMPLES} samples at {MODEL_TARGET_SAMPLE_RATE} Hz.")
    print("Press Ctrl+C to stop.")

    last_detection_time = 0.0

    try:
        with sd.InputStream(
                device=args.mic_id,
                channels=1,
                samplerate=mic_actual_samplerate,
                callback=audio_callback,
                blocksize=mic_stride_duration_samples
        ):
            while True:
                if len(audio_buffer) >= mic_window_duration_samples:
                    current_window_audio_mic_sr_np = audio_buffer[:mic_window_duration_samples].copy()
                    audio_buffer = audio_buffer[mic_stride_duration_samples:]

                    waveform_torch_model_input = resample_and_prepare_waveform(
                        current_window_audio_mic_sr_np,
                        current_sample_rate=mic_actual_samplerate,
                        target_sample_rate=MODEL_TARGET_SAMPLE_RATE,
                        target_num_samples=MODEL_WINDOW_SAMPLES
                    ).to(device_g)

                    processed_features = preprocessor_loaded(waveform_torch_model_input)

                    with torch.no_grad():
                        logits = model_loaded(processed_features)

                    probabilities = torch.softmax(logits, dim=1).squeeze()
                    top_prob, top_idx_tensor = torch.max(probabilities, dim=0)
                    top_idx = top_idx_tensor.item()

                    predicted_label = "unknown"  # Default
                    if 0 <= top_idx < len(class_names_loaded):
                        predicted_label = class_names_loaded[top_idx]
                    else:
                        print_once(
                            f"Warning: Predicted index {top_idx} is out of bounds for class_names_loaded (size {len(class_names_loaded)}).")

                    if top_prob.item() >= args.threshold:
                        current_time = time.time()
                        if current_time - last_detection_time >= args.cooldown_duration:
                            print(
                                f"Detected '{predicted_label}' (Prob: {top_prob.item():.3f}) at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                            last_detection_time = current_time
                    elif args.verbose:
                        print(f"  Top: '{predicted_label}' ({top_prob.item():.2f})", end='\r')

                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time wake word detection using prototype-based model.")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the .pt model checkpoint file."
    )
    parser.add_argument(
        "--wakewords",
        type=str,
        nargs='+',
        default=DEFAULT_TARGET_WAKE_WORDS,
        help="One or more wake words to detect (space-separated)."
    )
    parser.add_argument(
        "--threshold", type=float, default=PROBABILITY_THRESHOLD, help="Detection probability threshold."
    )
    parser.add_argument(
        "--cooldown_duration",
        type=float,
        default=DEFAULT_COOLDOWN_S,
        help="Minimum time in seconds between consecutive detections."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        choices=['cuda', 'cpu'], help="Device to use ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--mic_id", type=int, default=None, help="Microphone ID (integer). List with 'python -m sounddevice'."
    )
    parser.add_argument(
        "--mic_samplerate", type=int, default=MODEL_TARGET_SAMPLE_RATE,
        help=f"Sample rate of the microphone. Audio will be resampled. "
             f"Set to 44100 for custom FIR filtering if mic is 44.1kHz. "
             f"Default: {MODEL_TARGET_SAMPLE_RATE} Hz."
    )
    parser.add_argument(
        "--stride_duration_s", type=float, default=STRIDE_DURATION_S,
        help="How often to run inference, in seconds (stride of the sliding window)."
    )
    parser.add_argument(
        "--use_kdtree", action='store_true',
        help="Use KD-Tree for faster inference with LearntPrototypes (if scikit-learn is available)."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print all top predictions, not just the target."
    )

    cli_args = parser.parse_args()

    STRIDE_DURATION_S = cli_args.stride_duration_s

    try:
        sd.query_devices()
    except Exception as e:
        print(
            "Sounddevice issue. Do you have it installed ('pip install sounddevice soundfile') and a microphone connected?")
        print(f"Error details: {e}")
        sys.exit(1)

    main(cli_args)