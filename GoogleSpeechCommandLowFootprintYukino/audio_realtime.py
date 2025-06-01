# ---- real_time_detector.py ----
import os
import sys
import time
import argparse
import numpy as np
import torch
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

# Assuming these files are in the same directory or in PYTHONPATH
# Adjust these paths if your project structure is different
# For example, if Utils and Models are subdirectories of the current script's location:
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # If script is in a subdir like 'scripts'
# Or if they are in the same directory as this script, no sys.path modification is needed.
# For this example, I'll assume they are in Utils/ and Models/ relative to where this script is run.
# If you run this script from the project root (e.g. GoogleSpeechCommandLowFootprintYukino/), and
# your files are GoogleSpeechCommandLowFootprintYukino/Utils/util.py etc., then:
# import Utils.util as util
# import Utils.get_data as get_data
# from Models.bc_resnet_model_quaterionic import QBcResNetModel

# Corrected imports assuming the script is run from the project root
# and Utils and Models are top-level directories.
# If your structure is different (e.g., this script is *inside* a `scripts` folder,
# and Utils/Models are at the same level as `scripts`), you might need:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Then the imports would be:
import Utils.util as util
import Utils.get_data as get_data
from Models.bc_resnet_model_quaterionic import QBcResNetModel


# If you have quaternion_layers.py and subspectral_norm.py, ensure they are importable.
# (These are dependencies of QBcResNetModel if it uses QuaternionConv2d or SubSpectralNorm)

# --- Configuration ---
MODEL_PATH = "QBCResnet6QORGASM.torch"  # Default model path
MODEL_TARGET_SAMPLE_RATE = get_data.SAMPLE_RATE  # 16000 Hz, from get_data.py (used by the model)
WINDOW_DURATION_S = 1.0  # Process 1-second chunks (matches GSC dataset)
# MODEL_WINDOW_SAMPLES is the number of samples expected by the model after resampling
MODEL_WINDOW_SAMPLES = int(MODEL_TARGET_SAMPLE_RATE * WINDOW_DURATION_S)

STRIDE_DURATION_S = 0.25  # Check every 0.25 seconds
PROBABILITY_THRESHOLD = 0.7  # Minimum probability to consider a detection
DEFAULT_TARGET_WAKE_WORDS = ["marvin", "sheila"]  # Example wake words, change as needed

# Global buffer for audio
audio_buffer = np.array([], dtype=np.float32)
model_loaded = None
device_g = None

# Helper for printing messages only once
_printed_messages = set()


def print_once(message):
    global _printed_messages
    if message not in _printed_messages:
        print(message)
        _printed_messages.add(message)


def load_kws_model(model_path, device_str):
    """Loads the pre-trained KWS model."""
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        sys.exit(1)

    device = torch.device(device_str)

    # Instantiate the model with parameters used during training
    # (taken from custom_main_asm2.py for QBcResNetModelASM, applied to QBcResNetModel)
    model = QBcResNetModel(
        n_class=get_data.N_CLASS,  # Should be 35
        scale=6,
        dropout=0.2,
        use_subspectral=True # This depends on whether SubSpectralNorm was used in training
    ).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This might be due to a mismatch in model architecture or saved keys.")
        print("Ensure 'quaternion_layers.py' and 'subspectral_norm.py' are correct and accessible.")
        sys.exit(1)

    model.eval()
    print("Model loaded successfully.")
    return model, device


def preprocess_audio_chunk(waveform_np: np.ndarray, current_sample_rate: int, target_sample_rate: int,
                           device: torch.device):
    """
    Prepares a single audio chunk for the model.
    Input waveform_np is a 1D numpy array (audio at current_sample_rate).
    Output is a batched tensor ready for the model (B, 4, H, W).
    """

    # 1. FIR Filtering (if specific conditions met) and Resampling
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
                               resampling_method="sinc_interp_hann") # Updated from sinc_interpolation
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

    waveform_torch: torch.Tensor
    num_samples_after_resample = waveform_torch_resampled.shape[1]
    if num_samples_after_resample < MODEL_WINDOW_SAMPLES:
        padding = MODEL_WINDOW_SAMPLES - num_samples_after_resample
        waveform_torch = torch.nn.functional.pad(waveform_torch_resampled, (0, padding))
    elif num_samples_after_resample > MODEL_WINDOW_SAMPLES:
        waveform_torch = waveform_torch_resampled[:, :MODEL_WINDOW_SAMPLES]
    else:
        waveform_torch = waveform_torch_resampled

    to_mel = T.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=1024,
        f_max=8000,
        n_mels=40
    )
    mel_spec = to_mel(waveform_torch)
    log_mel = (mel_spec + get_data.EPS).log2()

    log_mel_batched_channel = log_mel.unsqueeze(1)
    log_mel_batched_channel = log_mel_batched_channel.to(device)

    zero_matrix = torch.zeros_like(log_mel_batched_channel).to(device)
    data_first_delta = torchaudio.functional.compute_deltas(log_mel_batched_channel)
    data_second_delta = torchaudio.functional.compute_deltas(data_first_delta)

    quaternion_input = torch.cat([zero_matrix, log_mel_batched_channel, data_first_delta, data_second_delta], dim=1)

    return quaternion_input


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    global audio_buffer
    if status:
        print(status, file=sys.stderr)
    audio_buffer = np.concatenate((audio_buffer, indata[:, 0]))


def main(args):
    global audio_buffer, model_loaded, device_g

    model_loaded, device_g = load_kws_model(args.model_path, args.device)

    target_labels = [word.lower() for word in args.wakewords]
    for label in target_labels:
        if label not in get_data.DEFAULT_LABELS:
            print(f"Warning: Target wake word '{label}' is not in the default label list.")
            print(f"Available labels are: {get_data.DEFAULT_LABELS}")
            # Continue anyway, user might have a custom label set implicitly handled by N_CLASS

    mic_actual_samplerate = args.mic_samplerate
    mic_window_duration_samples = int(mic_actual_samplerate * WINDOW_DURATION_S)
    mic_stride_duration_samples = int(mic_actual_samplerate * STRIDE_DURATION_S)

    print(f"Listening for wake words: {target_labels} with threshold {args.threshold}...")
    print(f"Using microphone: {args.mic_id if args.mic_id is not None else 'default'}")
    print(f"Mic sample rate: {mic_actual_samplerate} Hz. Model target SR: {MODEL_TARGET_SAMPLE_RATE} Hz.")
    print(f"Processing {WINDOW_DURATION_S}s audio chunks, striding by {STRIDE_DURATION_S}s.")
    print(f"Mic window: {mic_window_duration_samples} samples. Mic stride: {mic_stride_duration_samples} samples.")
    print("Press Ctrl+C to stop.")

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
                    current_window_audio_mic_sr = audio_buffer[:mic_window_duration_samples].copy()
                    audio_buffer = audio_buffer[mic_stride_duration_samples:]

                    model_input = preprocess_audio_chunk(
                        current_window_audio_mic_sr,
                        current_sample_rate=mic_actual_samplerate,
                        target_sample_rate=MODEL_TARGET_SAMPLE_RATE,
                        device=device_g
                    )

                    with torch.no_grad():
                        output = model_loaded(model_input) # output shape: (N_CLASS,)

                    probabilities = torch.exp(output) # Convert log-probabilities to probabilities
                    top_prob, top_idx = torch.max(probabilities, dim=0)
                    predicted_label = get_data.idx_to_label(top_idx.item())

                    if predicted_label in target_labels and top_prob.item() >= args.threshold:
                        print(
                            f"Detected '{predicted_label}' (one of {target_labels}) with probability {top_prob.item():.3f} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    elif args.verbose:
                        print(f"  Top: '{predicted_label}' ({top_prob.item():.2f})", end='\r')

                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time wake word detection.")
    parser.add_argument(
        "--model_path", type=str, default=MODEL_PATH, help="Path to the .torch model file."
    )
    parser.add_argument(
        "--wakewords", # Changed from --wakeword
        type=str,
        nargs='+',      # Accepts one or more arguments
        default=DEFAULT_TARGET_WAKE_WORDS, # Use the new default list
        help="One or more wake words to detect (space-separated if multiple)."
    )
    parser.add_argument(
        "--threshold", type=float, default=PROBABILITY_THRESHOLD, help="Detection probability threshold."
    )
    parser.add_argument(
        "--device", type=str, default=util.get_device(), help="Device to use ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--mic_id", type=int, default=None, help="Microphone ID (integer). List with 'python -m sounddevice'."
    )
    parser.add_argument(
        "--mic_samplerate", type=int, default=MODEL_TARGET_SAMPLE_RATE,
        help=f"Sample rate of the microphone. Audio will be resampled to {MODEL_TARGET_SAMPLE_RATE} Hz if different. "
             f"For custom FIR filtering of 44.1kHz audio (cutoff 8kHz before 16kHz resampling), set this to 44100."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print all top predictions, not just the target."
    )

    cli_args = parser.parse_args()

    try:
        sd.query_devices()
    except Exception as e:
        print("Sounddevice issue. Do you have it installed ('pip install sounddevice') and a microphone connected?")
        print(f"Error: {e}")
        sys.exit(1)

    main(cli_args)