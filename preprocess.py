import numpy as np
import librosa
import soundfile as sf
import io
import os
from pydub import AudioSegment # Import pydub
from pydub.exceptions import CouldntDecodeError # Import specific pydub error

# --- Check for FFmpeg (Optional but Recommended) ---
# Simple check, might need refinement depending on system setup
ffmpeg_found = False
try:
    # Check if ffmpeg command runs without error (exit code 0)
    import subprocess
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    ffmpeg_found = True
    print("INFO: FFmpeg found. Audio conversion enabled.")
except (FileNotFoundError, subprocess.CalledProcessError):
    print("WARNING: FFmpeg not found or not executable.")
    print("         pydub audio conversion might fail for non-WAV formats.")
    print("         Please install FFmpeg and ensure it's in your system's PATH.")
    # Depending on your needs, you could make this a fatal error:
    # raise RuntimeError("FFmpeg is required but not found. Please install it.")

# Parameters
SAMPLE_RATE = 16000  # ECAPA-TDNN expects 16kHz audio

def load_audio(file_path):
    """Loads audio from a file path. Less robust for web uploads."""
    # ... (keep existing code, but maybe add a warning) ...
    print(f"WARNING: Using load_audio for path {file_path}. Prefer _from_bytes for uploads.")
    try:
        audio, sr = sf.read(file_path, dtype='float32')
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(y=np.ascontiguousarray(audio), orig_sr=sr, target_sr=SAMPLE_RATE)
        return audio, SAMPLE_RATE
    except Exception as e:
        print(f"Error loading audio from path {file_path}: {e}")
        return None, None

def load_audio_from_bytes(audio_bytes):
    """
    Load audio from bytes, CONVERT TO WAV using pydub, then read and resample.
    """
    input_buffer = io.BytesIO(audio_bytes)
    output_buffer = io.BytesIO() # For storing the WAV data

    try:
        # 1. Load audio using pydub (auto-detects format)
        print("Attempting to load audio with pydub...")
        audio_segment = AudioSegment.from_file(input_buffer) # pydub detects format
        print(f"pydub loaded audio: Frame Rate={audio_segment.frame_rate}, Channels={audio_segment.channels}, Sample Width={audio_segment.sample_width}")

        # 2. Export as WAV into the output buffer
        print("Exporting audio segment as WAV...")
        # Ensure export parameters match expectations if needed (e.g., forcing mono)
        # audio_segment = audio_segment.set_channels(1) # Force mono *before* export if needed
        audio_segment.export(output_buffer, format="wav")
        output_buffer.seek(0) # Rewind buffer to the beginning for reading
        print("Export to WAV format successful.")

        # 3. Read the WAV data from the output buffer using soundfile
        print("Reading WAV data from buffer using soundfile...")
        audio, sr = sf.read(output_buffer, dtype='float32')
        print(f"Soundfile read WAV: Original SR={sr}, Samples={len(audio)}, Dtype={audio.dtype}, Dims={audio.ndim}")

        # 4. Ensure mono
        if audio.ndim > 1:
            print("Converting multi-channel audio to mono.")
            audio = audio.mean(axis=1)

        # 5. Resample if necessary
        if sr != SAMPLE_RATE:
            print(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz")
            audio_contiguous = np.ascontiguousarray(audio)
            audio = librosa.resample(y=audio_contiguous, orig_sr=sr, target_sr=SAMPLE_RATE)
            print(f"Resampling complete. New sample count: {len(audio)}")
        else:
             print(f"Audio already at target sample rate: {SAMPLE_RATE}Hz")

        print(f"Processed audio: SR={SAMPLE_RATE}, Samples={len(audio)}, Dtype={audio.dtype}")
        return audio, SAMPLE_RATE

    except CouldntDecodeError as pydub_err:
        print(f"Pydub Error: Could not decode audio stream. {pydub_err}")
        print("             This might happen if FFmpeg is missing or the format is truly unsupported.")
        return None, None
    except sf.LibsndfileError as sf_err:
        print(f"Soundfile Error reading converted WAV data: {sf_err}")
        # This *shouldn't* happen now if pydub export worked, but good to keep.
        return None, None
    except Exception as e:
        print(f"Generic Error during audio processing from bytes: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None
    finally:
        # Close buffers (optional, handled by garbage collection, but explicit is okay)
        input_buffer.close()
        output_buffer.close()


def preprocess_audio_from_path(file_path):
    """
    Load and preprocess audio from a file path. USE WITH CAUTION.
    """
    print(f"Preprocessing audio from path: {file_path}")
    audio, sr = load_audio(file_path)
    return audio, sr

def preprocess_audio_from_bytes(audio_bytes):
    """
    Load, convert, and preprocess audio from bytes for ECAPA-TDNN.
    """
    print("Preprocessing audio from bytes (with WAV conversion)...")
    audio, sr = load_audio_from_bytes(audio_bytes)
    # Add any other preprocessing steps here if needed AFTER loading/resampling
    return audio, sr

# Keep the __main__ block for potential standalone testing
if __name__ == "__main__":
    # ... (keep the existing __main__ block as is) ...
    # Create a dummy wav file for testing if needed
    dummy_file = "dummy_test.wav" # Make sure this exists or is created
    if os.path.exists(dummy_file):
        try:
            print(f"\n--- Testing preprocess_audio_from_bytes using {dummy_file} ---")
            with open(dummy_file, 'rb') as f:
                audio_bytes_content = f.read()
            print(f"Read {len(audio_bytes_content)} bytes from {dummy_file}")
            audio_b, sr_b = preprocess_audio_from_bytes(audio_bytes_content)
            if audio_b is not None:
                print("\nSUCCESS: Audio from bytes loaded and processed.")
                print("Shape:", audio_b.shape, "Sample rate:", sr_b)
            else:
                 print(f"\nFAILURE: Could not load/process audio from bytes of {dummy_file}")
            print("--- End Test ---")

        except Exception as e:
             print(f"\nError during __main__ test: {e}")
             import traceback
             print(traceback.format_exc())
    else:
        print(f"\nTest file '{dummy_file}' not found. Skipping __main__ test.")

    # --- Optional: Test with a known non-WAV file if you have one ---
    # test_webm_file = "test_audio.webm" # Replace with an actual webm file
    # if ffmpeg_found and os.path.exists(test_webm_file):
    #      try:
    #         print(f"\n--- Testing preprocess_audio_from_bytes using {test_webm_file} ---")
    #         with open(test_webm_file, 'rb') as f:
    #             webm_bytes_content = f.read()
    #         print(f"Read {len(webm_bytes_content)} bytes from {test_webm_file}")
    #         audio_w, sr_w = preprocess_audio_from_bytes(webm_bytes_content)
    #         if audio_w is not None:
    #             print("\nSUCCESS: WebM Audio from bytes loaded, converted and processed.")
    #             print("Shape:", audio_w.shape, "Sample rate:", sr_w)
    #         else:
    #              print(f"\nFAILURE: Could not load/process audio from bytes of {test_webm_file}")
    #         print("--- End Test ---")
    #      except Exception as e:
    #         print(f"\nError during WebM test: {e}")
    #         import traceback
    #         print(traceback.format_exc())
    # else:
    #     if not ffmpeg_found:
    #         print(f"\nSkipping WebM test because FFmpeg was not found.")
    #     elif not os.path.exists(test_webm_file):
    #         print(f"\nWebM test file '{test_webm_file}' not found. Skipping test.")