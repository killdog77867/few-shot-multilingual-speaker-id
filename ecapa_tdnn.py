# ecapa_tdnn.py
from speechbrain.pretrained import SpeakerRecognition # Or speechbrain.inference.speaker
import torch
import numpy as np
import os
import traceback

class ECAPA_TDNN:
    # --- ENSURE DEFAULTS MATCH SPEAKER RECOGNITION MODEL ---
    def __init__(self, model_source="speechbrain/spkrec-ecapa-voxceleb", device="cpu", savedir="pretrained_models"):
        """
        Initializes the ECAPA_TDNN model optimized for Speaker Recognition.
        """
        print(f"Initializing ECAPA_TDNN with Speaker Recognition model: {model_source}")
        # Correct embedding size for spkrec-ecapa-voxceleb is 192
        self.embedding_size = 192 # <<<< CORRECTED SIZE
        print(f"Expecting embedding size: {self.embedding_size}")

        os.makedirs(savedir, exist_ok=True)
        try:
            # Ensure the correct model source is used for loading HParams/checkpoints
            self.model = SpeakerRecognition.from_hparams(
                source=model_source, # Use the source passed to init
                savedir=savedir
            )
            self.device = torch.device(device)
            self.model.to(self.device)
            print(f"ECAPA-TDNN model '{model_source}' loaded successfully on {self.device}")

        except Exception as e:
            print(f"ERROR: Failed to load model '{model_source}'.")
            print(f"       Make sure the identifier is correct and dependencies are met.")
            print(f"       Error details: {e}")
            print(traceback.format_exc())
            raise

    def extract_embedding(self, audio, sr=16000):
        """
        Extracts speaker embedding (expected size: 192) from raw audio waveform.
        """
        # --- (Keep the robust audio processing and tensor conversion logic) ---
        if not isinstance(audio, np.ndarray):
             try: audio = np.array(audio, dtype=np.float32)
             except Exception as e: raise TypeError(f"Audio to NumPy failed: {e}")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if audio.size == 0: raise ValueError("Input audio is empty.")
        min_len = int(sr * 0.1);
        if len(audio) < min_len: audio = np.pad(audio, (0, min_len - len(audio)), 'constant')

        try:
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
            if audio_tensor.ndim != 2: raise ValueError(f"Bad tensor shape: {audio_tensor.ndim}")

            # --- Perform embedding extraction ---
            embedding = self.model.encode_batch(audio_tensor)
            print(f"Raw embedding shape from model: {embedding.shape}") # Should be [1, 1, 192] or [1, 192]

            # --- Handle potential extra dimensions if model outputs [1, 1, 192] ---
            if embedding.ndim > 2:
                 print(f"Reducing embedding dimensions from {embedding.ndim} to 2.")
                 # Squeeze redundant middle dimension if present
                 if embedding.shape[0] == 1 and embedding.shape[1] == 1:
                     embedding = embedding.squeeze(1) # Results in [1, 192]
                 else:
                      # Fallback if shape is unexpected, might need adjustment
                      embedding = embedding.mean(dim=list(range(embedding.ndim - 1)), keepdim=True)

            # --- Final shape check and return ---
            # Check against self.embedding_size (which is now 192)
            if embedding.shape != (1, self.embedding_size):
                 print(f"ERROR: Processed embedding shape {embedding.shape}, expected (1, {self.embedding_size}).")
                 return None # Indicate failure

            final_embedding = embedding.squeeze(0).cpu().detach().numpy()
            print(f"Final embedding shape: {final_embedding.shape}") # Should be (192,)
            if final_embedding.shape != (self.embedding_size,):
                 print(f"ERROR: Final numpy embedding shape {final_embedding.shape}, expected ({self.embedding_size},)")
                 return None

            return final_embedding # Shape: (192,)

        except Exception as e:
            print(f"Error during embedding extraction: {e}")
            print(traceback.format_exc())
            return None

    # --- (Keep save_embedding and load_embedding - they use self.embedding_size = 192) ---
    def save_embedding(self, embedding, file_path):
        if embedding is None: print(f"Error: Cannot save None embedding to {file_path}."); return
        if embedding.shape != (self.embedding_size,):
             print(f"Error: Cannot save embedding shape {embedding.shape} to {file_path}. Expected ({self.embedding_size},)."); return
        try: np.save(file_path, embedding); print(f"Embedding saved to {file_path}")
        except Exception as e: print(f"Error saving embedding to {file_path}: {e}")

    def load_embedding(self, file_path):
        if os.path.exists(file_path):
            try:
                embedding = np.load(file_path)
                if embedding.shape != (self.embedding_size,):
                   print(f"Warning: Loaded {file_path} shape {embedding.shape} != expected ({self.embedding_size},)."); return None
                return embedding
            except Exception as e: print(f"Error loading {file_path}: {e}"); return None
        else: return None