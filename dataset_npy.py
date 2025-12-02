# dataset_npy.py
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# --- Constants (no changes here) ---
# These are still useful for potential debugging or feature engineering
KEYPOINT_NAME_TO_ID = {
    'Nose': 0, 'Left Eye': 1, 'Right Eye': 2, 'Left Ear': 3, 'Right Ear': 4,
    'Left Shoulder': 5, 'Right Shoulder': 6, 'Left Elbow': 7, 'Right Elbow': 8,
    'Left Wrist': 9, 'Right Wrist': 10, 'Left Hip': 11, 'Right Hip': 12,
    'Left Knee': 13, 'Right Knee': 14, 'Left Ankle': 15, 'Right Ankle': 16
}
ID_TO_KEYPOINT_NAME = {v: k for k, v in KEYPOINT_NAME_TO_ID.items()}

class FallDetectionDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-processed keypoints from .npy files.
    This version is significantly faster as it avoids on-the-fly CSV parsing and video reading.
    """
    def __init__(self, 
                 dataset_dir: str, 
                 sequence_length: int = 45, 
                 subsample_step: int = 2, # More direct than using FPS
                 min_avg_confidence: float = 0.2):
        
        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.subsample_step = subsample_step
        self.min_avg_confidence = min_avg_confidence
        # Calculate the total number of frames we need to load before subsampling
        self.frames_to_load = self.sequence_length * self.subsample_step

        self.samples = []
        print(f"--- Scanning for pre-processed .npy keypoints in: {dataset_dir} ---")
        self._discover_samples()
        if not self.samples:
            raise RuntimeError(f"No valid .npy files found. Check the dataset_dir path: {dataset_dir}")

    def _discover_samples(self):
        """
        --- MODIFIED ---
        Scans for .npy files, validates them by checking their average confidence,
        and only adds high-quality samples to the list.
        """
        skipped_low_confidence = 0
        
        for label_name, label_id in [("Fall", 1), ("No_Fall", 0)]:
            # The new folder structure is simpler
            keypoints_folder = os.path.join(self.dataset_dir, label_name)
            
            if not os.path.exists(keypoints_folder):
                print(f"Warning: Directory not found, skipping: {keypoints_folder}")
                continue
            
            for npy_filename in os.listdir(keypoints_folder):
                if not npy_filename.lower().endswith(".npy"):
                    continue
                
                npy_path = os.path.join(keypoints_folder, npy_filename)

                # --- VALIDATION STEP: Check average confidence ---
                try:
                    # Load the data to check it
                    data = np.load(npy_path)
                    # Confidence is the 3rd value (index 2) in the last dimension
                    avg_conf = data[:, :, 2].mean()
                    
                    if avg_conf < self.min_avg_confidence:
                        skipped_low_confidence += 1
                        continue # Skip this file
                    
                    # If the file is valid and high-quality, add it.
                    # We no longer need the video path.
                    self.samples.append({
                        "npy_path": npy_path,
                        "label": label_id
                    })
                except Exception as e:
                    print(f"Warning: Could not load or process {npy_path}. Error: {e}")
                    continue

        print(f"Found {len(self.samples)} valid, high-confidence samples.")
        if skipped_low_confidence > 0:
            print(f"  - Skipped {skipped_low_confidence} files due to low average confidence (< {self.min_avg_confidence}).")

    def _load_keypoints_from_npy(self, npy_path):
        """
        --- NEW & SIMPLIFIED ---
        Loads keypoints directly from a .npy file. The data is already normalized.
        """
        try:
            # The .npy file is already in the format we need: (frames, 17, 3)
            return np.load(npy_path)
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        --- MODIFIED ---
        Loads a single data sample. This is now much faster.
        """
        sample_info = self.samples[idx]
        label = sample_info["label"]
        
        # --- REMOVED ---
        # No longer need to open the video with cv2 to get dimensions.
        # This is a major performance improvement.
        
        # --- MODIFIED ---
        # Load the pre-processed keypoints directly.
        full_keypoints_seq = self._load_keypoints_from_npy(sample_info["npy_path"])
        
        # Handle cases where loading might fail
        if full_keypoints_seq is None or full_keypoints_seq.shape[0] == 0:
            # Return a zero tensor if data is invalid
            return torch.zeros((self.sequence_length, 17, 3)), torch.tensor(0, dtype=torch.float32)
        
        # --- The rest of the logic for sampling/padding remains the same ---
        # It's robust and works on any NumPy sequence.
        
        total_frames = len(full_keypoints_seq)
        
        # 1. Get a chunk of frames of length `self.frames_to_load`
        if total_frames <= self.frames_to_load:
            # If the video is shorter, pad it
            keypoints_chunk = full_keypoints_seq
            padding_needed = self.frames_to_load - total_frames
            if padding_needed > 0:
                # Repeat the last frame for padding
                padding = np.repeat(keypoints_chunk[-1:, :, :], padding_needed, axis=0)
                keypoints_chunk = np.vstack([keypoints_chunk, padding])
        else:
            # If the video is longer, take a random chunk
            start_frame = random.randint(0, total_frames - self.frames_to_load)
            keypoints_chunk = full_keypoints_seq[start_frame : start_frame + self.frames_to_load]
        
        # 2. Subsample the chunk to get the final sequence length
        keypoints_subsequence = keypoints_chunk[::self.subsample_step]
        
        # Ensure the final shape is correct due to potential rounding
        keypoints_subsequence = keypoints_subsequence[:self.sequence_length]

        # The original script had a 'mode' parameter. We'll assume 'internal_features'
        # which means returning the raw keypoint tensor.
        return torch.from_numpy(keypoints_subsequence), torch.tensor(label, dtype=torch.float32)