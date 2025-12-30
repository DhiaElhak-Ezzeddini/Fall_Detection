# Fall Detection (Two-Stream Transformer)

This repository contains code and utilities for a fall detection pipeline that combines foreground masks and optical flow in a two-stream transformer model. The project includes scripts to preprocess raw videos, compute foreground masks and optical flow, prepare datasets saved as NumPy arrays, train and evaluate a two-stream transformer classifier, and tools to verify the processed data.

## Table of contents
- Project overview
- Repository structure
- Requirements
- Preprocessing (video -> processed data)
- Preparing data for training
- Model (Two-Stream Transformer)
- Training and evaluation
- Utilities
- Tips and troubleshooting
- Next steps

## Project overview

Goal: classify short video clips (sequences of frames) into `Fall` vs `No_Fall` using a two-stream approach:
- Stream 1: foreground masks (binary masks per frame)
- Stream 2: optical flow (per-frame 2-channel flow vectors)

The architecture extracts per-frame patch embeddings for each stream, aggregates them across time, and fuses stream-level class tokens in a simple transformer-based classifier.

Use cases:
- Offline training on a dataset of labeled videos
- Batch preprocessing of new videos into NumPy arrays for training or inference

## Repository structure (important files)

- `Video_Preprocessing.py` - Resize/clip videos and produce fixed-length video files in `Processed_Data/` (uses moviepy + ffmpeg).
- `background_substraction.py` / `BGSB_OpticalFlow.PY` / `optical_flow.py` - Small utilities / experiments for background subtraction and optical flow visualization and saving outputs.
- `dataset_npy.py` - PyTorch Dataset implementation that loads pre-processed keypoint `.npy` files (an alternative dataset pipeline used in experiments).
- `Two_Stream_Transformer.py` - Core training/validation scripts and model implementation. Contains:
  - `FGFLOWDataset` dataset class (loads foreground mask and optical flow `.npy` files)
  - `TwoStreamTransformer` model definition
  - training and validation loops
  - metric plotting utilities (`plot_metrics`)
- `verify_processed_data.py` - Script to verify data integrity of processed `.npy` files and labels CSV.
- `requirements.txt` - Project dependencies (see notes below about PyTorch installation and ffmpeg)
- Pretrained model weights: `best_two_stream_transformer*.pth` (checkpoint files saved during training)
- Training results directories: `training_results/`, `training_aug_results/`, `training_aug_2_results/`, etc. (plots and CSV summaries)

There are also several Jupyter notebooks (`*.ipynb`) in the repo for exploring the dataset and testing preprocessing.

## Requirements

Minimum Python: 3.8+ recommended.

Core Python packages (see `requirements.txt`):
- numpy, pandas, scikit-learn, matplotlib, tqdm, opencv-python, moviepy

PyTorch:
- Install `torch` and `torchvision` separately to match your platform and CUDA version. See https://pytorch.org/get-started/locally/ for the correct wheel/command. The `requirements.txt` contains guidance comments.

System dependencies:
- ffmpeg (for `moviepy` video writing/reading). On Ubuntu: `sudo apt install ffmpeg`.

GPU:
- For training, a CUDA-enabled GPU and a matching PyTorch build are recommended.

## Preprocessing

Typical preprocessing pipeline (high-level):

1. Raw videos are stored under `./Dataset/<Fall|No_Fall>/Raw_Video/`.
2. Run `Video_Preprocessing.py` to resize frames to 224x224 and produce fixed-length videos in `Processed_Data/<Fall|No_Fall>/`.
   - Configurable: target size and target frame count (defaults are in the script).
3. Compute foreground masks and optical flow per video and save NumPy arrays for each video.
   - Use `BGSB_OpticalFlow.PY` or `background_substraction.py` as reference scripts. They demonstrate:
     - background subtraction with MOG2
     - morphological cleaning of foreground masks
     - Farneback optical flow computation
   - A typical saved output per video:
     - `<video_stem>_fg.npy`  -> shape (T, H, W), binary or 0..255 masks
     - `<video_stem>_flow.npy` -> shape (T, H, W, 2), per-pixel flow vectors

4. Place processed `.npy` files under `Processed_For_DL/<Fall|No_Fall>/{fg,flow}/` or follow the layout expected by `verify_processed_data.py` and `Two_Stream_Transformer.py`.

Notes:
- Ensure `.npy` files have consistent lengths; scripts in this repo support padding or uniform sampling techniques.
- The `dataset_npy.py` file contains a different preprocessing path for keypoints saved as `.npy` (shape (frames, 17, 3)). Use whichever pipeline matches your data.

## Preparing data for training

Expected folder layout for the two-stream training dataset (used by `FGFLOWDataset`):

Processed_For_DL/
  ├─ Fall/
  │   ├─ fg/            # files named <stem>_fg.npy
  │   └─ flow/          # files named <stem>_flow.npy
  └─ No_Fall/
      ├─ fg/
      └─ flow/

Additionally, `verify_processed_data.py` expects a `Processed_For_DL/labels.csv` with columns `filename,label` linking stems to labels.

Dataset class behavior (Two_Stream_Transformer):
- `FGFLOWDataset` expects each `.npy` pair to have the same temporal length. It will convert masks to 0..1 floats, normalize/clamp optical flow to [-1,1] by a `flow_clip` factor, and optionally apply simple augmentations (horizontal flip, temporal jitter, additive flow noise).

## Model (Two-Stream Transformer)

High-level:
- Each frame is split into non-overlapping patches (via Conv2d with stride = patch_size).
- Per-frame patches are embedded, then tokens across time are concatenated to produce a long token sequence per stream.
- Each stream has a learnable [CLS] token and a transformer encoder stack; final CLS tokens are concatenated and fed to a small MLP classifier.

Key hyperparameters and defaults (in `Two_Stream_Transformer.py`):
- Image size: 224
- Patch size: 64
- Sequence length (frames): 63
- Patch embedding dimension (d_model): 64
- Transformer depth: configurable (default 1 in the script)

Output: single logit per input clip. Apply sigmoid to get probability of `Fall`.

## Training and evaluation

Two_Stream_Transformer.py includes training and validation loops and metric computation (precision, recall, F1, confusion matrix, ROC AUC). It saves plot artifacts in result folders such as `training_results/` or `training_aug_2_results/`.

Typical training steps (example flow):

1. Prepare processed `.npy` files and `labels.csv` as described above.
2. Create training and validation splits and generate a `labels` dictionary (mapping stems to 0/1). The repo contains notebooks and helper scripts used to build these splits during experiments.
3. Modify the script or write a small runner to instantiate `FGFLOWDataset` for train/val, create DataLoaders with `collate_fn`, instantiate `TwoStreamTransformer`, configure optimizer (Adam/SGD), loss (`BCEWithLogitsLoss`), and optionally a learning-rate scheduler.
4. Call `train_one_epoch` and `validate` across epochs, collecting metrics and saving checkpoints.

Saved artifacts in this repo include several `best_two_stream_transformer*.pth` files and result directories with plots and JSON/CSV training histories.

## Utilities

- `verify_processed_data.py` - Run this to detect missing/corrupted files, shape mismatches, and orphaned files not listed in `labels.csv`.
  - Usage: `python verify_processed_data.py`

- `background_substraction.py`, `BGSB_OpticalFlow.PY`, `optical_flow.py` - Reference scripts to compute foreground masks and visualize/save optical flow.

- Notebooks: Use the included Jupyter notebooks (e.g., `explore_dataset.ipynb`, `test_preprocessing.ipynb`) to inspect processed videos and debug preprocessing.

## Quick start examples

1. Install dependencies (example):

```bash
# system package
sudo apt install ffmpeg

# python deps (adjust torch install for your CUDA)
pip install -r requirements.txt
# then install torch following https://pytorch.org/get-started/locally/
```

2. Preprocess raw videos (example):

```bash
python Video_Preprocessing.py Fall
python Video_Preprocessing.py No_Fall
```

3. Compute foreground masks and optical flow for the processed videos (example script call):

```bash
python BGSB_OpticalFlow.PY Fall
python BGSB_OpticalFlow.PY No_Fall
```

4. Verify processed data and labels:

```bash
python verify_processed_data.py
```

5. Train the model (example skeleton — adapt to your config):

```bash
python -c "from Two_Stream_Transformer import TwoStreamTransformer, FGFLOWDataset; print('See README for full training recipe')"
```

Note: The repository contains training scripts and notebooks used in experiments. You may need to adapt paths and instantiate training loops according to your dataset split and environment.

## Tips and troubleshooting

- If `moviepy` fails to write videos, ensure `ffmpeg` is installed and available on the PATH.
- For PyTorch GPU training, install a matching `torch` wheel for your CUDA version. See PyTorch docs.
- If `verify_processed_data.py` reports low coverage or missing files, re-run the preprocessing and make sure the naming conventions (`<stem>_fg.npy`, `<stem>_flow.npy`) and `labels.csv` match.
- Watch for large memory usage when creating datasets for many videos; prefer lazy loading in the Dataset (the code already uses per-file NumPy loads in `FGFLOWDataset`).

## Next steps and suggestions

- Add a small CLI or config-driven training runner to make experiments reproducible (e.g., use `argparse` and a YAML config).
- Add unit tests for dataset loading and `verify_processed_data.py` logic.
- Consider storing processed arrays in an HDF5 store for more efficient random access to very large datasets.
- Add a small inference script that loads a checkpoint and runs a forward pass on a single `.npy` pair to produce a human-readable prediction.

## License

See the included `LICENSE` file in this repository for licensing details.

## Contact / Acknowledgements

Repository author: see repository metadata. If you need help running or extending this project, open an issue or contact the maintainer.
