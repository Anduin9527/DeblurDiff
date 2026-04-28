# Directory Structure

> Repository layout and ownership boundaries for DeblurDiff.

## Overview

This repository implements the paper "DeblurDiff: Real-World Image Deblurring
with Generative Diffusion Models" as a compact PyTorch research codebase. The
top-level scripts are thin entry points; most behavior lives in `model/`,
`utils/`, `dataset/`, and `configs/`.

## Directory Layout

```text
.
|-- README.md                    # Minimal usage notes, checkpoint link, acknowledgements
|-- environment.yml              # Python 3.8 / CUDA / PyTorch / CuPy environment
|-- train.py                     # Training entry point for ControlNet + LKPN
|-- inference.py                 # CLI entry point for restoration inference
|-- train.sh                     # accelerate launch wrapper
|-- test.sh                      # single-GPU inference wrapper
|-- file_img.sh                  # training file-list generator for HR image paths
|-- run_file.sh                  # cluster/srun wrapper around train.sh
|-- imgs/                        # curated 4-image paired visual-validation set
|   |-- input/                   # blurred visual-validation inputs
|   `-- target/                  # sharp visual-validation references
|-- configs/
|   |-- train/train.yaml         # ControlLDM, diffusion, dataset, and training config
|   `-- inference/*.yaml         # inference-time ControlLDM and diffusion configs
|-- model/
|   |-- cldm.py                  # ControlLDM assembly: SD UNet/VAE/CLIP + LKPN + ControlNet
|   |-- lkpn.py                  # LKPN and CuPy-backed EAC dynamic convolution
|   |-- controlnet.py            # ControlNet branch and controlled SD UNet wrapper
|   |-- gaussian_diffusion.py    # DDPM beta schedule, q_sample, training loss
|   |-- unet.py                  # SD-style UNet implementation
|   |-- attention.py             # spatial transformer and attention blocks
|   |-- vae.py                   # Stable Diffusion VAE
|   |-- clip.py                  # frozen OpenCLIP text embedder
|   `-- open_clip/               # vendored OpenCLIP implementation and tokenizer assets
|-- utils/
|   |-- inference.py             # model loading and batch/image IO loop
|   |-- pipeline.py              # preprocessing, conditioning, sampling, decode, color correction
|   |-- sampler.py               # spaced reverse diffusion sampler
|   |-- cond_fn.py               # optional restoration guidance losses
|   `-- common.py                # config instantiation, tiling, wavelet helpers
`-- dataset/
    |-- codeformer.py            # paired HR/Blur dataset and paired crop used by train.py
    |-- paired_dir.py            # sharp_dir/blur_dir paired validation dataset
    |-- degradation.py           # synthetic degradation utilities, not used by CodeformerDataset
    |-- file_backend.py          # hard-disk and Petrel-style storage backends
    `-- utils.py                 # file-list and crop helpers
```

## Module Boundaries

- `model/` owns neural network definitions and checkpoint-compatible module
  names. Preserve state-dict key compatibility unless explicitly changing
  checkpoint format.
- `utils/` owns execution orchestration: inference loops, sampling, tiling,
  guidance, and generic helper functions.
- `dataset/` owns data loading and training pair construction. Keep dataset
  output contract aligned with `train.py`: `(gt, lq, prompt)`.
- `configs/` owns runtime wiring. Avoid hard-coding new paths or hyperparameters
  in Python when an existing config file can carry them.
- Shell scripts are convenience wrappers and currently contain environment- or
  machine-specific paths. Keep them small and document assumptions.

## Naming Conventions

- Keep model classes in `PascalCase` (`ControlLDM`, `LKPN`, `Diffusion`).
- Keep functions, module files, and config keys in `snake_case`.
- Preserve paper terminology where it already appears in code: `LKPN`, `EAC`,
  `ControlLDM`, `c_img`, `c_txt`, `x_noisy`, `z_0`, `lq`, `gt`.
- New config files should mirror the existing `target` + `params` instantiation
  pattern used by `utils.common.instantiate_from_config`.

## Placement Rules

- Add new model variants under `model/`, and expose only stable public classes
  from `model/__init__.py`.
- Add new sampling or inference orchestration under `utils/`, not inside model
  classes, unless it changes the learned module itself.
- Add new data sources under `dataset/`; do not make `train.py` parse dataset
  formats directly.
- Add reproducible experiment configs under `configs/` instead of editing
  machine-specific paths directly into scripts.

## Examples To Follow

- `model/cldm.py` is the main assembly example for composing frozen SD modules
  with trainable restoration modules.
- `configs/train/train.yaml` shows the expected OmegaConf wiring style.
- `utils/pipeline.py` and `utils/sampler.py` show the inference-time separation
  between conditioning/preprocessing and reverse diffusion.
