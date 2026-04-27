# Quality Guidelines

> Code quality standards for DeblurDiff research development.

## Overview

DeblurDiff is checkpoint-sensitive PyTorch research code. Small signature,
tensor-shape, config, or device changes can silently break training or make
published checkpoints unusable. Favor narrow, reproducible edits with explicit
shape/device checks over broad refactors.

## Required Patterns

- Preserve existing config-driven construction via `target` and `params`.
  Prefer adding config fields over hard-coded paths or hyperparameters.
- Keep tensor ranges explicit:
  - GT images in training are expected in `[-1, 1]`.
  - LQ/blurry conditioning images are expected in `[0, 1]`.
  - VAE latents are scaled by `latent_scale_factor`.
- Keep model interface contracts stable unless updating all call sites:
  - `ControlLDM.forward(x_noisy, t, cond) -> (eps, lr_kpn)`.
  - `cond` must contain `c_txt` and `c_img`.
  - `Diffusion.p_losses(model, x_start, t, cond)` expects the model to return
    a noise prediction and KPN latent.
- When adding new training or inference options, wire them through argparse,
  config files, and call sites together.
- For large-image support, use the existing tiled helpers in `utils.common`,
  `model.cldm`, `utils.pipeline`, and `utils.sampler` instead of introducing a
  separate tiling convention.

## Forbidden Patterns

- Do not move learned modules or rename state-dict keys casually. Existing
  checkpoints depend on module paths such as `controlnet`, `kpn`, `unet`,
  `vae`, and `clip`.
- Do not assume CPU/MPS support for LKPN/EAC. `model/lkpn.py` depends on CuPy
  CUDA dynamic convolution.
- Do not add new absolute author-machine paths to committed config or shell
  scripts. The current absolute paths are legacy artifacts to replace when
  making runnable experiments.
- Do not let source comments or README claims override runtime behavior. Verify
  active code paths in `train.py`, `utils/inference.py`, `utils/pipeline.py`,
  and `utils/sampler.py`.
- Do not treat `dataset/degradation.py` parameters as active training behavior
  unless `CodeformerDataset.__getitem__()` actually applies them.

## Verification Checklist

For model or sampler changes:

- Instantiate `ControlLDM` from `configs/inference/cldm.yaml`.
- Check LKPN output shape is compatible with `IDynamicConv`.
- Check ControlNet receives an 8-channel hint.
- Run a minimal forward pass on CUDA if the EAC path changed.

For training changes:

- Confirm SD checkpoint loading still freezes VAE, CLIP, and base UNet.
- Confirm the optimizer only includes intended trainable modules.
- Confirm `resume` and `resume_kpn` handling is valid for the config in use.
- Confirm dataset output shapes and value ranges before `vae_encode()`.

For inference changes:

- Run `inference.py --help` to validate CLI import and argument parsing.
- Run a single-image smoke test when CUDA and checkpoint files are available.
- Exercise `--tiled` separately from guidance; current code asserts that tiled
  sampling does not support guidance.
- Check output image dimensions match the original unpadded input dimensions.

## Known Gaps To Preserve Or Fix Deliberately

- `configs/train/train.yaml` lacks explicit `resume` and `resume_kpn` keys even
  though `train.py` reads them. Fixing this should be done as a targeted config
  contract change.
- `Diffusion.p_losses()` implements latent KPN loss but not the paper's
  pixel-space KPN loss. Add pixel loss only with a clear experiment note because
  it changes training behavior.
- Training and test shell scripts contain machine-specific paths and GPU IDs.
  Treat them as examples until made portable.
- README is minimal and does not describe checkpoint format, expected data tree,
  or CUDA/CuPy constraints.

## Review Checklist

- Does the change keep paper concepts aligned with the code map in
  `architecture-map.md`?
- Are checkpoint keys, tensor shapes, and config fields still compatible?
- Are dataset and inference assumptions documented if changed?
- Did you avoid unrelated cleanup in model files copied from Stable Diffusion,
  DiffBIR, or DemystifyLocalViT?
- Did you run the narrowest useful smoke check available in the current
  environment?
