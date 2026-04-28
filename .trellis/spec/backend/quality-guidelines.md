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
- Treat the local macOS workspace as an editing workspace, not the development
  runtime. GPU/runtime checks should run on the configured Ubuntu server.
- Keep tensor ranges explicit:
  - GT images in training are expected in `[-1, 1]`.
  - LQ/blurry conditioning images are expected in `[0, 1]`.
  - VAE latents are scaled by `latent_scale_factor`.
- Keep paired dataset transforms spatially identical for GT/LQ images. Random
  crop must sample one crop box and apply it to both images.
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

## Dataset Config Contract

### Scope / Trigger

Use this contract when editing `dataset/codeformer.py`,
`configs/train/*.yaml`, or `train.py` dataset consumption.

### Signature

```python
CodeformerDataset(
    file_list: str,
    file_backend_cfg: Mapping[str, Any],
    out_size: int,
    crop_type: Literal["none", "center", "random"],
    path_replace_from: str = "HR",
    path_replace_to: str = "Blur",
) -> Dataset[(gt, lq, prompt)]
```

### Contracts

- `file_list` may be either a newline-delimited text file of sharp-image paths
  or a sharp-image directory. Directory mode recursively enumerates supported
  image files.
- The paired blurred path is derived by replacing `path_replace_from` with
  `path_replace_to` in each GT path. The default remains `HR -> Blur`.
- `crop_type: none` returns full-resolution pairs.
- `crop_type: center` or `random` applies one crop box to both GT and LQ. If an
  image is smaller than `out_size`, both images are resized by the same scale
  before cropping.
- `__getitem__()` returns `gt` as RGB `float32` in `[-1, 1]`, `lq` as RGB
  `float32` in `[0, 1]`, and `prompt` as a string.

### Validation & Error Matrix

| Case | Expected behavior |
|------|-------------------|
| `crop_type` not in `none/center/random` | Dataset construction fails. |
| `crop_type != none` and `out_size <= 0` | Dataset construction fails. |
| GT and LQ spatial sizes differ | `__getitem__()` raises `ValueError`. |
| Legacy synthetic degradation params in YAML | Config instantiation fails; remove them unless the dataset implementation actually uses them. |

## Training Config Contract

### Scope / Trigger

Use this contract when editing `configs/train/*.yaml` or adding fields consumed
by `train.py`.

### Active Fields

| Field | Consumer | Contract |
|-------|----------|----------|
| `train.sd_path` | `ControlLDM.load_pretrained_sd()` | Stable Diffusion 2.1 checkpoint. Required only when `resume_full` is `null`. |
| `train.resume_full` | `ControlLDM.load_state_dict()` | Optional full `ControlLDM.state_dict()` for fine-tuning released inference checkpoints. Takes precedence over `sd_path` and submodule resume fields. |
| `train.resume` | `load_controlnet_from_ckpt()` | Optional pure ControlNet checkpoint. Use `null` when initializing from SD. |
| `train.resume_kpn` | `cldm.kpn.load_state_dict()` | Optional pure LKPN checkpoint. Use `null` when training LKPN from scratch. |
| `train.exp_dir` | output setup | Directory for TensorBoard logs and `checkpoints/*.pt`. |
| `train.learning_rate` | AdamW optimizer | Single LR applied to ControlNet and LKPN parameter groups. |
| `train.batch_size` | DataLoader / Accelerator | Global batch size passed to `DataLoader`; `Accelerator(split_batches=True)` splits it across distributed processes. |
| `train.num_workers` | DataLoader | Number of worker processes for training data loading. |
| `train.train_steps` | training loop | Total optimizer steps before stopping. |
| `train.log_every` | TensorBoard scalar logging | Step interval for averaged loss logging. |
| `train.ckpt_every` | checkpoint writer | Step interval for saving full `ControlLDM.state_dict()`. |

### Validation

Do not add YAML fields that are not read by `train.py` unless the code path is
implemented in the same change. Removed examples include scheduler-only fields
without a scheduler and image-log fields without an image logging branch.

Checkpoint files may be direct state dicts or dicts with a `state_dict` entry.
DDP-style `module.` prefixes are stripped before loading.
When `resume_full` is used, `train.py` must still freeze `unet`, `vae`, and
`clip` after loading, because it skips `ControlLDM.load_pretrained_sd()`.

## Scenario: SwanLab Monitoring And Sparse Validation

### 1. Scope / Trigger

- Trigger: editing SwanLab logging, validation data loading, metric keys, or
  validation scheduling in `train.py` / `configs/train/*.yaml`.

### 2. Signatures

```python
build_validation_steps(first_step: int, max_steps: int, num_runs: int) -> list[int]
build_interval_steps(first_step: int, max_steps: int, every_n_steps: int) -> list[int]
PairedDirDataset(sharp_dir: str, blur_dir: str, max_images: int | None = None)
Diffusion.p_losses(model, x_start, t, cond, return_dict: bool = False)
```

### 3. Contracts

- SwanLab logging is main-process only. Non-main distributed processes must not
  initialize SwanLab or upload images.
- Regular validation is sharded across distributed ranks and then reduced via
  scalar PSNR/SSIM sums and image counts.
- Visual validation uses its own `validation.visual.first_step` +
  `validation.visual.every_n_steps` interval schedule. Keep it offset from
  regular validation steps so long full validation and visual uploads do not
  run in the same training step; the training loop drops visual steps that
  collide with regular validation. The visual split should use the curated
  `imgs/target` + `imgs/input` 4-image set and only the main process uploads
  SwanLab images.
- `build_validation_steps(1000, 10000, 5)` must produce
  `[1000, 3250, 5500, 7750, 10000]`.
- FLOPs are approximate 256x256, batch-1, single `ControlLDM.forward()` FLOPs;
  do not multiply by diffusion sampling steps.
- `PairedDirDataset` matches by same relative path first, then by unique
  filename fallback. GT/LQ image sizes must match.
- Metric tensors are RGB `[0, 1]`. Training GT stays `[-1, 1]`; validation code
  converts it back to `[0, 1]` before PSNR/SSIM.
- The current repository implements validation metrics only through
  TorchMetrics PSNR/SSIM in `train.py`; do not log additional metric keys unless
  their implementation and dependencies are added in the same change.
- `visual/pictures` uploads restored images only, not blur/restored/sharp
  comparison grids. Each SwanLab image caption must use the per-image metric
  format `psnr:<value>;ssim:<value>`.
- SwanLab keys are stable:
  `param/params`, `param/trainable_params`, `param/FLOPs`,
  `losses/l_total`, `losses/l_denoise`, `losses/l_kpn_latent`,
  `learning_rate/lr_controlnet`, `learning_rate/lr_kpn`, `metrics/psnr`,
  `metrics/ssim`, `visual/vis_psnr`, `visual/vis_ssim`, `visual/pictures`.

### 4. Validation & Error Matrix

| Case | Expected behavior |
|------|-------------------|
| `swanlab.enabled: false` | Training imports and runs without importing SwanLab. |
| `validation.enabled: false` | No validation datasets are constructed. |
| Missing validation pair | `PairedDirDataset` raises `FileNotFoundError` before training starts. |
| Duplicate blur filenames for fallback matching | `PairedDirDataset` raises `ValueError`. |
| FLOPs profiler unsupported | Training continues and skips `param/FLOPs` after printing the exception. |

### 5. Good/Base/Bad Cases

- Good: 2-GPU training logs SwanLab once, shards regular validation across both
  ranks, and only uploads visual images from the main process.
- Base: `validation.num_runs: 4` and `train.train_steps: 50000` runs regular
  validation at `[1000, 17333, 33667, 50000]`. For a 32-point visual monitor
  cadence across 50K optimizer steps, use
  `validation.visual.every_n_steps: 1562`.
- Bad: running full regular validation only on rank 0 while other ranks wait at
  a collective can hit the NCCL watchdog timeout on long validation runs.

### 6. Tests Required

- On the remote GPU server, run `py_compile` for `train.py`,
  `model/gaussian_diffusion.py`, `dataset/paired_dir.py`, and
  `utils/pipeline.py`.
- Dataset smoke test with two paired images validates length, shape, dtype, and
  value ranges.
- Validation schedule smoke test checks the 10K-step example and the active
  50K-step regular/visual split schedules.

### 7. Wrong vs Correct

#### Wrong

```python
if accelerator.is_local_main_process:
    run_validation(...)
# other ranks continue immediately into the next backward()
```

#### Correct

```python
accelerator.wait_for_everyone()
run_validation(..., run_val=run_val, run_visual=run_visual)
accelerator.wait_for_everyone()
```

## Forbidden Patterns

- Do not move learned modules or rename state-dict keys casually. Existing
  checkpoints depend on module paths such as `controlnet`, `kpn`, `unet`,
  `vae`, and `clip`.
- Do not assume CPU/MPS support for LKPN/EAC. `model/lkpn.py` depends on CuPy
  CUDA dynamic convolution.
- `IDynamicConv` launches the CuPy dynamic-convolution kernel per sample and
  casts inputs/weights to contiguous `float32` before launch. This avoids
  asynchronous illegal memory accesses seen with batched CuPy launches and
  autocast dtypes.
- Do not add new absolute author-machine paths to committed config or shell
  scripts. The current absolute paths are legacy artifacts to replace when
  making runnable experiments.
- Do not let source comments or README claims override runtime behavior. Verify
  active code paths in `train.py`, `utils/inference.py`, `utils/pipeline.py`,
  and `utils/sampler.py`.
- Do not reintroduce synthetic degradation parameters into
  `CodeformerDataset` unless `__getitem__()` actually applies them to the
  paired data path.

## Verification Checklist

Remote runtime:

- SSH target: `gaoyin@172.28.11.129`
- Project path: `/data/users/gaoyin/2024_CKB/DeblurDiff`
- Use this server for dependency-sensitive checks, CUDA/CuPy checks, SwanLab
  smoke tests, and training/validation dry runs. The local workspace may lack
  the Python and GPU dependencies required by this project.

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
- Confirm `crop_type: random` keeps paired GT/LQ crops aligned.

For inference changes:

- Run `inference.py --help` to validate CLI import and argument parsing.
- Run a single-image smoke test when CUDA and checkpoint files are available.
- Exercise `--tiled` separately from guidance; current code asserts that tiled
  sampling does not support guidance.
- Check output image dimensions match the original unpadded input dimensions.

For environment changes:

- Keep PyTorch, torchvision, xformers, and CuPy on one CUDA line.
- The current environment target is Python 3.10 with PyTorch 2.1.2,
  torchvision 0.16.2, `pytorch-cuda=12.1`, `cupy-cuda12x==12.3.0`, and
  `xformers==0.0.23.post1`.
- SwanLab monitoring uses `swanlab==0.7.16`.
- Do not reintroduce a mixed CUDA environment such as Conda `cudatoolkit=11.8`
  plus pip `nvidia-*-cu12` packages.
- After creating the environment, verify both `torch.version.cuda` and
  `cupy.show_config()` before running LKPN/EAC paths.

## Known Gaps To Preserve Or Fix Deliberately

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
