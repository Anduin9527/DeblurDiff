# Bootstrap: DeblurDiff Project Guidelines

## Purpose

Capture the real structure and development constraints of this DeblurDiff
research repository so future AI sessions use project-specific context instead
of generic Trellis fullstack templates.

## Scope

- Record the paper-to-code mapping for arXiv:2502.03810.
- Document the actual Python/PyTorch repository layout.
- Document quality and verification constraints for model, training, dataset,
  sampler, and inference changes.
- Remove irrelevant fullstack initialization specs for frontend, database, API
  error responses, and structured logging.

## Findings Recorded

- `model/cldm.py` is the system assembly point: frozen SD UNet/VAE/CLIP,
  trainable LKPN, and trainable ControlNet.
- `model/lkpn.py` implements LKPN and EAC through a CuPy-backed dynamic
  convolution. It predicts `4*5*5` kernel channels from concatenated `z_lq` and
  current `z_t`.
- `model/controlnet.py` implements the ControlNet branch and controlled UNet
  residual injection.
- `utils/sampler.py` is where the iterative paper loop appears at runtime:
  every reverse diffusion step calls `ControlLDM.forward()`, so LKPN receives
  the current latent and can refine kernels step by step.
- `model/gaussian_diffusion.py` implements denoising MSE plus latent KPN MSE;
  it does not currently implement the paper's pixel-space KPN loss.
- `dataset/codeformer.py` reads pre-generated `HR`/`Blur` pairs; the synthetic
  degradation parameters in config are not active in `__getitem__()`.
- `configs/train/train.yaml` contains author-machine paths and differs from the
  paper's reported batch size, learning rate, and iteration count.

## Updated Specs

| File | Purpose |
|------|---------|
| `.trellis/spec/backend/index.md` | Project-specific spec index for Python/ML work |
| `.trellis/spec/backend/architecture-map.md` | Paper-to-code mapping and runtime flow |
| `.trellis/spec/backend/directory-structure.md` | Actual repository layout and ownership boundaries |
| `.trellis/spec/backend/quality-guidelines.md` | Shape/device/config/checkpoint safety rules |

## Removed Initialization Specs

- `.trellis/spec/frontend/`
- `.trellis/spec/backend/database-guidelines.md`
- `.trellis/spec/backend/error-handling.md`
- `.trellis/spec/backend/logging-guidelines.md`

These files were Trellis bootstrap templates for systems this repository does
not currently contain.

## Acceptance Criteria

- [x] Project architecture conclusions are recorded in Trellis specs.
- [x] Actual directory structure is documented with real file examples.
- [x] Quality/spec guidance captures known paper-code differences and runnable
  caveats.
- [x] Irrelevant fullstack bootstrap specs are removed.
