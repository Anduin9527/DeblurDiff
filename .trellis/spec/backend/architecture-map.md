# Architecture Map

> Paper-to-code mapping for DeblurDiff.

## Paper Summary

The paper proposes DeblurDiff, a real-world image deblurring framework that
uses a pre-trained Stable Diffusion model as a generative prior. The key
addition is a Latent Kernel Prediction Network (LKPN) trained jointly with a
conditional diffusion model. LKPN predicts spatially variant latent-space
kernels at each diffusion step, Element-wise Adaptive Convolution (EAC) applies
those kernels to the blurry latent, and the refined latent becomes structural
guidance for ControlNet.

Relevant paper claims:

- LKPN is co-trained in latent space with conditional diffusion.
- LKPN predicts a per-latent-pixel spatially variant kernel.
- EAC applies the predicted kernel to the blurry latent while preserving input
  structure.
- `z_lq` and the EAC-refined `z^s_t` are concatenated as ControlNet condition.
- During sampling, each diffusion step feeds the current latent back into LKPN
  so kernels can be refined iteratively.

## Paper Concept To Code

| Paper concept | Code location | Notes |
|---------------|---------------|-------|
| Stable Diffusion backbone | `model/cldm.py`, `model/unet.py`, `model/vae.py`, `model/clip.py` | `ControlLDM.load_pretrained_sd()` loads SD UNet, VAE, and CLIP weights, then freezes them. |
| Conditional control branch | `model/controlnet.py` | `ControlNet` copies SD encoder-style blocks and emits zero-conv residual controls. |
| LKPN | `model/lkpn.py` | `LKPN` is a smaller UNet with `in_channels=8` and `out_channels=4*5*5`. |
| EAC | `model/lkpn.py` | `IDynamicConv` reshapes per-pixel kernels and dispatches a CuPy CUDA dynamic convolution. |
| Condition construction | `model/cldm.py` | `prepare_condition()` produces `c_txt` and `c_img`; `forward()` computes `lr_kpn` and concatenates `c_img` with `lr_kpn`. |
| DDPM training loss | `model/gaussian_diffusion.py` | `p_losses()` adds denoising MSE and latent KPN MSE. |
| Iterative sampling | `utils/sampler.py` | Every reverse step calls `model(x, t, cond)`, so LKPN sees the current `z_t`. |
| Inference orchestration | `utils/inference.py`, `utils/pipeline.py` | CLI loads checkpoint/config, builds conditions, runs spaced sampling, decodes VAE output, and applies color correction. |
| Dataset pair contract | `dataset/codeformer.py` | Reads `HR` image path from a list and derives the paired blurred path by replacing `HR` with `Blur`. |

## Runtime Flow

Training:

1. `train.py` loads `configs/train/train.yaml`.
2. `ControlLDM` is instantiated.
3. SD v2.1 weights are loaded into UNet/VAE/CLIP and frozen.
4. ControlNet is initialized from the frozen UNet if no resume checkpoint is
   provided.
5. Only `cldm.controlnet.parameters()` and `cldm.kpn.parameters()` are passed
   to AdamW.
6. Dataset returns `gt`, `lq`, and `prompt`.
7. `gt` is encoded to `z_0`; `lq` is encoded as `c_img`.
8. `Diffusion.p_losses()` samples `z_t`, calls `ControlLDM.forward()`, and
   optimizes denoising plus latent KPN reconstruction loss.

Inference:

1. `inference.py` parses CLI args and delegates to `utils.inference.InferenceLoop`.
2. `InferenceLoop` loads `configs/inference/cldm.yaml`,
   `configs/inference/diffusion.yaml`, and the model checkpoint.
3. `utils.pipeline.Pipeline.run()` converts input RGB images to tensors.
4. `Pipeline.run_diff()` pads to multiples of 64, prepares conditional and
   unconditional latents/text prompts, optionally creates a better start from
   low-frequency wavelet content, then calls `SpacedSampler.sample()`.
5. `SpacedSampler.sample()` iterates timesteps from high to low. Each step
   calls `ControlLDM.forward()`, which runs LKPN/EAC, ControlNet, then the
   controlled UNet noise predictor.
6. The final latent is decoded by the VAE and color-adjusted in `pipeline.py`.

## Important Implementation Details

- `ControlLDM.forward()` returns both predicted noise and `lr_kpn`. Training
  uses both; inference only uses the noise prediction for the reverse step.
- `ControlNet` expects an 8-channel hint because it receives
  `torch.cat((c_img, lr_kpn), dim=1)`.
- LKPN's EAC implementation requires CUDA tensors and CuPy. CPU/MPS inference
  may pass CLI device checks but the EAC path raises `NotImplementedError`.
- Tiled sampling blends predicted noise with Gaussian weights, but tiled
  guidance is explicitly unsupported.
- `configs/train/train.yaml` has machine-specific absolute paths and differs
  from the paper's reported training hyperparameters.

## Known Paper-Code Differences

- The paper describes `L_LKPN = L_latent + L_pixel`; this repository currently
  implements the latent KPN MSE term in `model/gaussian_diffusion.py` and does
  not decode KPN output for a pixel-space loss.
- The paper reports batch size 128, learning rate `5e-5`, and 100K iterations.
  The checked-in config uses batch size 64, learning rate `1e-4`, and 200K
  steps.
- The dataset config lists synthetic degradation parameters, but the active
  dataset path reads pre-generated `HR`/`Blur` image pairs and does not apply
  those degradation utilities inside `CodeformerDataset.__getitem__()`.

## Change-Safety Notes

- Before editing `model/lkpn.py`, verify output channel math remains
  `latent_channels * kernel_size * kernel_size`.
- Before editing `model/cldm.py`, verify the condition channel count matches
  `controlnet_cfg.hint_channels`.
- Before editing dataset logic, verify `train.py` still receives GT in `[-1, 1]`
  and LQ in `[0, 1]`.
- Before changing checkpoint loading, verify compatibility with both SD
  checkpoint keys and full `ControlLDM.state_dict()` inference checkpoints.
