import os
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from dataset.paired_dir import PairedDirDataset
from model import ControlLDM, Diffusion
from model.cldm import disabled_train
from utils.common import instantiate_from_config
from utils.pipeline import Pipeline


def load_checkpoint_state_dict(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint at {path} is not a state dict")

    keys = list(checkpoint.keys())
    if keys and all(key.startswith("module.") for key in keys):
        checkpoint = {key[len("module."):]: value for key, value in checkpoint.items()}
    return checkpoint


def freeze_pretrained_sd_modules(cldm: ControlLDM) -> None:
    for module in [cldm.vae, cldm.clip, cldm.unet]:
        module.eval()
        module.train = disabled_train
        for p in module.parameters():
            p.requires_grad = False


def build_validation_steps(first_step: int, max_steps: int, num_runs: int) -> List[int]:
    if first_step <= 0 or max_steps <= 0 or num_runs <= 0:
        return []
    first_step = min(first_step, max_steps)
    if num_runs == 1 or first_step == max_steps:
        return [first_step]
    return sorted({
        int(round(step))
        for step in np.linspace(first_step, max_steps, num_runs)
        if 1 <= int(round(step)) <= max_steps
    })


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params, trainable_params


def init_swanlab(cfg: DictConfig, exp_dir: str, is_main_process: bool):
    swanlab_cfg = cfg.get("swanlab", {})
    if not is_main_process or not swanlab_cfg.get("enabled", False):
        return None

    import swanlab

    logdir = swanlab_cfg.get("logdir") or os.path.join(exp_dir, "swanlog")
    swanlab.init(
        project=swanlab_cfg.get("project", "DeblurDiff"),
        experiment_name=swanlab_cfg.get("run_name"),
        mode=swanlab_cfg.get("mode", "cloud"),
        logdir=logdir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return swanlab


def log_swanlab(swanlab, data: Dict[str, object], step: int) -> None:
    if swanlab is not None and data:
        swanlab.log(data, step=step)


def estimate_forward_flops(
        cldm: ControlLDM,
        device: torch.device,
        image_size: int = 256
) -> Optional[float]:
    was_training = cldm.training
    try:
        from torch.profiler import ProfilerActivity, profile

        cldm.eval()
        clean = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32, device=device)
        x_noisy = torch.randn((1, 4, image_size // 8, image_size // 8), dtype=torch.float32, device=device)
        t = torch.zeros((1,), dtype=torch.long, device=device)
        with torch.no_grad():
            cond = cldm.prepare_condition(clean, [""])
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with torch.no_grad(), profile(activities=activities, with_flops=True) as prof:
            cldm(x_noisy, t, cond)
        return float(sum(getattr(evt, "flops", 0) or 0 for evt in prof.key_averages()))
    except Exception as exc:
        print(f"Failed to estimate FLOPs: {exc}")
        return None
    finally:
        if was_training:
            cldm.train()


def make_validation_loaders(cfg: DictConfig) -> Dict[str, DataLoader]:
    validation_cfg = cfg.get("validation", {})
    if not validation_cfg.get("enabled", False):
        return {}

    loaders = {}
    batch_size = validation_cfg.get("batch_size", 1)
    num_workers = validation_cfg.get("num_workers", 0)
    for split in ("val", "visual"):
        split_cfg = validation_cfg.get(split, {})
        if not split_cfg.get("enabled", True):
            continue
        dataset = PairedDirDataset(
            sharp_dir=split_cfg.sharp_dir,
            blur_dir=split_cfg.blur_dir,
            max_images=split_cfg.get("max_images"),
        )
        loaders[split] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )
    return loaders


def _metric_tensors(restored: np.ndarray, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    restored = torch.from_numpy(restored).float().div(255.0)
    restored = rearrange(restored, "b h w c -> b c h w").contiguous()
    gt = rearrange(gt.float(), "b h w c -> b c h w").contiguous()
    gt = (gt + 1.0).div(2.0).clamp(0, 1)
    return restored, gt


def _lq_tensor(lq: torch.Tensor) -> torch.Tensor:
    return rearrange(lq.float(), "b h w c -> b c h w").contiguous().clamp(0, 1)


def _swanlab_grid_image(grid: torch.Tensor) -> np.ndarray:
    grid = rearrange(grid.detach().cpu().clamp(0, 1), "c h w -> h w c")
    return np.round(grid.numpy() * 255.0).astype(np.uint8)


def evaluate_loader(
        pipeline: Pipeline,
        loader: DataLoader,
        validation_cfg: DictConfig,
        collect_images: bool,
        swanlab=None,
) -> Tuple[float, float, List[object]]:
    from torchmetrics.functional.image import peak_signal_noise_ratio
    from torchmetrics.functional.image import structural_similarity_index_measure

    psnr_sum = 0.0
    ssim_sum = 0.0
    num_images = 0
    images = []
    for batch in loader:
        lq_np = batch["lq"].numpy()
        lq_uint8 = np.clip(np.round(lq_np * 255.0), 0, 255).astype(np.uint8)
        restored_np = pipeline.run(
            lq_uint8,
            validation_cfg.get("steps", 50),
            validation_cfg.get("strength", 1.0),
            validation_cfg.get("tiled", False),
            validation_cfg.get("tile_size", 512),
            validation_cfg.get("tile_stride", 256),
            validation_cfg.get("pos_prompt", ""),
            validation_cfg.get("neg_prompt", "low quality, blurry, low-resolution, noisy, unsharp, weird textures"),
            validation_cfg.get("cfg_scale", 1.0),
            validation_cfg.get("better_start", False),
            progress=False,
        )
        restored, gt = _metric_tensors(restored_np, batch["gt"])
        batch_size = restored.shape[0]
        psnr_sum += peak_signal_noise_ratio(restored, gt, data_range=1.0).item() * batch_size
        ssim_sum += structural_similarity_index_measure(restored, gt, data_range=1.0).item() * batch_size
        num_images += batch_size

        if collect_images and swanlab is not None:
            lq = _lq_tensor(batch["lq"])
            for i, name in enumerate(batch["name"]):
                grid = make_grid(torch.stack([lq[i], restored[i], gt[i]], dim=0), nrow=3)
                images.append(swanlab.Image(_swanlab_grid_image(grid), caption=str(name)))

    return psnr_sum / num_images, ssim_sum / num_images, images


def run_validation(
        cldm: ControlLDM,
        diffusion: Diffusion,
        loaders: Dict[str, DataLoader],
        validation_cfg: DictConfig,
        device: torch.device,
        swanlab,
        step: int,
) -> None:
    if not loaders:
        return

    was_training = cldm.training
    cldm.eval()
    pipeline = Pipeline(cldm, diffusion, None, device)
    log_data = {}
    with torch.no_grad():
        if "val" in loaders:
            psnr, ssim, _ = evaluate_loader(pipeline, loaders["val"], validation_cfg, False)
            log_data.update({"metrics/psnr": psnr, "metrics/ssim": ssim})
        if "visual" in loaders:
            psnr, ssim, images = evaluate_loader(pipeline, loaders["visual"], validation_cfg, True, swanlab)
            log_data.update({"visual/vis_psnr": psnr, "visual/vis_ssim": ssim})
            if images:
                log_data["visual/pictures"] = images
    log_swanlab(swanlab, log_data, step)
    if was_training:
        cldm.train()


def mean_gathered_tensors(accelerator: Accelerator, values: List[torch.Tensor]) -> float:
    local = torch.stack(values).reshape(-1)
    gathered = accelerator.gather(local)
    return gathered.float().mean().item()


def log_txt_as_img(wh, xc):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        # font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    exp_dir = cfg.train.exp_dir

    # Setup an experiment folder:
    if accelerator.is_local_main_process:
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")
    swanlab = init_swanlab(cfg, exp_dir, accelerator.is_local_main_process)

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    resume_full = cfg.train.get("resume_full")
    resume = cfg.train.get("resume")
    resume_kpn = cfg.train.get("resume_kpn")
    if resume_full:
        cldm.load_state_dict(load_checkpoint_state_dict(resume_full), strict=True)
        freeze_pretrained_sd_modules(cldm)
        if accelerator.is_local_main_process:
            print(f"strictly load full ControlLDM weight from checkpoint: {resume_full}")
    else:
        sd_path = cfg.train.get("sd_path")
        if not sd_path:
            raise ValueError("train.sd_path is required when train.resume_full is not set")
        sd = load_checkpoint_state_dict(sd_path)
        unused = cldm.load_pretrained_sd(sd)
        if accelerator.is_local_main_process:
            print(f"strictly load pretrained SD weight from {sd_path}\n"
                  f"unused weights: {unused}")

        if resume:
            cldm.load_controlnet_from_ckpt(load_checkpoint_state_dict(resume))
            if accelerator.is_local_main_process:
                print(f"strictly load controlnet weight from checkpoint: {resume}")
        else:
            init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
            if accelerator.is_local_main_process:
                print(f"strictly load controlnet weight from pretrained SD\n"
                      f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                      f"weights initialized from scratch: {init_with_scratch}")

        if resume_kpn:
            sd = load_checkpoint_state_dict(resume_kpn)
            cldm.kpn.load_state_dict(sd, strict=True)
            if accelerator.is_local_main_process:
                print(f"strictly load kpn weight from checkpoint: {resume_kpn}")



    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    global_lr = cfg.train.learning_rate 
    
    controlnet_params = list(cldm.controlnet.parameters())
    kpn_params = list(cldm.kpn.parameters())

    
    opt = torch.optim.AdamW([
        {'params': controlnet_params, 'lr': global_lr, 'name': 'controlnet'},
        {'params': kpn_params, 'lr': global_lr, 'name': 'kpn'}
    ])

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True, drop_last=True
    )
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset):,} images from {dataset.file_list}")
    validation_loaders = make_validation_loaders(cfg) if accelerator.is_local_main_process else {}
    validation_cfg = cfg.get("validation", {})
    validation_steps = set(build_validation_steps(
        validation_cfg.get("first_step", 1000),
        cfg.train.train_steps,
        validation_cfg.get("num_runs", 5),
    )) if validation_cfg.get("enabled", False) else set()

    # Prepare models for training:
    cldm.train().to(device)
#    swinir.eval().to(device)
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    if accelerator.is_local_main_process:
        params, trainable_params = count_parameters(pure_cldm)
        param_log = {
            "param/params": params,
            "param/trainable_params": trainable_params,
        }
        flops = estimate_forward_flops(
            pure_cldm,
            device,
            cfg.get("swanlab", {}).get("flops_image_size", 256),
        )
        if flops is not None:
            param_log["param/FLOPs"] = flops
        log_swanlab(swanlab, param_log, step=0)

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_losses = {}
    epoch = 0
    epoch_loss = []
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for gt, lq, prompt in loader:
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                clean = lq
                cond = pure_cldm.prepare_condition(clean, prompt)
            t = torch.randint(0, diffusion.num_timesteps, (z_0.shape[0],), device=device)

            loss_dict = diffusion.p_losses(cldm, z_0, t, cond, return_dict=True)
            loss = loss_dict["losses/l_total"]
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            for key, value in loss_dict.items():
                step_losses.setdefault(key, []).append(value.detach())
            epoch_loss.append(loss.detach())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}")

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                log_data = {
                    key: mean_gathered_tensors(accelerator, values)
                    for key, values in step_losses.items()
                    if values
                }
                step_losses.clear()
                log_data.update({
                    "learning_rate/lr_controlnet": opt.param_groups[0]["lr"],
                    "learning_rate/lr_kpn": opt.param_groups[1]["lr"],
                })
                if accelerator.is_local_main_process:
                    for key, value in log_data.items():
                        writer.add_scalar(key, value, global_step)
                    log_swanlab(swanlab, log_data, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_local_main_process:
                    checkpoint = pure_cldm.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
 
                    torch.save(checkpoint, ckpt_path)

            if global_step in validation_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    run_validation(
                        pure_cldm, diffusion, validation_loaders, validation_cfg,
                        device, swanlab, global_step
                    )
                accelerator.wait_for_everyone()

            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = mean_gathered_tensors(accelerator, epoch_loss)
        epoch_loss.clear()
        if accelerator.is_local_main_process:
            writer.add_scalar("losses/l_total_epoch", avg_epoch_loss, global_step)

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()
        if swanlab is not None:
            swanlab.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
