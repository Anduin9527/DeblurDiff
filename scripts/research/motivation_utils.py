from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image


DEFAULT_STEPS = [50, 30, 20, 15, 10, 8, 5]
DEFAULT_NEG_PROMPT = "low quality, blurry, low-resolution, noisy, unsharp, weird textures"


@dataclass(frozen=True)
class ImagePair:
    image_id: str
    input_path: Path
    target_path: Optional[Path]


def repo_root_from_notebook() -> Path:
    """Return the repository root when called from notebooks/ or repo root."""
    cwd = Path.cwd().resolve()
    if (cwd / "configs" / "inference").exists():
        return cwd
    if cwd.name == "notebooks" and (cwd.parent / "configs" / "inference").exists():
        return cwd.parent
    raise RuntimeError(
        "Run the notebook from the DeblurDiff repo root or from the notebooks directory."
    )


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    dirs = {
        "root": output_dir,
        "restored": output_dir / "restored",
        "figures": output_dir / "figures",
        "csv": output_dir / "csv",
        "tensors": output_dir / "tensors",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_image_pairs(input_dir: Path, target_dir: Optional[Path] = None) -> List[ImagePair]:
    input_dir = Path(input_dir)
    target_dir = Path(target_dir) if target_dir is not None else None
    suffixes = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    input_paths = sorted(path for path in input_dir.iterdir() if path.suffix in suffixes)
    pairs: List[ImagePair] = []
    for input_path in input_paths:
        target_path = None
        if target_dir is not None:
            candidate = target_dir / input_path.name
            if candidate.exists():
                target_path = candidate
        pairs.append(ImagePair(input_path.stem, input_path, target_path))
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(path)


def build_pipeline(model_path: Path, device: str = "cuda"):
    """Load DeblurDiff models and return a Pipeline instance for notebook use."""
    from model.gaussian_diffusion import Diffusion
    from model.cldm import ControlLDM
    from utils.common import instantiate_from_config
    from utils.pipeline import Pipeline

    cldm: ControlLDM = instantiate_from_config(OmegaConf.load("configs/inference/cldm.yaml"))
    state = torch.load(str(model_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cldm.load_state_dict(state)
    cldm.eval().to(device)

    diffusion: Diffusion = instantiate_from_config(OmegaConf.load("configs/inference/diffusion.yaml"))
    diffusion.to(device)
    return Pipeline(cldm, diffusion, cond_fn=None, device=device)


def _mean_abs_map(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape {tuple(tensor.shape)}")
    return tensor.abs().mean(dim=1)[0].numpy()


class SamplingTrace:
    """Collect small CPU summaries from the optional sampler trace callback."""

    def __init__(self) -> None:
        self.records: List[Dict[str, float]] = []
        self.maps: Dict[str, List[np.ndarray]] = {
            "eac_residual": [],
            "eps_change": [],
            "pred_x0_change": [],
        }
        self._prev_eps: Optional[torch.Tensor] = None
        self._prev_pred_x0: Optional[torch.Tensor] = None

    def __call__(self, payload: Mapping[str, Any]) -> None:
        eps = payload["eps"].detach().float().cpu()
        pred_x0 = payload["pred_x0"].detach().float().cpu()
        lr_kpn = payload["lr_kpn"]
        c_img = payload["c_img"]
        eac_residual = _mean_abs_map(lr_kpn - c_img)

        if self._prev_eps is None:
            eps_change = np.zeros_like(eac_residual)
        else:
            eps_change = _mean_abs_map(eps - self._prev_eps)
        if self._prev_pred_x0 is None:
            pred_x0_change = np.zeros_like(eac_residual)
        else:
            pred_x0_change = _mean_abs_map(pred_x0 - self._prev_pred_x0)

        self._prev_eps = eps
        self._prev_pred_x0 = pred_x0
        self.maps["eac_residual"].append(eac_residual)
        self.maps["eps_change"].append(eps_change)
        self.maps["pred_x0_change"].append(pred_x0_change)

        timestep = payload["timestep"].detach().cpu().flatten()[0].item()
        self.records.append({
            "step_index": float(payload["step_index"]),
            "total_steps": float(payload["total_steps"]),
            "timestep": float(timestep),
            "eac_residual_mean": float(eac_residual.mean()),
            "eps_change_mean": float(eps_change.mean()),
            "pred_x0_change_mean": float(pred_x0_change.mean()),
            "eps_norm": float(torch.linalg.vector_norm(eps).item()),
            "pred_x0_norm": float(torch.linalg.vector_norm(pred_x0).item()),
        })

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        for name, values in self.maps.items():
            arrays[name] = np.stack(values, axis=0) if values else np.empty((0,))
        arrays["records_json"] = np.array(json.dumps(self.records), dtype=object)
        np.savez_compressed(path, **arrays)


def _to_metric_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image).float().div(255.0)
    tensor = rearrange(tensor, "h w c -> 1 c h w").contiguous()
    return tensor.to(device)


def compute_basic_metrics(restored: np.ndarray, target: Optional[np.ndarray]) -> Dict[str, float]:
    if target is None:
        return {"psnr": float("nan"), "ssim": float("nan")}
    try:
        from torchmetrics.functional.image import peak_signal_noise_ratio
        from torchmetrics.functional.image import structural_similarity_index_measure
    except ImportError as exc:
        raise ImportError("Install torchmetrics to compute PSNR/SSIM in this notebook.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    restored_t = _to_metric_tensor(restored, device)
    target_t = _to_metric_tensor(target, device)
    psnr = peak_signal_noise_ratio(restored_t, target_t, data_range=1.0).item()
    ssim = structural_similarity_index_measure(restored_t, target_t, data_range=1.0).item()
    return {"psnr": float(psnr), "ssim": float(ssim)}


def run_step_sweep(
        pipeline,
        image_pairs: Sequence[ImagePair],
        output_dir: Path,
        steps_list: Sequence[int] = DEFAULT_STEPS,
        seed: int = 231,
        strength: float = 1.0,
        tiled: bool = False,
        tile_size: int = 512,
        tile_stride: int = 256,
        pos_prompt: str = "",
        neg_prompt: str = DEFAULT_NEG_PROMPT,
        cfg_scale: float = 1.0,
        better_start: bool = False,
        capture_trace: bool = True,
        progress: bool = True,
) -> List[Dict[str, Any]]:
    dirs = ensure_output_dirs(output_dir)
    rows: List[Dict[str, Any]] = []
    for pair in image_pairs:
        lq = load_rgb(pair.input_path)
        target = load_rgb(pair.target_path) if pair.target_path is not None else None
        for steps in steps_list:
            set_all_seeds(seed)
            trace = SamplingTrace() if capture_trace else None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            restored = pipeline.run(
                lq[None],
                steps,
                strength,
                tiled,
                tile_size,
                tile_stride,
                pos_prompt,
                neg_prompt,
                cfg_scale,
                better_start,
                progress=progress,
                trace_callback=trace,
            )[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            runtime_sec = time.perf_counter() - start

            restored_path = dirs["restored"] / f"{pair.image_id}_steps{steps}.png"
            save_rgb(restored, restored_path)
            trace_path = dirs["tensors"] / f"{pair.image_id}_steps{steps}_trace.npz"
            if trace is not None:
                trace.save(trace_path)
            else:
                trace_path = None
            metrics = compute_basic_metrics(restored, target)
            rows.append({
                "image_id": pair.image_id,
                "steps": int(steps),
                "nfe": int(steps),
                "runtime_sec": float(runtime_sec),
                "input_path": str(pair.input_path),
                "target_path": str(pair.target_path) if pair.target_path else "",
                "restored_path": str(restored_path),
                "trace_path": str(trace_path) if trace_path is not None else "",
                **metrics,
            })
    write_csv(rows, dirs["csv"] / "step_sweep_metrics.csv")
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_to_dataframe(rows: Sequence[Mapping[str, Any]]):
    try:
        import pandas as pd
    except ImportError:
        return list(rows)
    return pd.DataFrame(rows)


def _imshow(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")


def plot_step_grid(
        image_id: str,
        input_image: np.ndarray,
        target_image: Optional[np.ndarray],
        restored_by_step: Mapping[int, np.ndarray],
        steps_list: Sequence[int],
        save_path: Optional[Path] = None,
) -> plt.Figure:
    panels = [("input", input_image)]
    if target_image is not None:
        panels.append(("target", target_image))
    panels.extend((f"{steps} steps", restored_by_step[steps]) for steps in steps_list if steps in restored_by_step)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2), squeeze=False)
    for ax, (title, image) in zip(axes[0], panels):
        _imshow(ax, image, title)
    fig.suptitle(image_id)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def load_restored_by_step(output_dir: Path, image_id: str, steps_list: Sequence[int]) -> Dict[int, np.ndarray]:
    restored_dir = Path(output_dir) / "restored"
    out: Dict[int, np.ndarray] = {}
    for steps in steps_list:
        path = restored_dir / f"{image_id}_steps{steps}.png"
        if path.exists():
            out[int(steps)] = load_rgb(path)
    return out


def read_trace_npz(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    records = json.loads(str(data["records_json"].item()))
    return {
        "records": records,
        "eac_residual": data["eac_residual"],
        "eps_change": data["eps_change"],
        "pred_x0_change": data["pred_x0_change"],
    }


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.nanpercentile(values, [1, 99])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0, 1)


def resize_heatmap(heatmap: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    img = Image.fromarray((normalize_map(heatmap) * 255).astype(np.uint8))
    img = img.resize((width, height), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def patch_pool_map(heatmap: np.ndarray, patch_size: int = 16) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    h, w = heatmap.shape
    h_crop = h - h % patch_size
    w_crop = w - w % patch_size
    if h_crop == 0 or w_crop == 0:
        return heatmap
    cropped = heatmap[:h_crop, :w_crop]
    pooled = cropped.reshape(h_crop // patch_size, patch_size, w_crop // patch_size, patch_size)
    return pooled.mean(axis=(1, 3))


def image_l1_heatmap(restored: np.ndarray, target: Optional[np.ndarray], latent_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if target is None:
        return None
    diff = np.abs(restored.astype(np.float32) - target.astype(np.float32)).mean(axis=2) / 255.0
    return resize_heatmap(diff, latent_shape)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, percentile: float = 80.0, alpha: float = 0.45) -> np.ndarray:
    heat = resize_heatmap(heatmap, image.shape[:2])
    mask = heat >= np.percentile(heat, percentile)
    overlay = image.astype(np.float32).copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255.0
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * red[mask]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_internal_maps(
        trace_path: Path,
        input_image: np.ndarray,
        target_image: Optional[np.ndarray],
        restored_image: np.ndarray,
        save_path: Optional[Path] = None,
) -> plt.Figure:
    trace = read_trace_npz(trace_path)
    eac = trace["eac_residual"].mean(axis=0)
    eps = trace["eps_change"].mean(axis=0)
    pred = trace["pred_x0_change"].mean(axis=0)
    error = image_l1_heatmap(restored_image, target_image, eac.shape)
    overlay = overlay_heatmap(input_image, eac + eps + pred)

    titles = ["EAC residual", "eps change", "pred_x0 change", "hard overlay"]
    maps = [eac, eps, pred, overlay]
    if error is not None:
        titles.insert(0, "patch error")
        maps.insert(0, error)

    fig, axes = plt.subplots(1, len(maps), figsize=(3.2 * len(maps), 3.2), squeeze=False)
    for ax, title, value in zip(axes[0], titles, maps):
        if value.ndim == 2:
            ax.imshow(value, cmap="magma")
        else:
            ax.imshow(value)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_runtime_quality(rows: Sequence[Mapping[str, Any]], save_path: Optional[Path] = None) -> plt.Figure:
    data = rows_to_dataframe(rows)
    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
    if hasattr(data, "groupby"):
        grouped = data.groupby("steps", as_index=False).agg({
            "runtime_sec": "mean",
            "psnr": "mean",
            "ssim": "mean",
        }).sort_values("steps")
        steps = grouped["steps"].to_numpy()
        runtime = grouped["runtime_sec"].to_numpy()
        psnr = grouped["psnr"].to_numpy()
    else:
        steps = np.array([row["steps"] for row in rows])
        runtime = np.array([row["runtime_sec"] for row in rows])
        psnr = np.array([row["psnr"] for row in rows])
    ax1.plot(steps, runtime, marker="o", label="runtime")
    ax1.set_xlabel("Sampling steps")
    ax1.set_ylabel("Runtime (sec)")
    ax2 = ax1.twinx()
    ax2.plot(steps, psnr, marker="s", color="tab:orange", label="PSNR")
    ax2.set_ylabel("PSNR")
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def check_pyiqa_environment() -> Dict[str, str]:
    import pyiqa
    import torchvision

    info = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pyiqa": getattr(pyiqa, "__version__", "unknown"),
    }
    return info


def run_pyiqa_pair(
        metric_name: str,
        restored_path: Path,
        target_path: Optional[Path] = None,
        device: Optional[str] = None,
) -> float:
    import pyiqa

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = pyiqa.create_metric(metric_name, device=torch.device(device))
    if target_path is None:
        score = metric(str(restored_path))
    else:
        score = metric(str(restored_path), str(target_path))
    if torch.is_tensor(score):
        score = score.detach().cpu().flatten()[0].item()
    return float(score)
