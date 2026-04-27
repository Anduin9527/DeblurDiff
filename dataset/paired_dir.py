from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch.utils.data as data


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _collect_images(root: Path) -> List[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    )


def _build_name_index(paths: List[Path]) -> Dict[str, List[Path]]:
    index = {}
    for path in paths:
        index.setdefault(path.name, []).append(path)
    return index


class PairedDirDataset(data.Dataset):
    def __init__(
            self,
            sharp_dir: str,
            blur_dir: str,
            max_images: Optional[int] = None,
    ) -> "PairedDirDataset":
        self.sharp_dir = Path(sharp_dir).expanduser()
        self.blur_dir = Path(blur_dir).expanduser()
        if not self.sharp_dir.is_dir():
            raise FileNotFoundError(f"sharp_dir does not exist: {self.sharp_dir}")
        if not self.blur_dir.is_dir():
            raise FileNotFoundError(f"blur_dir does not exist: {self.blur_dir}")

        sharp_paths = _collect_images(self.sharp_dir)
        blur_by_name = _build_name_index(_collect_images(self.blur_dir))
        pairs = []
        for sharp_path in sharp_paths:
            rel_path = sharp_path.relative_to(self.sharp_dir)
            blur_path = self.blur_dir / rel_path
            if not blur_path.is_file():
                fallback_paths = blur_by_name.get(sharp_path.name, [])
                if len(fallback_paths) > 1:
                    names = ", ".join(str(path) for path in fallback_paths[:5])
                    raise ValueError(f"Duplicate fallback blur matches for {sharp_path}: {names}")
                blur_path = fallback_paths[0] if fallback_paths else None
            if blur_path is None or not blur_path.is_file():
                raise FileNotFoundError(f"Missing blur pair for sharp image: {sharp_path}")
            pairs.append((sharp_path, blur_path, str(rel_path)))

        if max_images is not None:
            pairs = pairs[:max_images]
        if not pairs:
            raise ValueError(f"No paired images found in {self.sharp_dir} and {self.blur_dir}")
        self.pairs = pairs

    def __getitem__(self, index: int):
        sharp_path, blur_path, name = self.pairs[index]
        sharp = np.array(Image.open(sharp_path).convert("RGB"))
        blur = np.array(Image.open(blur_path).convert("RGB"))
        if sharp.shape[:2] != blur.shape[:2]:
            raise ValueError(
                f"Sharp and blur image sizes must match: {sharp_path} {sharp.shape[:2]} "
                f"vs {blur_path} {blur.shape[:2]}"
            )

        gt = (sharp / 127.5 - 1.0).astype(np.float32)
        lq = (blur / 255.0).astype(np.float32)
        return {"gt": gt, "lq": lq, "name": name}

    def __len__(self) -> int:
        return len(self.pairs)
