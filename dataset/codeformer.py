from typing import Dict, Mapping, Any, Optional, Tuple, Union
import io

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data

from dataset.utils import load_file_list
from utils.common import instantiate_from_config


def _resize_pair_to_min_size(
        img_gt: np.ndarray,
        img_blur: np.ndarray,
        min_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img_gt.shape[:2]
    if min(h, w) >= min_size:
        return img_gt, img_blur

    scale = min_size / min(h, w)
    size = (round(w * scale), round(h * scale))
    img_gt = cv2.resize(img_gt, size, interpolation=cv2.INTER_CUBIC)
    img_blur = cv2.resize(img_blur, size, interpolation=cv2.INTER_CUBIC)
    return img_gt, img_blur


def _paired_crop(
        img_gt: np.ndarray,
        img_blur: np.ndarray,
        out_size: int,
        crop_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    if img_gt.shape[:2] != img_blur.shape[:2]:
        raise ValueError(
            f"GT and blur image sizes must match before paired crop: "
            f"{img_gt.shape[:2]} vs {img_blur.shape[:2]}"
        )

    if crop_type == "none":
        return img_gt, img_blur

    img_gt, img_blur = _resize_pair_to_min_size(img_gt, img_blur, out_size)
    h, w = img_gt.shape[:2]

    if crop_type == "center":
        top = (h - out_size) // 2
        left = (w - out_size) // 2
    elif crop_type == "random":
        top = np.random.randint(0, h - out_size + 1)
        left = np.random.randint(0, w - out_size + 1)
    else:
        raise ValueError(f"Unsupported crop_type: {crop_type}")

    return (
        img_gt[top:top + out_size, left:left + out_size],
        img_blur[top:top + out_size, left:left + out_size],
    )


class CodeformerDataset(data.Dataset):

    def __init__(
            self,
            file_list: str,
            file_backend_cfg: Mapping[str, Any],
            out_size: int,
            crop_type: str
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        self.file_list = file_list
        self.image_files = load_file_list(file_list)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        assert self.crop_type == "none" or self.out_size > 0

    def load_gt_image(self, image_path: str, max_retry: int = 5) -> Optional[np.ndarray]:

        image_bytes = self.file_backend.get(image_path)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)

        return image

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        img_gt = None
        while img_gt is None:
            # load meta file
            image_file = self.image_files[index]

            gt_path = image_file["image_path"]
            blur_path = gt_path.replace("HR", "Blur")
            prompt = image_file["prompt"]
            img_gt = self.load_gt_image(gt_path)
            img_blur = self.load_gt_image(blur_path)

        img_gt, img_blur = _paired_crop(img_gt, img_blur, self.out_size, self.crop_type)

        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        img_blur = (img_blur[..., ::-1] / 255.0).astype(np.float32)

        if np.random.uniform() < 0.5:
            prompt = ""

        # BGR to RGB, [-1, 1]
        gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        lq = img_blur[..., ::-1].astype(np.float32)

        return gt, lq, prompt

    def __len__(self) -> int:
        return len(self.image_files)
