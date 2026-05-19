from __future__ import annotations

import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
from torch.utils.data import Dataset

from .pairs import PairRecord


def _read_sar(path: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as ds:
            return ds.read(1)


def _read_opt(path: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as ds:
            return ds.read().transpose(1, 2, 0)


def _crop(arr: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr[y : y + size, x : x + size]
    return arr[y : y + size, x : x + size, :]


def _augment_sar_patch(
    patch: np.ndarray,
    rotation_deg: float,
    scale_min: float,
    scale_max: float,
    translation_px: float,
) -> np.ndarray:
    h, w = patch.shape
    angle = random.uniform(-rotation_deg, rotation_deg)
    scale = random.uniform(scale_min, scale_max)
    tx = random.uniform(-translation_px, translation_px)
    ty = random.uniform(-translation_px, translation_px)
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[:, 2] += [tx, ty]
    return cv2.warpAffine(patch, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _augment_opt_patch(patch: np.ndarray, contrast: float, brightness: float) -> np.ndarray:
    alpha = random.uniform(1.0 - contrast, 1.0 + contrast)
    beta = random.uniform(-brightness, brightness)
    patch = np.clip(patch.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return patch


def _to_tensor_sar(patch: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(patch.astype(np.float32)[None, :, :] / 255.0)


def _to_tensor_opt(patch: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(patch.astype(np.float32).transpose(2, 0, 1) / 255.0)


class PatchMatchingDataset(Dataset):
    """Online SAR-Optical patch pair generator.

    Positive samples use the same image coordinate. Negative samples use a
    coordinate shifted by at least ``min_negative_offset`` pixels.
    """

    def __init__(
        self,
        records: list[PairRecord],
        patch_size: int = 128,
        samples_per_epoch: int = 50000,
        positive_ratio: float = 0.5,
        min_negative_offset: int = 64,
        augment: bool = True,
        rotation_deg: float = 10.0,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        translation_px: float = 4.0,
        flip_prob: float = 0.5,
        optical_contrast: float = 0.15,
        optical_brightness: float = 12.0,
        seed: int = 42,
    ) -> None:
        self.records = records
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.positive_ratio = positive_ratio
        self.min_negative_offset = min_negative_offset
        self.augment = augment
        self.rotation_deg = rotation_deg
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.translation_px = translation_px
        self.flip_prob = flip_prob
        self.optical_contrast = optical_contrast
        self.optical_brightness = optical_brightness
        self.rng = random.Random(seed)
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load(self, path: str, kind: str) -> np.ndarray:
        key = f"{kind}:{path}"
        if key not in self._cache:
            self._cache[key] = _read_sar(path) if kind == "sar" else _read_opt(path)
        return self._cache[key]

    def _random_xy(self, width: int, height: int) -> tuple[int, int]:
        max_x = width - self.patch_size
        max_y = height - self.patch_size
        return self.rng.randint(0, max_x), self.rng.randint(0, max_y)

    def _negative_xy(self, x: int, y: int, width: int, height: int) -> tuple[int, int]:
        for _ in range(20):
            nx, ny = self._random_xy(width, height)
            if abs(nx - x) + abs(ny - y) >= self.min_negative_offset:
                return nx, ny
        return self._random_xy(width, height)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.rng.choice(self.records)
        sar = self._load(record.sar_path, "sar")
        opt = self._load(record.opt_path, "opt")
        height, width = sar.shape
        x, y = self._random_xy(width, height)
        is_positive = self.rng.random() < self.positive_ratio
        ox, oy = (x, y) if is_positive else self._negative_xy(x, y, width, height)

        sar_patch = _crop(sar, x, y, self.patch_size)
        opt_patch = _crop(opt, ox, oy, self.patch_size)

        if self.augment:
            if self.rng.random() < self.flip_prob:
                sar_patch = np.flip(sar_patch, axis=1).copy()
                opt_patch = np.flip(opt_patch, axis=1).copy()
            sar_patch = _augment_sar_patch(
                sar_patch,
                rotation_deg=self.rotation_deg,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
                translation_px=self.translation_px,
            )
            opt_patch = _augment_opt_patch(
                opt_patch,
                contrast=self.optical_contrast,
                brightness=self.optical_brightness,
            )

        return {
            "sar": _to_tensor_sar(sar_patch),
            "opt": _to_tensor_opt(opt_patch),
            "label": torch.tensor([1.0 if is_positive else 0.0], dtype=torch.float32),
        }


def load_pair_images(record: PairRecord) -> tuple[np.ndarray, np.ndarray]:
    return _read_sar(record.sar_path), _read_opt(record.opt_path)
