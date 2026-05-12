from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import rasterio

from common import ensure_parent, write_json


def lee_filter(image: np.ndarray, size: int) -> np.ndarray:
    img = image.astype(np.float32)
    mean = cv2.blur(img, (size, size))
    mean_sq = cv2.blur(img * img, (size, size))
    variance = np.maximum(mean_sq - mean * mean, 0)
    noise_var = float(np.percentile(variance[np.isfinite(variance)], 10))
    weights = variance / (variance + noise_var + 1e-12)
    return mean + weights * (img - mean)


def anisotropic_diffusion(image: np.ndarray, iterations: int, kappa: float, gamma: float) -> np.ndarray:
    img = image.astype(np.float32).copy()
    for _ in range(iterations):
        north = np.zeros_like(img)
        south = np.zeros_like(img)
        east = np.zeros_like(img)
        west = np.zeros_like(img)
        north[1:, :] = img[:-1, :] - img[1:, :]
        south[:-1, :] = img[1:, :] - img[:-1, :]
        east[:, :-1] = img[:, 1:] - img[:, :-1]
        west[:, 1:] = img[:, :-1] - img[:, 1:]
        c_n = np.exp(-(north / kappa) ** 2)
        c_s = np.exp(-(south / kappa) ** 2)
        c_e = np.exp(-(east / kappa) ** 2)
        c_w = np.exp(-(west / kappa) ** 2)
        img += gamma * (c_n * north + c_s * south + c_e * east + c_w * west)
    return img


def filter_band(image: np.ndarray, method: str, args: argparse.Namespace) -> np.ndarray:
    mask = np.isfinite(image)
    valid_min = float(np.nanmin(image[mask])) if np.any(mask) else 0.0
    filled = np.where(mask, image, valid_min).astype(np.float32)

    if method == "gaussian":
        out = cv2.GaussianBlur(filled, (0, 0), args.sigma)
    elif method == "median":
        scaled = np.log1p(np.maximum(filled, 0))
        lo, hi = np.percentile(scaled, (1, 99))
        u8 = np.clip((scaled - lo) / max(hi - lo, 1e-6), 0, 1)
        u8 = (u8 * 255).astype(np.uint8)
        med = cv2.medianBlur(u8, args.kernel)
        out = np.expm1(med.astype(np.float32) / 255 * (hi - lo) + lo)
    elif method == "bilateral":
        scaled = np.log1p(np.maximum(filled, 0))
        lo, hi = np.percentile(scaled, (1, 99))
        u8 = np.clip((scaled - lo) / max(hi - lo, 1e-6), 0, 1)
        u8 = (u8 * 255).astype(np.uint8)
        bil = cv2.bilateralFilter(u8, args.diameter, args.sigma_color, args.sigma_space)
        out = np.expm1(bil.astype(np.float32) / 255 * (hi - lo) + lo)
    elif method == "lee":
        out = lee_filter(filled, args.kernel)
    elif method == "diffusion":
        scale = float(np.percentile(filled[mask], 98)) if np.any(mask) else 1.0
        norm = np.clip(filled / max(scale, 1e-6), 0, 5)
        diff = anisotropic_diffusion(norm, args.iterations, args.kappa, args.gamma)
        out = diff * scale
    else:
        raise ValueError(f"Unsupported filter method: {method}")

    out = np.where(mask, out, image)
    return np.maximum(out.astype(np.float32), 0)


def append_log(log_path: str | Path, metrics: dict) -> None:
    lines = [
        "",
        f"## SAR Filtering - {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- method: `{metrics['method']}`",
        f"- input: `{metrics['input']}`",
        f"- output: `{metrics['output']}`",
        f"- nonzero pct: `{metrics['nonzero_pct']:.2f}`",
        f"- p50: `{metrics['p50']:.6f}`",
        f"- p98: `{metrics['p98']:.6f}`",
        f"- max: `{metrics['max']:.6f}`",
        "",
    ]
    with ensure_parent(log_path).open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create filtered SAR variants for registration experiments.")
    parser.add_argument("--input", default="data/processed/sar_10m.tif")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--log", default="EXPERIMENT_LOG.md")
    parser.add_argument("--method", choices=["gaussian", "median", "bilateral", "lee", "diffusion"], required=True)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--diameter", type=int, default=7)
    parser.add_argument("--sigma-color", type=float, default=45.0)
    parser.add_argument("--sigma-space", type=float, default=7.0)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--kappa", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.18)
    args = parser.parse_args()

    with rasterio.open(args.input) as src:
        data = src.read()
        profile = src.profile.copy()

    out = np.empty_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        out[i] = filter_band(data[i], args.method, args)

    output_path = ensure_parent(args.output)
    profile.update(dtype="float32")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out)

    valid = np.isfinite(out)
    metrics = {
        "method": args.method,
        "input": args.input,
        "output": args.output,
        "nonzero_pct": float((out != 0).mean() * 100),
        "p2": float(np.nanpercentile(out[valid], 2)),
        "p50": float(np.nanpercentile(out[valid], 50)),
        "p98": float(np.nanpercentile(out[valid], 98)),
        "max": float(np.nanmax(out)),
        "parameters": {
            "kernel": args.kernel,
            "sigma": args.sigma,
            "diameter": args.diameter,
            "sigma_color": args.sigma_color,
            "sigma_space": args.sigma_space,
            "iterations": args.iterations,
            "kappa": args.kappa,
            "gamma": args.gamma,
        },
    }
    write_json(args.metrics, metrics)
    append_log(args.log, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
