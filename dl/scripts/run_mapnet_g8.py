from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_pair_images
from src.data.pairs import read_pairs_csv
from src.image_based.mapnet import run_mapnet_like


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G8 MAP-Net-like image-based registration baseline.")
    parser.add_argument("--pairs", type=Path, default=Path("dl/outputs/pairs.csv"))
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--max-points", type=int, default=600)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--ratio", type=float, default=0.95)
    parser.add_argument("--ransac-thresh", type=float, default=5.0)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    records = read_pairs_csv(args.pairs)
    record = next((r for r in records if r.sample_id == args.sample_id), None)
    if record is None:
        raise SystemExit(f"sample id {args.sample_id} not found in pair index")

    sar, opt = load_pair_images(record)
    result = run_mapnet_like(
        sample_id=record.sample_id,
        stem=record.stem,
        sar=sar,
        opt=opt,
        device=torch.device(args.device),
        max_size=args.max_size,
        max_points=args.max_points,
        pca_dim=args.pca_dim,
        ratio=args.ratio,
        ransac_thresh=args.ransac_thresh,
        pretrained=args.pretrained,
    )
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()

