from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import PatchMatchingDataset
from src.data.pairs import read_pairs_csv
from src.models import build_model
from src.utils.metrics import binary_metrics, count_parameters


def _parse_int_tuple(value: str | None) -> tuple[int, ...]:
    if value is None:
        return (64, 128, 256)
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _parse_float_tuple(value: str | None) -> tuple[float, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(float(v.strip()) for v in value.split(",") if v.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("dl/outputs/pairs.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = args.model or checkpoint.get("model_name")
    ckpt_args = checkpoint.get("args", {})
    records = read_pairs_csv(args.pairs, split=args.split)
    dataset = PatchMatchingDataset(records, patch_size=args.patch_size, samples_per_epoch=args.samples, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = build_model(
        model_name,
        dropout=float(ckpt_args.get("dropout", 0.2)),
        cbam_reduction=int(ckpt_args.get("cbam_reduction", 16)),
        attention_placement=ckpt_args.get("attention_placement", "layer4"),
        final_backbone=ckpt_args.get("final_backbone", "resnet34"),
        patch_sizes=_parse_int_tuple(ckpt_args.get("patch_sizes", "64,128,256")),
        fusion_weights=_parse_float_tuple(ckpt_args.get("fusion_weights")),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    y_true: list[float] = []
    y_score: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["sar"].to(device), batch["opt"].to(device))
            y_score.extend(torch.sigmoid(logits).cpu().flatten().tolist())
            y_true.extend(batch["label"].cpu().flatten().tolist())
    metrics = binary_metrics(y_true, y_score)
    metrics["parameter_count"] = count_parameters(model)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
