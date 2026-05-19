from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import PatchMatchingDataset
from src.data.pairs import read_pairs_csv
from src.models import build_model
from src.utils.metrics import binary_metrics, count_parameters


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _parse_float_tuple(value: str | None) -> tuple[float, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(float(v.strip()) for v in value.split(",") if v.strip())


def _jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    data = vars(args).copy()
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    loss_total = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            opt = batch["opt"].to(device)
            target = batch["label"].to(device)
            logits = model(sar, opt)
            loss_total += float(criterion(logits, target).item()) * sar.size(0)
            prob = torch.sigmoid(logits).detach().cpu().flatten().tolist()
            scores.extend(prob)
            labels.extend(target.detach().cpu().flatten().tolist())
    metrics = binary_metrics(labels, scores)
    metrics["loss"] = loss_total / max(len(labels), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("dl/outputs/pairs.csv"))
    parser.add_argument("--model", default="ps_resnet18_cbam")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=50000)
    parser.add_argument("--val-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--cbam-reduction", type=int, default=16)
    parser.add_argument("--attention-placement", default="layer4")
    parser.add_argument("--final-backbone", default="resnet34", choices=["resnet34", "resnet50"])
    parser.add_argument("--patch-sizes", default="64,128,256")
    parser.add_argument("--fusion-weights", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--rotation-deg", type=float, default=10.0)
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.1)
    parser.add_argument("--translation-px", type=float, default=4.0)
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--optical-contrast", type=float, default=0.15)
    parser.add_argument("--optical-brightness", type=float, default=12.0)
    parser.add_argument("--out-dir", type=Path, default=Path("dl/outputs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.model}_{timestamp}"
    run_dir = args.out_dir / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable_args(args), f, indent=2)

    train_records = read_pairs_csv(args.pairs, split="train")
    val_records = read_pairs_csv(args.pairs, split="val")
    if not train_records or not val_records:
        raise SystemExit("Run build_index.py first and ensure train/val records exist.")

    train_ds = PatchMatchingDataset(
        train_records,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        augment=not args.no_augment,
        rotation_deg=args.rotation_deg,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        translation_px=args.translation_px,
        flip_prob=args.flip_prob,
        optical_contrast=args.optical_contrast,
        optical_brightness=args.optical_brightness,
    )
    val_ds = PatchMatchingDataset(
        val_records,
        patch_size=args.patch_size,
        samples_per_epoch=args.val_samples,
        augment=False,
    )
    pin_memory = args.device.startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    model = build_model(
        args.model,
        dropout=args.dropout,
        cbam_reduction=args.cbam_reduction,
        attention_placement=args.attention_placement,
        final_backbone=args.final_backbone,
        patch_sizes=_parse_int_tuple(args.patch_sizes),
        fusion_weights=_parse_float_tuple(args.fusion_weights),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_f1 = -1.0
    start_epoch = 1
    history: list[dict[str, float | int]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state"])
        best_f1 = float(checkpoint.get("best_f1", checkpoint.get("metrics", {}).get("f1", -1.0)))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history = checkpoint.get("history", [])

    print(json.dumps({"model": args.model, "params": count_parameters(model), "device": str(device), "run_dir": str(run_dir)}, indent=2))
    log_path = run_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            sar = batch["sar"].to(device)
            opt = batch["opt"].to(device)
            target = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(sar, opt)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * sar.size(0)
            seen += sar.size(0)
            pbar.set_postfix(loss=total_loss / max(seen, 1))

        metrics = run_eval(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(seen, 1)
        history.append(metrics)
        print(json.dumps(metrics, indent=2))
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        state = {
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "args": _jsonable_args(args),
            "metrics": metrics,
            "best_f1": max(best_f1, metrics["f1"]),
            "epoch": epoch,
            "history": history,
        }
        torch.save(state, ckpt_dir / "last.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pt")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            state["best_f1"] = best_f1
            torch.save(state, ckpt_dir / "best.pt")

    with (run_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
