from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import PatchMatchingDataset
from src.data.pairs import read_pairs_csv
from src.models import build_model
from src.utils.metrics import binary_metrics


def _suggest_final_multiscale(trial: optuna.Trial, model_name: str) -> tuple[str, tuple[int, ...], tuple[float, ...] | None]:
    if model_name == "g10_hspm_dabm":
        final_backbone = trial.suggest_categorical("final_backbone", ["resnet34", "resnet50"])
        return final_backbone, (64, 128, 256), None
    if model_name != "final_multiscale":
        return "resnet34", (64, 128, 256), None
    final_backbone = trial.suggest_categorical("final_backbone", ["resnet34", "resnet50"])
    patch_combo = trial.suggest_categorical("patch_size_combination", ["64,128", "128,256", "64,128,256"])
    patch_sizes = tuple(int(v) for v in patch_combo.split(","))
    if len(patch_sizes) == 2:
        first = trial.suggest_categorical("patch_fusion_weight_first", [0.2, 0.3, 0.5])
        fusion_weights = (first, 1.0 - first)
    else:
        w1 = trial.suggest_categorical("patch_fusion_weight_64", [0.2, 0.3, 0.5])
        w2 = trial.suggest_categorical("patch_fusion_weight_128", [0.2, 0.3, 0.5])
        w3 = max(1.0 - w1 - w2, 0.1)
        fusion_weights = (w1, w2, w3)
    return final_backbone, patch_sizes, fusion_weights


def objective(trial: optuna.Trial, args) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    cbam_reduction = trial.suggest_categorical("cbam_reduction", [4, 8, 16])
    attention_placement = trial.suggest_categorical("attention_placement", ["layer2", "layer3", "layer4", "layer3,layer4"])
    rotation_deg = trial.suggest_categorical("rotation_deg", [10.0, 15.0])
    scale_min = trial.suggest_categorical("scale_min", [0.8, 0.9])
    scale_max = 2.0 - scale_min
    translation_px = trial.suggest_categorical("translation_px", [4.0, 8.0])
    final_backbone, patch_sizes, fusion_weights = _suggest_final_multiscale(trial, args.model)

    train_records = read_pairs_csv(args.pairs, split="train")
    val_records = read_pairs_csv(args.pairs, split="val")
    train_ds = PatchMatchingDataset(
        train_records,
        args.patch_size,
        args.samples_per_trial,
        augment=True,
        rotation_deg=rotation_deg,
        scale_min=scale_min,
        scale_max=scale_max,
        translation_px=translation_px,
    )
    val_ds = PatchMatchingDataset(val_records, args.patch_size, args.val_samples, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = build_model(
        args.model,
        dropout=dropout,
        cbam_reduction=cbam_reduction,
        attention_placement=attention_placement,
        final_backbone=final_backbone,
        patch_sizes=patch_sizes,
        fusion_weights=fusion_weights,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs_per_trial):
        model.train()
        for batch in train_loader:
            sar, opt, label = batch["sar"].to(device), batch["opt"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sar, opt), label)
            loss.backward()
            optimizer.step()

    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["sar"].to(device), batch["opt"].to(device))
            scores.extend(torch.sigmoid(logits).cpu().flatten().tolist())
            labels.extend(batch["label"].cpu().flatten().tolist())
    metrics = binary_metrics(labels, scores)
    trial.set_user_attr("metrics", metrics)
    return metrics["f1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("dl/outputs/pairs.csv"))
    parser.add_argument("--model", default="ps_resnet18_cbam")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--epochs-per-trial", type=int, default=2)
    parser.add_argument("--samples-per-trial", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("dl/outputs/optuna_best.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)
    result = {
        "model": args.model,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_metrics": study.best_trial.user_attrs.get("metrics", {}),
        "train_args": {
            "lr": study.best_params.get("lr"),
            "weight_decay": study.best_params.get("weight_decay"),
            "dropout": study.best_params.get("dropout"),
            "batch_size": study.best_params.get("batch_size"),
            "cbam_reduction": study.best_params.get("cbam_reduction"),
            "attention_placement": study.best_params.get("attention_placement"),
            "rotation_deg": study.best_params.get("rotation_deg"),
            "scale_min": study.best_params.get("scale_min"),
            "scale_max": 2.0 - study.best_params["scale_min"] if "scale_min" in study.best_params else None,
            "translation_px": study.best_params.get("translation_px"),
            "final_backbone": study.best_params.get("final_backbone"),
            "patch_size_combination": study.best_params.get("patch_size_combination"),
            "patch_fusion_weight_first": study.best_params.get("patch_fusion_weight_first"),
            "patch_fusion_weight_64": study.best_params.get("patch_fusion_weight_64"),
            "patch_fusion_weight_128": study.best_params.get("patch_fusion_weight_128"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
