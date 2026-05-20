from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PATCH_MODELS = [
    ("G2", "resnet18"),
    ("G3", "resnet18_cbam"),
    ("G4", "ps_resnet18"),
    ("G5", "ps_resnet18_cbam"),
    ("G6", "ps_resnet34_cbam"),
    ("G7", "ps_resnet50_cbam"),
    ("G8", "mapnet_like"),
    ("G9", "final_multiscale"),
    ("G10", "g10_hspm_dabm"),
]

PATCH_MODEL_NAMES = {model for group, model in PATCH_MODELS if group != "G8"}


def run(command: list[str], log_path: Path, dry_run: bool = False) -> str:
    text = "$ " + " ".join(command)
    print("\n" + text, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
    if dry_run:
        return ""
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    output = (completed.stdout or "") + (completed.stderr or "")
    print(output, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(output + "\n")
    return output


def parse_last_json(output: str) -> dict:
    decoder = json.JSONDecoder()
    last = None
    for idx, char in enumerate(output):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(output[idx:])
        except json.JSONDecodeError:
            continue
        last = obj
    return last or {}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def model_batch_size(model: str, default_batch_size: int) -> int:
    if model in {"ps_resnet50_cbam", "final_multiscale", "g10_hspm_dabm"} and default_batch_size >= 32:
        return 16
    return default_batch_size


def build_index(args: argparse.Namespace, log_path: Path) -> None:
    run(
        [
            sys.executable,
            "dl/scripts/build_index.py",
            "--data-root",
            str(args.data_root),
            "--out",
            str(args.pairs),
        ],
        log_path,
        args.dry_run,
    )


def optuna_trials_for_group(group: str, args: argparse.Namespace) -> int:
    if args.optuna_trials > 0:
        return args.optuna_trials
    if group in {"G2", "G3", "G4", "G5"}:
        return 30
    if group in {"G6", "G7"}:
        return 50
    if group in {"G9", "G10"}:
        return 80
    return 0


def run_optuna(args: argparse.Namespace, group: str, model: str, log_path: Path) -> dict:
    out_path = args.out_dir / "optuna" / f"{group}_{model}_best.json"
    if args.skip_existing and out_path.exists():
        return read_json(out_path)
    trials = optuna_trials_for_group(group, args)
    command = [
        sys.executable,
        "dl/scripts/optimize_hparams.py",
        "--pairs",
        str(args.pairs),
        "--model",
        model,
        "--patch-size",
        str(args.patch_size),
        "--trials",
        str(trials),
        "--epochs-per-trial",
        str(args.optuna_epochs_per_trial),
        "--samples-per-trial",
        str(args.optuna_samples_per_trial),
        "--val-samples",
        str(args.optuna_val_samples),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--out",
        str(out_path),
    ]
    run(command, log_path, args.dry_run)
    return {} if args.dry_run else read_json(out_path)


def training_args_from_optuna(best: dict) -> list[str]:
    train_args = best.get("train_args", {})
    command: list[str] = []
    mapping = {
        "lr": "--lr",
        "weight_decay": "--weight-decay",
        "dropout": "--dropout",
        "batch_size": "--batch-size",
        "cbam_reduction": "--cbam-reduction",
        "attention_placement": "--attention-placement",
        "rotation_deg": "--rotation-deg",
        "scale_min": "--scale-min",
        "scale_max": "--scale-max",
        "translation_px": "--translation-px",
        "final_backbone": "--final-backbone",
    }
    for key, flag in mapping.items():
        value = train_args.get(key)
        if value is not None:
            command.extend([flag, str(value)])

    combo = train_args.get("patch_size_combination")
    if combo:
        command.extend(["--patch-sizes", str(combo)])
        parts = [int(v) for v in str(combo).split(",")]
        if len(parts) == 2 and train_args.get("patch_fusion_weight_first") is not None:
            w1 = float(train_args["patch_fusion_weight_first"])
            command.extend(["--fusion-weights", f"{w1},{1.0 - w1}"])
        elif len(parts) == 3:
            w1 = float(train_args.get("patch_fusion_weight_64") or 0.33)
            w2 = float(train_args.get("patch_fusion_weight_128") or 0.33)
            w3 = max(1.0 - w1 - w2, 0.1)
            command.extend(["--fusion-weights", f"{w1},{w2},{w3}"])
    return command


def train_model(args: argparse.Namespace, group: str, model: str, run_name: str, log_path: Path, optuna_best: dict | None = None) -> None:
    checkpoint = args.out_dir / "runs" / run_name / "checkpoints" / "best.pt"
    if args.skip_existing and checkpoint.exists():
        print(f"Skip existing checkpoint: {checkpoint}")
        return
    command = [
        sys.executable,
        "dl/scripts/train_patch_matcher.py",
        "--pairs",
        str(args.pairs),
        "--model",
        model,
        "--epochs",
        str(args.epochs),
        "--samples-per-epoch",
        str(args.samples_per_epoch),
        "--val-samples",
        str(args.val_samples),
        "--batch-size",
        str(model_batch_size(model, args.batch_size)),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--run-name",
        run_name,
        "--seed",
        str(args.seed),
        "--out-dir",
        str(args.out_dir),
    ]
    if args.amp:
        command.append("--amp")
    if args.save_every > 0:
        command.extend(["--save-every", str(args.save_every)])
    if optuna_best:
        command.extend(training_args_from_optuna(optuna_best))
    run(command, log_path, args.dry_run)


def evaluate_model(args: argparse.Namespace, model: str, run_name: str, log_path: Path) -> dict:
    checkpoint = args.out_dir / "runs" / run_name / "checkpoints" / "best.pt"
    output = run(
        [
            sys.executable,
            "dl/scripts/evaluate_patch_matcher.py",
            "--pairs",
            str(args.pairs),
            "--checkpoint",
            str(checkpoint),
            "--model",
            model,
            "--split",
            "test",
            "--samples",
            str(args.eval_samples),
            "--batch-size",
            str(args.eval_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
        ],
        log_path,
        args.dry_run,
    )
    return parse_last_json(output)


def register_samples(args: argparse.Namespace, model: str, run_name: str, log_path: Path) -> list[dict]:
    checkpoint = args.out_dir / "runs" / run_name / "checkpoints" / "best.pt"
    results = []
    for sample_id in args.registration_sample_ids:
        output = run(
            [
                sys.executable,
                "dl/scripts/register_pair.py",
                "--pairs",
                str(args.pairs),
                "--checkpoint",
                str(checkpoint),
                "--model",
                model,
                "--sample-id",
                str(sample_id),
                "--device",
                args.device,
                "--threshold",
                str(args.registration_threshold),
                "--grid-step",
                str(args.registration_grid_step),
                "--search-radius",
                str(args.registration_search_radius),
            ],
            log_path,
            args.dry_run,
        )
        result = parse_last_json(output)
        if result:
            results.append(result)
    return results


def register_g10_samples(args: argparse.Namespace, model: str, run_name: str, log_path: Path) -> list[dict]:
    checkpoint = args.out_dir / "runs" / run_name / "checkpoints" / "best.pt"
    results = []
    for sample_id in args.registration_sample_ids:
        output = run(
            [
                sys.executable,
                "dl/scripts/register_g10_hspm.py",
                "--pairs",
                str(args.pairs),
                "--checkpoint",
                str(checkpoint),
                "--model",
                model,
                "--sample-id",
                str(sample_id),
                "--device",
                args.device,
                "--region-size",
                str(args.g10_region_size),
                "--block-size",
                str(args.g10_block_size),
                "--region-step",
                str(args.g10_region_step),
                "--block-step",
                str(args.g10_block_step),
                "--search-radius",
                str(args.g10_search_radius),
                "--threshold",
                str(args.g10_threshold),
                "--top-k",
                str(args.g10_top_k),
                "--temperature",
                str(args.g10_temperature),
                "--batch-size",
                str(args.g10_batch_size),
            ],
            log_path,
            args.dry_run,
        )
        result = parse_last_json(output)
        if result:
            results.append(result)
    return results


def run_g8_mapnet(args: argparse.Namespace, log_path: Path) -> list[dict]:
    results = []
    for sample_id in args.registration_sample_ids:
        command = [
            sys.executable,
            "dl/scripts/run_mapnet_g8.py",
            "--pairs",
            str(args.pairs),
            "--sample-id",
            str(sample_id),
            "--device",
            args.device,
            "--max-size",
            str(args.g8_max_size),
            "--max-points",
            str(args.g8_max_points),
            "--pca-dim",
            str(args.g8_pca_dim),
            "--ratio",
            str(args.g8_ratio),
            "--ransac-thresh",
            str(args.g8_ransac_thresh),
        ]
        if args.g8_pretrained:
            command.append("--pretrained")
        output = run(command, log_path, args.dry_run)
        result = parse_last_json(output)
        if result:
            results.append(result)
    return results


def write_summary(summary_path: Path, records: list[dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full SAR-Optical DL pipeline end to end.")
    parser.add_argument("--data-root", type=Path, default=Path("dl/data/extracted"))
    parser.add_argument("--pairs", type=Path, default=Path("dl/outputs/pairs.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("dl/outputs"))
    parser.add_argument("--models", default="all", help="all or comma-separated model names.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=50000)
    parser.add_argument("--val-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--registration-sample-ids", default="100,300,500,610")
    parser.add_argument("--registration-threshold", type=float, default=0.7)
    parser.add_argument("--registration-grid-step", type=int, default=128)
    parser.add_argument("--registration-search-radius", type=int, default=32)
    parser.add_argument("--g8-max-size", type=int, default=512)
    parser.add_argument("--g8-max-points", type=int, default=600)
    parser.add_argument("--g8-pca-dim", type=int, default=64)
    parser.add_argument("--g8-ratio", type=float, default=0.95)
    parser.add_argument("--g8-ransac-thresh", type=float, default=5.0)
    parser.add_argument("--g8-pretrained", action="store_true")
    parser.add_argument("--g10-region-size", type=int, default=256)
    parser.add_argument("--g10-block-size", type=int, default=128)
    parser.add_argument("--g10-region-step", type=int, default=256)
    parser.add_argument("--g10-block-step", type=int, default=128)
    parser.add_argument("--g10-search-radius", type=int, default=192)
    parser.add_argument("--g10-threshold", type=float, default=0.55)
    parser.add_argument("--g10-top-k", type=int, default=4)
    parser.add_argument("--g10-temperature", type=float, default=0.05)
    parser.add_argument("--g10-batch-size", type=int, default=64)
    parser.add_argument("--skip-registration", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--use-optuna", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=0, help="0 uses group defaults: G2-G5 30, G6-G7 50, G9 80.")
    parser.add_argument("--optuna-epochs-per-trial", type=int, default=2)
    parser.add_argument("--optuna-samples-per-trial", type=int, default=4096)
    parser.add_argument("--optuna-val-samples", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.registration_sample_ids = [int(x.strip()) for x in args.registration_sample_ids.split(",") if x.strip()]

    if args.models == "all":
        selected = PATCH_MODELS
    else:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        selected = [(g, m) for g, m in PATCH_MODELS if m in wanted or g in wanted]
        if not selected:
            raise SystemExit(f"No models selected from: {args.models}")

    pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = args.out_dir / "full_pipeline" / pipeline_id
    log_path = pipeline_dir / "pipeline.log"
    summary_path = pipeline_dir / "summary.json"

    pipeline_config = vars(args).copy()
    pipeline_config["registration_sample_ids"] = args.registration_sample_ids
    pipeline_config["selected_models"] = selected
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    with (pipeline_dir / "pipeline_config.json").open("w", encoding="utf-8") as f:
        json.dump(pipeline_config, f, indent=2, default=str)

    build_index(args, log_path)

    summary_records: list[dict] = []
    for group, model in selected:
        run_name = f"{group}_{model}"
        if group == "G8":
            reg_results = [] if args.skip_registration else run_g8_mapnet(args, log_path)
            record = {
                "group": group,
                "model": model,
                "run_name": run_name,
                "checkpoint": None,
                "test_patch_metrics": None,
                "registration_results": reg_results,
                "note": "G8 is an image-based MAP-Net-like literature baseline; it is not patch-trained.",
            }
            summary_records.append(record)
            write_summary(summary_path, summary_records)
            continue

        optuna_best = run_optuna(args, group, model, log_path) if args.use_optuna else None
        train_model(args, group, model, run_name, log_path, optuna_best=optuna_best)
        eval_metrics = evaluate_model(args, model, run_name, log_path)
        if args.skip_registration:
            reg_results = []
        elif group == "G10":
            reg_results = register_g10_samples(args, model, run_name, log_path)
        else:
            reg_results = register_samples(args, model, run_name, log_path)
        record = {
            "group": group,
            "model": model,
            "run_name": run_name,
            "checkpoint": str(args.out_dir / "runs" / run_name / "checkpoints" / "best.pt"),
            "test_patch_metrics": eval_metrics,
            "registration_results": reg_results,
            "optuna": optuna_best,
        }
        summary_records.append(record)
        write_summary(summary_path, summary_records)

    print(f"\nFull pipeline summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
