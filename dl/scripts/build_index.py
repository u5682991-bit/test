from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.pairs import build_pair_records, write_pairs_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("dl/data/extracted"))
    parser.add_argument("--out", type=Path, default=Path("dl/outputs/pairs.csv"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    records = build_pair_records(args.data_root, args.train_ratio, args.val_ratio)
    write_pairs_csv(records, args.out)
    summary = {
        "out": str(args.out),
        "total_complete_pairs": len(records),
        "splits": {name: sum(r.split == name for r in records) for name in ["train", "val", "test"]},
        "cities": sorted({r.city for r in records}),
        "missing_optical_022_excluded": all(r.sample_id != 22 for r in records),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
