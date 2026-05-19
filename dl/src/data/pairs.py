from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PairRecord:
    sample_id: int
    stem: str
    city: str
    sar_path: str
    opt_path: str
    lab_path: str | None
    split: str


SAR_RE = re.compile(r"(.+)_SAR_(\d+)\.tif$", re.IGNORECASE)
OPT_RE = re.compile(r"(.+)_Optical_(\d+)\.tif$", re.IGNORECASE)
LAB_RE = re.compile(r"(.+)_Label_(\d+)\.tif$", re.IGNORECASE)


def _scan(folder: Path, pattern: re.Pattern[str]) -> dict[int, tuple[str, Path]]:
    records: dict[int, tuple[str, Path]] = {}
    for path in sorted(folder.glob("*.tif")):
        match = pattern.match(path.name)
        if not match:
            continue
        stem, sample_id = match.group(1), int(match.group(2))
        records[sample_id] = (stem, path)
    return records


def build_pair_records(
    data_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> list[PairRecord]:
    sar = _scan(data_root / "SAR_1024", SAR_RE)
    opt = _scan(data_root / "OPT_1024", OPT_RE)
    lab = _scan(data_root / "LAB_1024", LAB_RE)

    complete_ids = sorted(set(sar) & set(opt))
    records: list[PairRecord] = []
    n = len(complete_ids)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    for index, sample_id in enumerate(complete_ids):
        sar_stem, sar_path = sar[sample_id]
        opt_stem, opt_path = opt[sample_id]
        if sar_stem != opt_stem:
            raise ValueError(f"Stem mismatch for id {sample_id}: {sar_stem} vs {opt_stem}")
        if index < train_end:
            split = "train"
        elif index < val_end:
            split = "val"
        else:
            split = "test"
        lab_path = lab.get(sample_id, (None, None))[1]
        records.append(
            PairRecord(
                sample_id=sample_id,
                stem=sar_stem,
                city=sar_stem.split("_")[0],
                sar_path=str(sar_path),
                opt_path=str(opt_path),
                lab_path=str(lab_path) if lab_path else None,
                split=split,
            )
        )
    return records


def write_pairs_csv(records: Iterable[PairRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "stem", "city", "sar_path", "opt_path", "lab_path", "split"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def read_pairs_csv(path: Path, split: str | None = None) -> list[PairRecord]:
    records: list[PairRecord] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split and row["split"] != split:
                continue
            records.append(
                PairRecord(
                    sample_id=int(row["sample_id"]),
                    stem=row["stem"],
                    city=row["city"],
                    sar_path=row["sar_path"],
                    opt_path=row["opt_path"],
                    lab_path=row["lab_path"] or None,
                    split=row["split"],
                )
            )
    return records

