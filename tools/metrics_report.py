#!/usr/bin/env python3
"""Summarise training runs from metrics CSV logs and rank models at token milestones."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional


@dataclass
class RunMetadata:
    """Metadata parsed from the CSV filename."""

    model_type: str = "unknown"
    size: str = "unknown"
    batch_size: str = "unknown"
    block_size: str = "unknown"
    precision: str = "unknown"
    compile_mode: str = "unknown"
    optimizer: str = "unknown"
    dataset: str = "unknown"
    date: str = "unknown"
    time: str = "unknown"
    counter: Optional[str] = None
    filename: str = ""

    @property
    def model_label(self) -> str:
        return f"{self.model_type}-{self.size}" if self.size != "unknown" else self.model_type

    @property
    def config_label(self) -> str:
        compile_tag = self.compile_mode
        return f"{self.precision}/{compile_tag}/{self.optimizer}"


@dataclass
class MilestoneRecord:
    milestone: float
    tokens: float
    step: Optional[int]
    train_loss: Optional[float]
    val_loss: Optional[float]
    val_perplexity: Optional[float]
    learning_rate: Optional[float]
    grad_norm: Optional[float]
    timestamp: str
    path: Path
    meta: RunMetadata


ROW_MAP_COMPACT = [
    "step",
    "train_loss",
    "val_loss",
    "val_perplexity",
    "learning_rate",
    "tokens_per_sec",
    "avg_seq_len",
    "total_tokens",
    "grad_norm",
    "timestamp",
]

ROW_MAP_VERBOSE = [
    "step",
    "train_loss",
    "val_loss",
    "val_perplexity",
    "learning_rate",
    "tokens_per_sec",
    "avg_seq_len",
    "batch_size",
    "block_size",
    "total_tokens",
    "grad_norm",
    "timestamp",
]

VALID_NULLS = {"", "n/a", "na", "nan", "none", "null"}


def to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.lower() in VALID_NULLS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: Optional[str]) -> Optional[int]:
    fval = to_float(value)
    if fval is None or math.isnan(fval):
        return None
    return int(fval)


def parse_filename(path: Path) -> RunMetadata:
    stem = path.stem
    parts = stem.split("_")
    counter = None
    if parts and parts[-1].isdigit() and len(parts[-1]) <= 4:
        counter = parts.pop()
    time = parts.pop() if parts else "unknown"
    date = parts.pop() if parts else "unknown"

    model_type = "unknown"
    size = "unknown"
    batch_size = "unknown"
    block_size = "unknown"
    precision = "unknown"
    compile_mode = "unknown"
    optimizer = "unknown"
    dataset_tokens: List[str] = []

    if parts:
        compile_candidates = {"compile", "no-compile"}
        compile_idx = None
        for idx in range(len(parts) - 1, -1, -1):
            if parts[idx] in compile_candidates:
                compile_idx = idx
                break
        if compile_idx is not None and compile_idx >= 4 and compile_idx + 1 < len(parts):
            compile_mode = parts[compile_idx]
            optimizer = parts[compile_idx + 1]
            if compile_idx - 1 >= 0:
                precision = parts[compile_idx - 1]
            if compile_idx - 2 >= 0:
                block_size = parts[compile_idx - 2]
            if compile_idx - 3 >= 0:
                batch_size = parts[compile_idx - 3]
            if compile_idx - 4 >= 0:
                size = parts[compile_idx - 4]
            model_tokens = parts[: max(0, compile_idx - 4)]
            if model_tokens:
                model_type = "_".join(model_tokens)
            elif compile_idx - 4 == 0 and parts:
                model_type = parts[0]
            dataset_tokens = parts[compile_idx + 2 :]
        else:
            # Fallback to simple positional parsing when the expected pattern is missing
            model_type = parts[0]
            if len(parts) >= 2:
                size = parts[1]
            if len(parts) >= 3:
                batch_size = parts[2]
            if len(parts) >= 4:
                block_size = parts[3]
            if len(parts) >= 5:
                precision = parts[4]
            if len(parts) >= 6:
                compile_mode = parts[5]
            if len(parts) >= 7:
                optimizer = parts[6]
            if len(parts) >= 8:
                dataset_tokens = parts[7:]

    dataset = "_".join(dataset_tokens) if dataset_tokens else "unknown"
    return RunMetadata(
        model_type=model_type,
        size=size,
        batch_size=batch_size,
        block_size=block_size,
        precision=precision,
        compile_mode=compile_mode,
        optimizer=optimizer,
        dataset=dataset,
        date=date,
        time=time,
        counter=counter,
        filename=path.name,
    )


def parse_row(row: List[str]) -> Dict[str, Optional[str]]:
    if not row:
        return {}
    row = [entry.strip() for entry in row]
    if len(row) >= len(ROW_MAP_VERBOSE):
        keys = ROW_MAP_VERBOSE
    elif len(row) == len(ROW_MAP_COMPACT):
        keys = ROW_MAP_COMPACT
    else:
        keys = ROW_MAP_COMPACT[: len(row)]
    return dict(zip(keys, row))


def modified_z_outliers(values: List[float], threshold: float) -> List[int]:
    if len(values) < 3:
        return []
    med = median(values)
    deviations = [abs(v - med) for v in values]
    mad = median(deviations)
    if mad == 0:
        return []
    flagged = []
    for idx, value in enumerate(values):
        z_score = 0.6745 * (value - med) / mad
        if abs(z_score) > threshold:
            flagged.append(idx)
    return flagged


def analyse_file(path: Path, milestones: Iterable[float]) -> List[MilestoneRecord]:
    milestones_sorted = sorted(milestones)
    hits: Dict[float, MilestoneRecord] = {}
    meta = parse_filename(path)
    try:
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            try:
                next(reader)
            except StopIteration:
                return []
            milestone_index = 0
            milestones_list = milestones_sorted
            for row in reader:
                parsed = parse_row(row)
                if not parsed:
                    continue
                tokens = to_float(parsed.get("total_tokens"))
                if tokens is None:
                    continue
                # Advance milestones as we reach them
                while milestone_index < len(milestones_list) and tokens >= milestones_list[milestone_index]:
                    milestone_value = milestones_list[milestone_index]
                    if milestone_value not in hits:
                        record = MilestoneRecord(
                            milestone=milestone_value,
                            tokens=tokens,
                            step=to_int(parsed.get("step")),
                            train_loss=to_float(parsed.get("train_loss")),
                            val_loss=to_float(parsed.get("val_loss")),
                            val_perplexity=to_float(parsed.get("val_perplexity")),
                            learning_rate=to_float(parsed.get("learning_rate")),
                            grad_norm=to_float(parsed.get("grad_norm")),
                            timestamp=parsed.get("timestamp") or "",
                            path=path,
                            meta=meta,
                        )
                        hits[milestone_value] = record
                    milestone_index += 1
                if milestone_index >= len(milestones_list):
                    break
    except OSError as exc:
        print(f"Could not read {path}: {exc}")
        return []
    return list(hits.values())


def format_tokens(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.2f}K"
    return f"{value:.0f}"


def format_loss(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "--"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def sort_key(record: MilestoneRecord, sort_by: str) -> float:
    metric = getattr(record, sort_by)
    if metric is None or math.isnan(metric):
        return math.inf
    return metric


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default="out/metrics_logs",
        type=Path,
        help="Directory containing metrics CSV files",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern to select metrics files",
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=float,
        default=[5e7, 1e8, 2e8, 5e8, 1e9],
        help="Token milestones to evaluate (supports scientific notation)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of runs to display per milestone",
    )
    parser.add_argument(
        "--sort-by",
        choices=["val_loss", "train_loss"],
        default="val_loss",
        help="Primary metric for ranking",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=3.5,
        help="Modified z-score threshold used to drop loss outliers",
    )
    parser.add_argument(
        "--include-train-only",
        action="store_true",
        help="Include runs without validation loss in the ranking",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    log_dir: Path = args.log_dir
    if not log_dir.exists():
        raise SystemExit(f"Log directory not found: {log_dir}")

    files = sorted(log_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No CSV files found in {log_dir} matching {args.pattern}")

    milestones = sorted(args.milestones)
    results: Dict[float, List[MilestoneRecord]] = {milestone: [] for milestone in milestones}
    for file_path in files:
        records = analyse_file(file_path, milestones)
        for record in records:
            results[record.milestone].append(record)

    for milestone in milestones:
        milestone_records = results[milestone]
        header = f"\n=== {format_tokens(milestone)} tokens ==="
        print(header)
        if not milestone_records:
            print("No runs reached this milestone.")
            continue

        val_losses = [rec.val_loss for rec in milestone_records if rec.val_loss is not None]
        train_losses = [rec.train_loss for rec in milestone_records if rec.train_loss is not None]

        outlier_indices: set[int] = set()
        if val_losses:
            idx_map = {i: rec.val_loss for i, rec in enumerate(milestone_records) if rec.val_loss is not None}
            indices = list(idx_map.keys())
            values = [idx_map[i] for i in indices]
            for pos in modified_z_outliers(values, args.outlier_threshold):
                outlier_indices.add(indices[pos])
        if train_losses:
            idx_map = {i: rec.train_loss for i, rec in enumerate(milestone_records) if rec.train_loss is not None}
            indices = list(idx_map.keys())
            values = [idx_map[i] for i in indices]
            for pos in modified_z_outliers(values, args.outlier_threshold):
                outlier_indices.add(indices[pos])

        filtered_records = [
            rec
            for i, rec in enumerate(milestone_records)
            if i not in outlier_indices and (args.include_train_only or rec.val_loss is not None)
        ]

        if not filtered_records:
            print("All runs filtered out (either outliers or missing validation loss).")
            continue

        filtered_records.sort(key=lambda rec: (sort_key(rec, args.sort_by), sort_key(rec, "train_loss"), rec.tokens))

        print(
            f"Rank  Model               Config                  Dataset        val_loss  train_loss  Tokens    Step   Time                File"
        )
        print(
            f"----  ------------------  ----------------------  ------------  --------  ----------  --------  -----  -------------------  ----"
        )
        for rank, record in enumerate(filtered_records[: args.top_k], start=1):
            model_label = record.meta.model_label[:18]
            config_label = record.meta.config_label[:22]
            dataset_label = record.meta.dataset[:12]
            val_field = format_loss(record.val_loss)
            train_field = format_loss(record.train_loss)
            token_field = format_tokens(record.tokens)
            step_field = f"{record.step}" if record.step is not None else "--"
            time_field = record.timestamp[:19]
            print(
                f"{rank:>4}  {model_label:<18}  {config_label:<22}  {dataset_label:<12}  "
                f"{val_field:>8}  {train_field:>10}  {token_field:>8}  {step_field:>5}  {time_field:<19}  {record.meta.filename}"
            )

        missing_val_count = sum(1 for rec in milestone_records if rec.val_loss is None)
        notes: List[str] = []
        if outlier_indices:
            label = "outlier" if len(outlier_indices) == 1 else "outliers"
            notes.append(f"{len(outlier_indices)} potential {label}")
        if missing_val_count and not args.include_train_only:
            run_label = "run" if missing_val_count == 1 else "runs"
            notes.append(
                f"{missing_val_count} {run_label} without validation loss (use --include-train-only to show)"
            )
        if notes:
            print("Excluded: " + "; ".join(notes) + ".")


if __name__ == "__main__":
    main()
