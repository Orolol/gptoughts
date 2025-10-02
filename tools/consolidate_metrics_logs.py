#!/usr/bin/env python3
"""Consolidate Lightning CSV metric logs.

This script scans metrics log directories (default: ouputs/*/metrics_logs)
and merges multiple CSV chunks that share the same run prefix into a single
chronological file. When a run resumes, the first recorded step duplicates
the last step from the previous chunk; duplicate or decreasing steps are
skipped during consolidation. Empty CSV files (header-only) are deleted.

Usage:
    python scripts/consolidate_metrics_logs.py [root_dir]

If root_dir is omitted, 'ouputs' is used by default.
"""

from __future__ import annotations

import csv
import itertools
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

TIMESTAMP_PATTERN = re.compile(r"^(?P<base>.+)_\d{8}_\d{6}(?:_\d+)?\.csv$")


def find_metrics_dirs(root: Path) -> List[Path]:
    """Return all metrics_logs directories under the given root."""
    metrics_dirs = []
    if not root.exists():
        return metrics_dirs
    for path in root.rglob("metrics_logs"):
        if path.is_dir():
            metrics_dirs.append(path)
    return metrics_dirs


def cleanup_empty_file(path: Path, header: List[str], rows: List[List[str]]) -> bool:
    """Delete the file if it only contains the header (no data rows)."""
    if not rows:
        try:
            path.unlink()
            print(f"Deleted empty log: {path}")
        except OSError as exc:
            print(f"Warning: failed to delete empty log {path}: {exc}")
        return True
    return False


def consolidate_group(base: str, files: List[Path]) -> None:
    """Merge CSV files belonging to the same run prefix."""
    files = sorted(files)
    header: List[str] = []
    combined_rows: List[List[str]] = []
    seen_steps = set()
    last_step = -1
    total_idx: int | None = None
    offset = 0.0
    last_raw_total: float | None = None
    last_total_adjusted = 0.0

    for file_path in files:
        with file_path.open('r', newline='') as handle:
            reader = csv.reader(handle)
            file_header = next(reader, None)
            rows = list(reader)

        if file_header is None:
            # Completely empty file
            cleanup_empty_file(file_path, [], [])
            continue

        if not header:
            header = file_header
            if 'total_tokens' in header:
                total_idx = header.index('total_tokens')
        elif file_header != header:
            print(f"Warning: header mismatch in {file_path}; using header from first file.")

        if cleanup_empty_file(file_path, header, rows):
            continue

        for row in rows:
            if not row:
                continue
            step_str = row[0]
            try:
                step = int(step_str)
            except (ValueError, TypeError):
                combined_rows.append(row)
                continue
            if step in seen_steps:
                continue
            if step < last_step:
                # Skip strictly decreasing steps (likely duplicate chunk)
                continue
            seen_steps.add(step)
            last_step = step

            if total_idx is not None and total_idx < len(row):
                value_str = row[total_idx]
                try:
                    raw_total = float(value_str)
                except (TypeError, ValueError):
                    raw_total = None
                if raw_total is not None:
                    if last_raw_total is not None and raw_total < last_raw_total:
                        offset += last_total_adjusted
                    adjusted_total = raw_total + offset
                    value_lower = value_str.lower() if isinstance(value_str, str) else ''
                    if raw_total.is_integer() and '.' not in value_lower and 'e' not in value_lower:
                        formatted_total = str(int(round(adjusted_total)))
                    else:
                        formatted_total = ("{:.6f}".format(adjusted_total)).rstrip('0').rstrip('.')
                        if formatted_total == "":
                            formatted_total = "0"
                    row[total_idx] = formatted_total
                    last_raw_total = raw_total
                    last_total_adjusted = adjusted_total

            combined_rows.append(row)

    if not combined_rows:
        return

    output_path = files[0].with_name(f"{base}_consolidated.csv")
    try:
        with output_path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(combined_rows)
        print(f"Wrote consolidated log: {output_path} ({len(combined_rows)} rows)")
    except OSError as exc:
        print(f"Error writing consolidated log {output_path}: {exc}")
        return

    # Optionally remove original chunks (except consolidated file)
    for file_path in files:
        if file_path == output_path:
            continue
        try:
            file_path.unlink()
            print(f"Removed chunk log: {file_path}")
        except OSError as exc:
            print(f"Warning: failed to remove chunk log {file_path}: {exc}")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('ouputs')
    metrics_dirs = find_metrics_dirs(root)
    if not metrics_dirs:
        print(f"No metrics_logs directories found under {root.resolve()}")
        return

    for metrics_dir in metrics_dirs:
        csv_files = [p for p in metrics_dir.glob('*.csv') if p.is_file()]
        groups: Dict[str, List[Path]] = defaultdict(list)
        for csv_path in csv_files:
            match = TIMESTAMP_PATTERN.match(csv_path.name)
            if not match:
                continue
            groups[match.group('base')].append(csv_path)

        if not groups:
            continue

        print(f"Processing {metrics_dir}...")
        for base, files in groups.items():
            consolidate_group(base, files)


if __name__ == '__main__':
    main()
