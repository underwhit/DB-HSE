#!/usr/bin/env python3
"""Create a fixed XSTest calibration/holdout split.

The default split samples 50 safe and 50 unsafe prompts for calibration,
stratified by XSTest `label` and `type`, and leaves the remaining prompts as a
held-out test set. This is intended for gate calibration only, not for selecting
DB-HSE layer/scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_xstest_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"id", "prompt", "type", "label"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    for idx, row in enumerate(rows):
        row["source_index"] = idx
        row["label"] = row["label"].strip().lower()
        row["type"] = row["type"].strip()
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def allocate_quotas(groups: Dict[str, List[dict]], total: int, rng: random.Random) -> Dict[str, int]:
    available = sum(len(items) for items in groups.values())
    if total > available:
        raise ValueError(f"requested {total} rows, but only {available} available")
    raw: List[Tuple[str, float, int]] = []
    for group_name, items in groups.items():
        exact = total * len(items) / available
        raw.append((group_name, exact, int(exact)))

    quotas = {group_name: base for group_name, _, base in raw}
    remaining = total - sum(quotas.values())

    # Largest remainder allocation. Break exact ties with a seeded random score.
    tie_scores = {group_name: rng.random() for group_name in groups}
    ranked = sorted(
        raw,
        key=lambda item: (item[1] - item[2], tie_scores[item[0]]),
        reverse=True,
    )
    for group_name, _, _ in ranked[:remaining]:
        quotas[group_name] += 1

    for group_name, quota in quotas.items():
        if quota > len(groups[group_name]):
            raise RuntimeError(f"quota {quota} exceeds group size {len(groups[group_name])} for {group_name}")
    return quotas


def stratified_sample(
    rows: List[dict],
    *,
    label: str,
    n: int,
    rng: random.Random,
) -> Tuple[List[dict], Dict[str, int]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if row["label"] == label:
            groups[row["type"]].append(row)
    if not groups:
        raise RuntimeError(f"no rows for label={label}")

    quotas = allocate_quotas(groups, n, rng)
    selected: List[dict] = []
    for group_name, items in groups.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        selected.extend(shuffled[: quotas[group_name]])

    selected.sort(key=lambda row: int(row["source_index"]))
    return selected, quotas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("xstest_prompts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/xstest_seed13"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--safe-size", type=int, default=50)
    parser.add_argument("--unsafe-size", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output dir is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows = read_xstest_csv(args.input)
    safe_calib, safe_quotas = stratified_sample(rows, label="safe", n=args.safe_size, rng=rng)
    unsafe_calib, unsafe_quotas = stratified_sample(rows, label="unsafe", n=args.unsafe_size, rng=rng)

    calib_ids = {int(row["source_index"]) for row in safe_calib + unsafe_calib}
    calibration_rows: List[dict] = []
    holdout_rows: List[dict] = []
    for row in rows:
        obj = dict(row)
        obj["split_seed"] = args.seed
        if int(row["source_index"]) in calib_ids:
            obj["split"] = "calibration"
            calibration_rows.append(obj)
        else:
            obj["split"] = "holdout"
            holdout_rows.append(obj)

    calibration_rows.sort(key=lambda row: int(row["source_index"]))
    holdout_rows.sort(key=lambda row: int(row["source_index"]))

    total_calib = args.safe_size + args.unsafe_size
    cal_path = args.output_dir / f"xstest_calibration_{total_calib}.jsonl"
    safe_cal_path = args.output_dir / f"xstest_calibration_safe_{args.safe_size}.jsonl"
    unsafe_cal_path = args.output_dir / f"xstest_calibration_unsafe_{args.unsafe_size}.jsonl"
    hold_path = args.output_dir / f"xstest_holdout_{len(holdout_rows)}.jsonl"
    manifest_path = args.output_dir / "split_manifest.json"
    readme_path = args.output_dir / "split_summary.md"

    write_jsonl(cal_path, calibration_rows)
    write_jsonl(safe_cal_path, [row for row in calibration_rows if row["label"] == "safe"])
    write_jsonl(unsafe_cal_path, [row for row in calibration_rows if row["label"] == "unsafe"])
    write_jsonl(hold_path, holdout_rows)

    manifest = {
        "input": args.input.name,
        "seed": args.seed,
        "safe_size": args.safe_size,
        "unsafe_size": args.unsafe_size,
        "calibration_path": cal_path.name,
        "safe_calibration_path": safe_cal_path.name,
        "unsafe_calibration_path": unsafe_cal_path.name,
        "holdout_path": hold_path.name,
        "calibration_counts_by_label": dict(Counter(row["label"] for row in calibration_rows)),
        "holdout_counts_by_label": dict(Counter(row["label"] for row in holdout_rows)),
        "safe_type_quotas": safe_quotas,
        "unsafe_type_quotas": unsafe_quotas,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# XSTest Calibration Split",
        "",
        f"- Input: `{args.input.name}`",
        f"- Seed: `{args.seed}`",
        f"- Calibration: `{cal_path.name}`",
        f"- Safe calibration only: `{safe_cal_path.name}`",
        f"- Unsafe calibration only: `{unsafe_cal_path.name}`",
        f"- Holdout: `{hold_path.name}`",
        "",
        "## Counts",
        "",
        "| Split | Safe | Unsafe | Total |",
        "|---|---:|---:|---:|",
    ]
    cal_counts = Counter(row["label"] for row in calibration_rows)
    hold_counts = Counter(row["label"] for row in holdout_rows)
    lines.append(f"| Calibration | {cal_counts['safe']} | {cal_counts['unsafe']} | {len(calibration_rows)} |")
    lines.append(f"| Holdout | {hold_counts['safe']} | {hold_counts['unsafe']} | {len(holdout_rows)} |")
    lines.extend(["", "## Calibration Type Quotas", "", "| Label | Type | Count |", "|---|---|---:|"])
    for label, quotas in (("safe", safe_quotas), ("unsafe", unsafe_quotas)):
        for type_name, count in sorted(quotas.items()):
            lines.append(f"| {label} | {type_name} | {count} |")
    lines.append("")
    readme_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] wrote {cal_path} ({len(calibration_rows)} rows)")
    print(f"[OK] wrote {safe_cal_path} ({args.safe_size} rows)")
    print(f"[OK] wrote {unsafe_cal_path} ({args.unsafe_size} rows)")
    print(f"[OK] wrote {hold_path} ({len(holdout_rows)} rows)")
    print(f"[OK] wrote {manifest_path}")
    print(f"[OK] wrote {readme_path}")


if __name__ == "__main__":
    main()
