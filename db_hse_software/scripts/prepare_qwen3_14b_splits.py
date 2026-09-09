#!/usr/bin/env python3
"""Create disjoint calibration and held-out subsets for Qwen3-14B."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "prompt" not in row:
                    raise RuntimeError(f"{path}:{line_no} missing prompt")
                rows.append(row)
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"{path}: expected a top-level list")
    for idx, row in enumerate(data):
        if not isinstance(row, dict) or "prompt" not in row:
            raise RuntimeError(f"{path}:{idx} missing prompt")
    return data


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    low = str(value).strip().lower()
    if low in {"1", "true", "yes", "y", "harmful", "unsafe"}:
        return True
    if low in {"0", "false", "no", "n", "harmless", "safe"}:
        return False
    raise ValueError(f"Cannot interpret is_harmful value: {value!r}")


def annotate(row: Dict[str, Any], source_index: int, split: str, seed: int) -> Dict[str, Any]:
    item = dict(row)
    harmful = as_bool(item["is_harmful"])
    item.update(
        {
            "source_index": source_index,
            "split": split,
            "split_seed": seed,
            "label": "unsafe" if harmful else "safe",
            "should_refuse": harmful,
        }
    )
    return item


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose(pool: List[int], n: int, rng: random.Random, name: str) -> List[int]:
    if n > len(pool):
        raise RuntimeError(f"Requested {n} {name} rows, but only {len(pool)} are available")
    values = list(pool)
    rng.shuffle(values)
    return values[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/qwen3_14b_seed13"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--calibration-harmful", type=int, default=80)
    parser.add_argument("--calibration-safe", type=int, default=20)
    parser.add_argument("--test-harmful", type=int, default=289)
    parser.add_argument("--test-safe", type=int, default=11)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.input)
    harmful = [idx for idx, row in enumerate(rows) if as_bool(row.get("is_harmful"))]
    safe = [idx for idx, row in enumerate(rows) if not as_bool(row.get("is_harmful"))]
    rng = random.Random(args.seed)

    calibration_harmful = choose(harmful, args.calibration_harmful, rng, "harmful calibration")
    remaining_harmful = [idx for idx in harmful if idx not in set(calibration_harmful)]
    calibration_safe = choose(safe, args.calibration_safe, rng, "safe calibration")
    remaining_safe = [idx for idx in safe if idx not in set(calibration_safe)]

    test_harmful = choose(remaining_harmful, args.test_harmful, rng, "harmful test")
    test_safe = choose(remaining_safe, args.test_safe, rng, "safe test")

    calibration_idx = sorted(calibration_harmful + calibration_safe)
    test_idx = sorted(test_harmful + test_safe)
    if set(calibration_idx) & set(test_idx):
        raise RuntimeError("Calibration and test indices overlap")

    calibration = [annotate(rows[idx], idx, "calibration", args.seed) for idx in calibration_idx]
    test = [annotate(rows[idx], idx, "heldout_test", args.seed) for idx in test_idx]
    calibration_unsafe = [row for row in calibration if row["should_refuse"]]
    calibration_safe_rows = [row for row in calibration if not row["should_refuse"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "calibration_100.jsonl": calibration,
        "calibration_unsafe.jsonl": calibration_unsafe,
        "calibration_safe.jsonl": calibration_safe_rows,
        "heldout_test_300.jsonl": test,
    }
    for name, selected in outputs.items():
        path = args.output_dir / name
        if path.exists() and not args.overwrite:
            raise RuntimeError(f"Output exists: {path}; pass --overwrite")
        write_jsonl(path, selected)

    summary = {
        "input": args.input.name,
        "seed": args.seed,
        "source_total": len(rows),
        "source_harmful": len(harmful),
        "source_safe": len(safe),
        "calibration_total": len(calibration),
        "calibration_harmful": len(calibration_unsafe),
        "calibration_safe": len(calibration_safe_rows),
        "test_total": len(test),
        "test_harmful": sum(bool(row["should_refuse"]) for row in test),
        "test_safe": sum(not bool(row["should_refuse"]) for row in test),
        "calibration_source_indices": calibration_idx,
        "test_source_indices": test_idx,
    }
    summary_path = args.output_dir / "split_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise RuntimeError(f"Output exists: {summary_path}; pass --overwrite")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("indices")}, indent=2))
    print(f"[OK] wrote split files to {args.output_dir}")


if __name__ == "__main__":
    main()
