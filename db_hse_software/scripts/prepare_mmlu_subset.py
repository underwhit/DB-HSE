#!/usr/bin/env python3
"""Create a deterministic MMLU JSONL subset compatible with generation scripts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path).to_dict(orient="records")
        except ImportError as exc:
            raise RuntimeError(
                "Reading MMLU parquet requires pyarrow. Install the updated "
                "requirements_qwen3_14b.txt or run: pip install pyarrow"
            ) from exc
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise RuntimeError(f"{path}: expected a top-level list")
        return data
    raise RuntimeError(f"Unsupported MMLU input format: {path.suffix}")


def normalize_choices(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        choices = list(value)
    elif hasattr(value, "tolist"):
        choices = list(value.tolist())
    else:
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Cannot parse choices: {value!r}") from exc
        choices = list(parsed)
    if len(choices) != 4:
        raise RuntimeError(f"Expected four choices, got {len(choices)}: {choices!r}")
    return [str(choice) for choice in choices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="MMLU validation parquet")
    parser.add_argument("--output", type=Path, default=Path("data/mmlu_seed13_300.jsonl"))
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise RuntimeError(f"Output exists: {args.output}; pass --overwrite")
    records = load_records(args.input)
    if args.size > len(records):
        raise RuntimeError(f"Requested {args.size} rows from a dataset of size {len(records)}")

    indices = list(range(len(records)))
    random.Random(args.seed).shuffle(indices)
    indices = sorted(indices[: args.size])
    letters = "ABCD"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for source_index in indices:
            row = records[source_index]
            choices = normalize_choices(row["choices"])
            prompt = (
                f"Question: {row['question']}\nOptions:\n"
                f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\n"
                "Output strictly and only the single letter of the correct option (A, B, C, or D)."
            )
            raw_answer = row["answer"]
            if isinstance(raw_answer, str) and raw_answer.strip().upper() in letters:
                ground_truth = raw_answer.strip().upper()
            else:
                ground_truth = letters[int(raw_answer)]
            item = {
                "id": source_index,
                "source_index": source_index,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "split": "mmlu_utility_test",
                "split_seed": args.seed,
            }
            if "subject" in row:
                item["subject"] = str(row["subject"])
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(indices)} rows to {args.output}")


if __name__ == "__main__":
    main()
