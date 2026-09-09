#!/usr/bin/env python3
"""Score one or more generated MMLU JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def extract_answer(text: str) -> str:
    stripped = text.strip().upper()
    leading = re.match(r"^([A-D])(?:\b|[.)])", stripped)
    if leading:
        return leading.group(1)
    match = re.search(r"\b([A-D])\b", stripped)
    return match.group(1) if match else "INVALID"


def summarize(path: Path) -> Dict[str, Any]:
    total = correct = invalid = 0
    method = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prediction = extract_answer(str(row.get("model_response", "")))
            gold = str(row["ground_truth"]).strip().upper()
            total += 1
            correct += int(prediction == gold)
            invalid += int(prediction == "INVALID")
            method = method or row.get("method")
    return {
        "file": path.name,
        "method": method or path.stem,
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "invalid": invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Generated JSONL file or directory")
    parser.add_argument("--out", type=Path, default=Path("analysis/qwen3_14b_mmlu_300_summary.json"))
    args = parser.parse_args()

    files: List[Path]
    if args.input.is_dir():
        files = sorted(args.input.glob("*.jsonl"))
    else:
        files = [args.input]
    if not files:
        raise RuntimeError(f"No JSONL files found at {args.input}")
    rows = [summarize(path) for path in files]
    for row in rows:
        print(
            f"{row['method']}: {row['correct']}/{row['n']} "
            f"accuracy={row['accuracy'] * 100:.2f}% invalid={row['invalid']}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
