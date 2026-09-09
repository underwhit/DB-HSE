#!/usr/bin/env python3
"""Compare one target JSONL against one or more paired comparator JSONLs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REFUSAL_FIELDS = (
    "manual_is_refusal",
    "is_refusal",
    "llm_is_refusal",
    "auto_is_refusal",
    "refusal",
)
PROMPT_FIELDS = ("prompt", "question", "instruction", "user_prompt")
PromptKey = Tuple[str, int]


def first_present(row: dict, fields: Iterable[str]) -> Tuple[str, object]:
    for field in fields:
        if field in row:
            return field, row[field]
    raise KeyError(f"none of {list(fields)} found")


def parse_bool(value: object, *, path: Path, line_no: int, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "refusal", "refused"}:
            return True
        if text in {"false", "no", "n", "0", "non-refusal", "non_refusal"}:
            return False
    raise ValueError(f"{path.name}:{line_no}: cannot parse {field}={value!r} as bool")


def load_jsonl(path: Path) -> Dict[PromptKey, bool]:
    data: Dict[PromptKey, bool] = {}
    seen: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            _, prompt_value = first_present(row, PROMPT_FIELDS)
            prompt = str(prompt_value)
            occurrence = seen.get(prompt, 0)
            seen[prompt] = occurrence + 1
            field, value = first_present(row, REFUSAL_FIELDS)
            data[(prompt, occurrence)] = parse_bool(
                value, path=path, line_no=line_no, field=field
            )
    if not data:
        raise RuntimeError(f"{path.name}: no rows found")
    return data


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(
    diffs: Sequence[int], n_bootstrap: int, seed: int, alpha: float
) -> Tuple[float, float]:
    n = len(diffs)
    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        values = np.asarray(diffs, dtype=float)
        means = np.empty(n_bootstrap, dtype=float)
        batch_size = min(1000, n_bootstrap)
        for start in range(0, n_bootstrap, batch_size):
            end = min(start + batch_size, n_bootstrap)
            indices = rng.integers(0, n, size=(end - start, n))
            means[start:end] = values[indices].mean(axis=1) * 100.0
        low, high = np.percentile(
            means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)]
        )
        return float(low), float(high)
    except ImportError:
        rng = random.Random(seed)
        means = [
            sum(diffs[rng.randrange(n)] for _ in range(n)) / n * 100.0
            for _ in range(n_bootstrap)
        ]
        return percentile(means, alpha / 2.0), percentile(means, 1.0 - alpha / 2.0)


def exact_mcnemar_p(target_only: int, comparator_only: int) -> float:
    n = target_only + comparator_only
    if n == 0:
        return 1.0
    k = min(target_only, comparator_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def parse_comparator(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use NAME=PATH, for example Base=results/base.jsonl")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("both NAME and PATH are required")
    return name.strip(), Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Display label for the model")
    parser.add_argument("--target-name", default="DB-HSE")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--comparator",
        action="append",
        type=parse_comparator,
        required=True,
        help="Repeatable NAME=PATH comparator specification",
    )
    parser.add_argument("--n-bootstrap", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    target = load_jsonl(args.target)
    rows: List[dict] = []
    for offset, (name, path) in enumerate(args.comparator):
        comparator = load_jsonl(path)
        if set(target) != set(comparator):
            missing = len(set(target) - set(comparator))
            extra = len(set(comparator) - set(target))
            raise RuntimeError(
                f"{path.name}: prompt alignment mismatch (missing={missing}, extra={extra})"
            )
        keys = list(target)
        target_values = [int(target[key]) for key in keys]
        comparator_values = [int(comparator[key]) for key in keys]
        diffs = [a - b for a, b in zip(target_values, comparator_values)]
        low, high = bootstrap_ci(
            diffs, args.n_bootstrap, args.seed + offset, args.alpha
        )
        target_only = sum(a == 1 and b == 0 for a, b in zip(target_values, comparator_values))
        comparator_only = sum(a == 0 and b == 1 for a, b in zip(target_values, comparator_values))
        rows.append(
            {
                "model": args.model,
                "target": args.target_name,
                "comparator": name,
                "n": len(keys),
                "target_rr": sum(target_values) / len(keys) * 100.0,
                "comparator_rr": sum(comparator_values) / len(keys) * 100.0,
                "delta_pp": sum(diffs) / len(diffs) * 100.0,
                "ci_low_pp": low,
                "ci_high_pp": high,
                "target_only": target_only,
                "comparator_only": comparator_only,
                "mcnemar_p": exact_mcnemar_p(target_only, comparator_only),
            }
        )

    print("| Model | Comparator | n | Comparator RR | Target RR | Delta [95% CI], pp | McNemar p |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['comparator']} | {row['n']} | "
            f"{row['comparator_rr']:.2f} | {row['target_rr']:.2f} | "
            f"{row['delta_pp']:+.2f} [{row['ci_low_pp']:+.2f}, {row['ci_high_pp']:+.2f}] | "
            f"{row['mcnemar_p']:.4g} |"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
