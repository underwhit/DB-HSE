import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_METHODS = ["base", "prompt_well", "actadd", "isp", "dbhse"]
DEFAULT_MODELS = ["Qwen2.5-7B", "Qwen3-8B", "Llama3.1-8B"]

METHOD_LABELS = {
    "base": "Base",
    "prompt_well": "Prompt-Well",
    "actadd": "ActAdd",
    "isp": "ISP",
    "dbhse": "DB-HSE",
}


def iter_jsonl_files(inputs: Sequence[Path]) -> Iterable[Path]:
    for path in inputs:
        if path.is_dir():
            for file in sorted(path.glob("*.jsonl")):
                if file.name == "analysis.jsonl":
                    continue
                yield file
        elif path.is_file() and path.suffix == ".jsonl":
            if path.name != "analysis.jsonl":
                yield path
        else:
            raise RuntimeError(f"Input is not a JSONL file or directory: {path}")


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        for row_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield row_no, json.loads(line)


def infer_method(path: Path) -> Optional[str]:
    text = "/".join(part.lower() for part in path.parts)
    name = path.stem.lower()

    if "actadd" in text:
        return "actadd"
    if "isp" in text:
        return "isp"
    if "dbhsd" in text or "dbhse" in text or "缩放因子" in text:
        return "dbhse"
    if "well_specified" in text or "prompt_well" in text or "prompt-well" in text:
        return "prompt_well"
    if name.endswith("_prompt") or name.endswith("_prompt_safe") or name.endswith("_prompt_well"):
        return "prompt_well"
    if "base" in text:
        return "base"
    return None


def infer_model(path: Path) -> str:
    text = "/".join(part.lower() for part in path.parts)
    if "qwen25" in text or "qwen2.5" in text:
        return "Qwen2.5-7B"
    if "qwen38" in text or "qwen3" in text:
        return "Qwen3-8B"
    if "llama3" in text or "llama-3" in text or "llama" in text:
        return "Llama3.1-8B"
    return "unknown"


def refusal_bucket(row: Dict) -> Optional[str]:
    if "is_refusal" not in row:
        return None
    return "refusal" if bool(row.get("is_refusal")) else "non_refusal"


def collect_records(inputs: Sequence[Path], methods: Sequence[str]) -> Dict[Tuple[str, str], List[Dict]]:
    allowed = set(methods)
    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    skipped = Counter()

    for file in iter_jsonl_files(inputs):
        method = infer_method(file)
        if method is None:
            skipped["unknown_method_file"] += 1
            continue
        if method not in allowed:
            skipped[f"method_not_requested:{method}"] += 1
            continue

        model = infer_model(file)
        for row_no, row in read_jsonl(file):
            bucket = refusal_bucket(row)
            if bucket is None:
                skipped["missing_is_refusal"] += 1
                continue
            prompt = row.get("prompt")
            response = row.get("model_response")
            if prompt is None or response is None:
                skipped["missing_prompt_or_response"] += 1
                continue

            grouped[(method, bucket)].append(
                {
                    "model": model,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "prompt": prompt,
                    "response": response,
                    "llm_is_refusal": bool(row.get("is_refusal")),
                    "judge_source": row.get("judge_source"),
                    "judge_note": row.get("judge_note"),
                    "source_file": file.name,
                    "source_row": row_no,
                    "source_index": row.get("index", row.get("source_index", row_no)),
                }
            )

    if skipped:
        print("[Skip]", dict(skipped))
    return grouped


def sample_records(
    grouped: Dict[Tuple[str, str], List[Dict]],
    methods: Sequence[str],
    refusal_n: int,
    non_refusal_n: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    sampled: List[Dict] = []

    for method in methods:
        for bucket, want in (("refusal", refusal_n), ("non_refusal", non_refusal_n)):
            pool = list(grouped.get((method, bucket), []))
            rng.shuffle(pool)
            take = min(want, len(pool))
            if take < want:
                print(
                    f"[Warn] {METHOD_LABELS.get(method, method)} {bucket}: "
                    f"requested={want}, available={len(pool)}, taking={take}"
                )
            for row in pool[:take]:
                sampled.append(dict(row, sample_bucket=bucket))

    return add_manual_fields(sampled)


def allocate_quotas(total: int, labels: Sequence[str], rng: random.Random) -> Dict[str, int]:
    labels = list(labels)
    if not labels:
        return {}
    shuffled = list(labels)
    rng.shuffle(shuffled)
    base = total // len(shuffled)
    remainder = total % len(shuffled)
    return {label: base + (1 if i < remainder else 0) for i, label in enumerate(shuffled)}


def sample_records_stratified_by_model(
    grouped: Dict[Tuple[str, str], List[Dict]],
    methods: Sequence[str],
    models: Sequence[str],
    refusal_n: int,
    non_refusal_n: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    sampled: List[Dict] = []

    for method in methods:
        for bucket, want in (("refusal", refusal_n), ("non_refusal", non_refusal_n)):
            pool = list(grouped.get((method, bucket), []))
            by_model: Dict[str, List[Dict]] = defaultdict(list)
            for row in pool:
                by_model[row["model"]].append(row)

            selected: List[Dict] = []
            selected_keys = set()
            quotas = allocate_quotas(want, models, rng)

            for model in models:
                model_pool = list(by_model.get(model, []))
                rng.shuffle(model_pool)
                quota = quotas.get(model, 0)
                take = min(quota, len(model_pool))
                if take < quota:
                    print(
                        f"[Warn] {METHOD_LABELS.get(method, method)} {bucket} / {model}: "
                        f"requested={quota}, available={len(model_pool)}, taking={take}"
                    )
                for row in model_pool[:take]:
                    key = (row["source_file"], row["source_row"])
                    selected.append(row)
                    selected_keys.add(key)

            if len(selected) < want:
                remaining = [
                    row
                    for row in pool
                    if (row["source_file"], row["source_row"]) not in selected_keys
                ]
                rng.shuffle(remaining)
                selected.extend(remaining[: want - len(selected)])

            if len(selected) < want:
                print(
                    f"[Warn] {METHOD_LABELS.get(method, method)} {bucket}: "
                    f"requested={want}, available={len(pool)}, taking={len(selected)}"
                )

            for row in selected[:want]:
                sampled.append(dict(row, sample_bucket=bucket))

    return add_manual_fields(sampled)


def add_manual_fields(rows: Sequence[Dict]) -> List[Dict]:
    output = []
    for i, row in enumerate(rows):
        new_row = dict(row)
        new_row["validation_id"] = f"rrval_{i:04d}"
        new_row["manual_is_refusal"] = ""
        new_row["manual_note"] = ""
        output.append(new_row)
    return output


def write_jsonl(rows: Sequence[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(rows: Sequence[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# RR Human Validation Sample\n\n")
        f.write("Manual label: set `manual_is_refusal` to `true`, `false`, or `unclear`.\n\n")
        for row in rows:
            response = " ".join(str(row.get("response", "")).split())
            if len(response) > 1200:
                response = response[:1197] + "..."
            f.write(
                f"## {row['validation_id']} | {row['model']} | {row['method_label']} "
                f"| LLM={row['llm_is_refusal']}\n\n"
            )
            f.write(f"Source: `{row['source_file']}:{row['source_row']}`\n\n")
            f.write(f"Prompt: {row['prompt']}\n\n")
            f.write("Manual refusal? `[ ] true  [ ] false  [ ] unclear`\n\n")
            f.write(f"> {response}\n\n")
            f.write("---\n\n")


def print_summary(rows: Sequence[Dict]) -> None:
    print(f"[OK] sampled {len(rows)} rows")
    by_method = Counter(row["method_label"] for row in rows)
    by_bucket = Counter((row["method_label"], row["sample_bucket"]) for row in rows)
    by_model = Counter(row["model"] for row in rows)
    by_method_model = Counter((row["method_label"], row["model"]) for row in rows)
    print("[By method]", dict(by_method))
    print("[By model]", dict(by_model))
    print("[By method/model]")
    for key, val in sorted(by_method_model.items()):
        print(f"  {key[0]} / {key[1]}: {val}")
    for key, val in sorted(by_bucket.items()):
        print(f"  {key[0]} / {key[1]}: {val}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a human validation set for the main RR LLM-judge labels. "
            "Each method is sampled with refusal and non-refusal examples."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more judged JSONL files or directories containing judged JSONL files.",
    )
    parser.add_argument(
        "--output",
        default="analysis/rr_human_validation_sample.jsonl",
        help="Output JSONL annotation file.",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional Markdown review sheet.",
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--stratify-model",
        action="store_true",
        help=(
            "Within each method and refusal bucket, distribute samples across "
            "models as evenly as possible."
        ),
    )
    parser.add_argument("--refusal-n", type=int, default=10)
    parser.add_argument("--non-refusal-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    methods = [m.strip().lower() for m in args.methods]
    unknown = [m for m in methods if m not in METHOD_LABELS]
    if unknown:
        raise RuntimeError(f"Unknown methods: {unknown}; allowed={sorted(METHOD_LABELS)}")

    grouped = collect_records([Path(p) for p in args.inputs], methods)
    if args.stratify_model:
        rows = sample_records_stratified_by_model(
            grouped,
            methods,
            args.models,
            args.refusal_n,
            args.non_refusal_n,
            args.seed,
        )
    else:
        rows = sample_records(grouped, methods, args.refusal_n, args.non_refusal_n, args.seed)
    out_path = Path(args.output)
    write_jsonl(rows, out_path)
    if args.markdown_output:
        write_markdown(rows, Path(args.markdown_output))

    print_summary(rows)
    print(f"JSONL: {out_path}")
    if args.markdown_output:
        print(f"MD   : {args.markdown_output}")


if __name__ == "__main__":
    main()
