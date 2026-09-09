import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SAFE_HINTS = ("safe", "benign", "allowed", "harmless", "good")
UNSAFE_HINTS = ("unsafe", "harmful", "illegal", "dangerous", "disallowed", "bad")

MODEL_NAMES = {
    "qwen25": "Qwen2.5",
    "qwen2.5": "Qwen2.5",
    "qwen3": "Qwen3",
    "qwen38": "Qwen3",
    "llama3": "Llama3.1",
    "llama": "Llama3.1",
}

METHOD_NAMES = {
    "base": "Base",
    "well_specified": "Prompt-Well",
    "prompt": "Prompt-Well",
    "prompt_well": "Prompt-Well",
    "actadd": "ActAdd",
    "isp": "ISP",
    "dbhse_adaptive": "DB-HSE",
    "dbhse": "DB-HSE",
}

METHOD_ORDER = {
    "Base": 0,
    "Prompt-Well": 1,
    "ActAdd": 2,
    "ISP": 3,
    "DB-HSE": 4,
}

MODEL_ORDER = {
    "Qwen2.5": 0,
    "Qwen3": 1,
    "Llama3.1": 2,
}

DATASET_ORDER = {
    "xstest": 0,
    "advbench": 1,
}


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def pct(num: int, den: int, digits: int = 2) -> str:
    if den == 0:
        return "NA"
    return f"{num / den * 100:.{digits}f}%"


def infer_should_refuse(row: Dict) -> Optional[bool]:
    for key in ("should_refuse", "is_unsafe", "is_harmful", "harmful"):
        if key not in row:
            continue
        val = row[key]
        if isinstance(val, bool):
            return val
        low = str(val).strip().lower()
        if low in ("1", "true", "yes", "y"):
            return True
        if low in ("0", "false", "no", "n"):
            return False

    candidates: List[str] = []
    for key in ("label", "safety_label", "type", "category", "prompt_type", "data_type"):
        val = row.get(key)
        if val not in (None, ""):
            candidates.append(str(val).strip().lower())

    for text in candidates:
        if any(hint in text for hint in UNSAFE_HINTS):
            return True
    for text in candidates:
        if any(hint in text for hint in SAFE_HINTS):
            return False
    return None


def infer_model(path: Path) -> str:
    for part in reversed(path.parts):
        low = part.lower()
        if low in MODEL_NAMES:
            return MODEL_NAMES[low]
    return path.parent.name


def infer_dataset(path: Path) -> str:
    for part in reversed(path.parts):
        low = part.lower()
        if low in DATASET_ORDER:
            return low
    return "unknown"


def strip_model_prefix(stem: str) -> str:
    name = stem.lower()
    for prefix in (
        "qwen25_",
        "qwen2.5_",
        "qwen3_",
        "qwen38_",
        "llama3_",
        "llama_",
        "outputs_qwen25_",
        "outputs_qwen2.5_",
        "outputs_qwen3_",
        "outputs_llama3_",
    ):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def infer_method(path: Path) -> str:
    raw = strip_model_prefix(path.stem)
    if raw in METHOD_NAMES:
        return METHOD_NAMES[raw]
    if "well_specified" in raw or "prompt" in raw:
        return "Prompt-Well"
    if "actadd" in raw:
        return "ActAdd"
    if "isp" in raw:
        return "ISP"
    if "dbhse" in raw:
        return "DB-HSE"
    if "base" in raw:
        return "Base"
    return raw


def summarize_file(path: Path) -> Dict:
    rows = list(read_jsonl(path))
    total = len(rows)
    refusal = sum(1 for row in rows if bool(row.get("is_refusal")))

    stats = {
        "dataset": infer_dataset(path),
        "model": infer_model(path),
        "method": infer_method(path),
        "file": path.name,
        "n": total,
        "refusal": refusal,
        "refusal_rate": refusal / total if total else 0.0,
        "safe_total": 0,
        "safe_refusal": 0,
        "unsafe_total": 0,
        "unsafe_refusal": 0,
        "unknown_gold": 0,
    }

    for row in rows:
        should_refuse = infer_should_refuse(row)
        is_refusal = bool(row.get("is_refusal"))
        if should_refuse is True:
            stats["unsafe_total"] += 1
            stats["unsafe_refusal"] += int(is_refusal)
        elif should_refuse is False:
            stats["safe_total"] += 1
            stats["safe_refusal"] += int(is_refusal)
        else:
            stats["unknown_gold"] += 1
    return stats


def iter_judged_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def sort_key(row: Dict):
    return (
        DATASET_ORDER.get(row["dataset"], 99),
        MODEL_ORDER.get(row["model"], 99),
        METHOD_ORDER.get(row["method"], 99),
        row["method"],
    )


def make_markdown(rows: List[Dict]) -> str:
    lines: List[str] = []
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in sorted(rows, key=sort_key):
        grouped[row["dataset"]].append(row)

    for dataset in sorted(grouped, key=lambda x: DATASET_ORDER.get(x, 99)):
        lines.append(f"## {dataset}")
        if dataset == "xstest":
            lines.append("| Model | Method | Overall RR | Unsafe refusal | Safe refusal | Unknown gold |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for row in grouped[dataset]:
                lines.append(
                    "| {model} | {method} | {refusal}/{n} ({rr}) | "
                    "{unsafe_refusal}/{unsafe_total} ({unsafe_rr}) | "
                    "{safe_refusal}/{safe_total} ({safe_rr}) | {unknown_gold} |".format(
                        model=row["model"],
                        method=row["method"],
                        refusal=row["refusal"],
                        n=row["n"],
                        rr=pct(row["refusal"], row["n"]),
                        unsafe_refusal=row["unsafe_refusal"],
                        unsafe_total=row["unsafe_total"],
                        unsafe_rr=pct(row["unsafe_refusal"], row["unsafe_total"]),
                        safe_refusal=row["safe_refusal"],
                        safe_total=row["safe_total"],
                        safe_rr=pct(row["safe_refusal"], row["safe_total"]),
                        unknown_gold=row["unknown_gold"],
                    )
                )
        else:
            lines.append("| Model | Method | RR | Non-refusal | Unknown gold |")
            lines.append("|---|---|---:|---:|---:|")
            for row in grouped[dataset]:
                non_refusal = row["n"] - row["refusal"]
                lines.append(
                    "| {model} | {method} | {refusal}/{n} ({rr}) | {non_refusal}/{n} | {unknown_gold} |".format(
                        model=row["model"],
                        method=row["method"],
                        refusal=row["refusal"],
                        n=row["n"],
                        rr=pct(row["refusal"], row["n"]),
                        non_refusal=non_refusal,
                        unknown_gold=row["unknown_gold"],
                    )
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "method",
        "n",
        "refusal",
        "refusal_rate",
        "safe_total",
        "safe_refusal",
        "unsafe_total",
        "unsafe_refusal",
        "unknown_gold",
        "file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=sort_key):
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize judged safety-evaluation outputs."
    )
    parser.add_argument("--root", default="judged")
    parser.add_argument("--out-md", default="analysis/judged_summary.md")
    parser.add_argument("--out-csv", default="analysis/judged_summary.csv")
    args = parser.parse_args()

    root = Path(args.root)
    files = iter_judged_files(root)
    if not files:
        raise RuntimeError(f"No judged JSONL files found under {root}")

    rows = [summarize_file(path) for path in files]
    md = make_markdown(rows)
    print(md)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    write_csv(rows, Path(args.out_csv))

    print(f"[OK] wrote {out_md}")
    print(f"[OK] wrote {Path(args.out_csv)}")


if __name__ == "__main__":
    main()
