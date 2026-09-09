import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SAFE_HINTS = ("safe", "benign", "allowed", "harmless", "good")
UNSAFE_HINTS = ("unsafe", "harmful", "illegal", "dangerous", "disallowed", "bad")


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def method_from_path(path: Path) -> str:
    name = path.stem.lower()
    for prefix in ("qwen25_", "qwen3_", "llama3_", "llama_", "outputs_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def infer_should_refuse(row: Dict) -> Optional[bool]:
    for key in ("should_refuse", "is_unsafe", "is_harmful", "harmful"):
        if key in row:
            val = row[key]
            if isinstance(val, bool):
                return val
            low = str(val).strip().lower()
            if low in ("1", "true", "yes", "y"):
                return True
            if low in ("0", "false", "no", "n"):
                return False

    values = []
    for key in ("label", "safety_label", "type", "category", "prompt_type", "data_type"):
        if key in row and row[key] not in (None, ""):
            values.append(str(row[key]).strip().lower())

    for value in values:
        if any(h in value for h in UNSAFE_HINTS):
            return True
    for value in values:
        if any(h in value for h in SAFE_HINTS):
            return False
    return None


def load_judged_dir(directory: Path) -> Dict[int, Dict]:
    aligned: Dict[int, Dict] = defaultdict(lambda: {"responses": {}, "preds": {}})
    for path in sorted(directory.glob("*.jsonl")):
        method = method_from_path(path)
        for pos, row in enumerate(read_jsonl(path)):
            idx = int(row.get("index", row.get("source_index", pos)))
            entry = aligned[idx]
            entry["index"] = idx
            entry["prompt"] = row.get("prompt")
            entry["gold_should_refuse"] = infer_should_refuse(row)
            entry["metadata"] = {
                k: row.get(k)
                for k in ("label", "safety_label", "type", "category", "prompt_type", "data_type")
                if k in row
            }
            entry["preds"][method] = bool(row.get("is_refusal"))
            entry["responses"][method] = {
                "is_refusal": bool(row.get("is_refusal")),
                "judge_source": row.get("judge_source"),
                "judge_note": row.get("judge_note"),
                "response": row.get("model_response", ""),
            }
    return dict(aligned)


def load_disagreement_file(path: Path) -> Dict[int, Dict]:
    aligned: Dict[int, Dict] = {}
    for row in read_jsonl(path):
        aligned[int(row["index"])] = row
    return aligned


def selected_safe_false_refusals(aligned: Dict[int, Dict], only_method: Optional[str]) -> List[Dict]:
    selected: List[Dict] = []
    for idx, row in sorted(aligned.items()):
        if row.get("gold_should_refuse") is not False:
            continue
        preds = row.get("preds", {})
        if only_method:
            if not bool(preds.get(only_method)):
                continue
        elif not any(bool(v) for v in preds.values()):
            continue
        selected.append(row)
    return selected


def compact(text: str, limit: int = 300) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def write_csv(rows: List[Dict], out_path: Path) -> None:
    methods = sorted({m for row in rows for m in row.get("responses", {})})
    fields = ["index", "label", "type", "prompt"]
    for method in methods:
        fields.extend([
            f"{method}_is_refusal",
            f"{method}_judge_source",
            f"{method}_judge_note",
            f"{method}_manual_refusal",
            f"{method}_manual_note",
            f"{method}_response",
        ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            meta = row.get("metadata", {})
            obj = {
                "index": row.get("index"),
                "label": meta.get("label") or meta.get("safety_label"),
                "type": meta.get("type") or meta.get("category") or meta.get("prompt_type"),
                "prompt": row.get("prompt"),
            }
            for method in methods:
                resp = row.get("responses", {}).get(method, {})
                obj[f"{method}_is_refusal"] = resp.get("is_refusal")
                obj[f"{method}_judge_source"] = resp.get("judge_source")
                obj[f"{method}_judge_note"] = resp.get("judge_note")
                obj[f"{method}_manual_refusal"] = ""
                obj[f"{method}_manual_note"] = ""
                obj[f"{method}_response"] = resp.get("response", "")
            writer.writerow(obj)


def write_markdown(rows: List[Dict], out_path: Path) -> None:
    methods = sorted({m for row in rows for m in row.get("responses", {})})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# XSTest Safe False Refusal Review\n\n")
        f.write(f"Total safe samples with at least one refusal: {len(rows)}\n\n")
        f.write("Manual label suggestion: mark each response as `refusal`, `not_refusal`, or `unclear`.\n\n")
        for row in rows:
            meta = row.get("metadata", {})
            f.write(f"## Index {row.get('index')} | type={meta.get('type') or meta.get('category')} | label={meta.get('label') or meta.get('safety_label')}\n\n")
            f.write(f"**Prompt:** {row.get('prompt')}\n\n")
            for method in methods:
                resp = row.get("responses", {}).get(method)
                if not resp:
                    continue
                f.write(
                    f"### {method} | judge={resp.get('is_refusal')} "
                    f"({resp.get('judge_source')}/{resp.get('judge_note')})\n\n"
                )
                f.write("Manual refusal? `[ ] refusal  [ ] not_refusal  [ ] unclear`\n\n")
                f.write(f"> {compact(resp.get('response'), 1200)}\n\n")
            f.write("---\n\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Judged XSTest dir OR xstest_qwen25_disagreements.jsonl")
    parser.add_argument("--out-prefix", default="analysis/xstest_safe_false_refusals")
    parser.add_argument("--only-method", default=None, help="Example: dbhse_adaptive")
    args = parser.parse_args()

    in_path = Path(args.input)
    if in_path.is_dir():
        aligned = load_judged_dir(in_path)
    else:
        aligned = load_disagreement_file(in_path)

    rows = selected_safe_false_refusals(aligned, args.only_method)
    prefix = Path(args.out_prefix)
    write_csv(rows, prefix.with_suffix(".csv"))
    write_markdown(rows, prefix.with_suffix(".md"))

    print(f"[OK] exported {len(rows)} safe false-refusal candidates")
    print(f"  CSV: {prefix.with_suffix('.csv')}")
    print(f"  MD : {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
