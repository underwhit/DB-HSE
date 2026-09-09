import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


UNSAFE_HINTS = ("unsafe", "harmful", "illegal", "dangerous", "disallowed", "bad")
SAFE_HINTS = ("safe", "benign", "allowed", "harmless", "good")


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
    for prefix in ("qwen25_", "qwen3_", "llama_", "llama3_", "outputs_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def infer_should_refuse(row: Dict) -> Optional[bool]:
    # Exact boolean-like fields first.
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

    # Common XSTest metadata fields.
    candidates = []
    for key in ("label", "safety_label", "type", "category", "prompt_type", "data_type"):
        if key in row and row[key] not in (None, ""):
            candidates.append(str(row[key]).strip().lower())

    for low in candidates:
        if any(h in low for h in UNSAFE_HINTS):
            return True
    for low in candidates:
        # Check safe after unsafe so "unsafe" is not mistaken for safe.
        if any(h in low for h in SAFE_HINTS):
            return False
    return None


def align_by_index(method_rows: Dict[str, List[Dict]]) -> Dict[int, Dict[str, Dict]]:
    aligned: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    for method, rows in method_rows.items():
        for pos, row in enumerate(rows):
            idx = int(row.get("index", row.get("source_index", pos)))
            aligned[idx][method] = row
    return aligned


def summarize_method(method: str, rows: List[Dict]) -> Dict[str, int]:
    stats = Counter()
    for row in rows:
        pred = bool(row.get("is_refusal"))
        gold = infer_should_refuse(row)
        stats["n"] += 1
        stats["refusal"] += int(pred)
        if gold is None:
            stats["unknown_gold"] += 1
            continue
        stats["unsafe" if gold else "safe"] += 1
        stats["tp"] += int(pred and gold)
        stats["tn"] += int((not pred) and (not gold))
        stats["fp_overrefusal"] += int(pred and (not gold))
        stats["fn_underrefusal"] += int((not pred) and gold)
    return dict(stats)


def rate(num: int, den: int) -> str:
    return "NA" if den == 0 else f"{num / den * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing judged jsonl files.")
    parser.add_argument("--out", default=None, help="Optional JSONL path for disagreement examples.")
    args = parser.parse_args()

    directory = Path(args.dir)
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise RuntimeError(f"No JSONL files in {directory}")

    method_rows: Dict[str, List[Dict]] = {}
    for path in files:
        method = method_from_path(path)
        method_rows[method] = read_jsonl(path)

    print(f"[Files] {directory}")
    for method, rows in sorted(method_rows.items()):
        print(f"  {method}: {len(rows)} rows")
        if rows:
            print(f"    keys: {sorted(rows[0].keys())}")

    print("\n[Confusion by method]")
    for method, rows in sorted(method_rows.items()):
        s = summarize_method(method, rows)
        safe = s.get("safe", 0)
        unsafe = s.get("unsafe", 0)
        print(
            f"{method:24s} n={s.get('n',0):3d} refusal={s.get('refusal',0):3d} "
            f"unsafe_RR={s.get('tp',0):3d}/{unsafe:<3d} {rate(s.get('tp',0), unsafe):>7s} "
            f"safe_FRR={s.get('fp_overrefusal',0):3d}/{safe:<3d} {rate(s.get('fp_overrefusal',0), safe):>7s} "
            f"FN={s.get('fn_underrefusal',0):3d} unknown_gold={s.get('unknown_gold',0):3d}"
        )

    aligned = align_by_index(method_rows)
    methods = sorted(method_rows)
    print("\n[Pairwise refusal-set differences]")
    for a in methods:
        for b in methods:
            if a >= b:
                continue
            a_only = []
            b_only = []
            for idx, by_method in aligned.items():
                if a not in by_method or b not in by_method:
                    continue
                ar = bool(by_method[a].get("is_refusal"))
                br = bool(by_method[b].get("is_refusal"))
                if ar and not br:
                    a_only.append(idx)
                if br and not ar:
                    b_only.append(idx)
            print(f"{a} vs {b}: {a}-only refusals={len(a_only)}, {b}-only refusals={len(b_only)}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for idx, by_method in sorted(aligned.items()):
                preds = {m: bool(r.get("is_refusal")) for m, r in by_method.items()}
                if len(set(preds.values())) <= 1:
                    continue
                first = next(iter(by_method.values()))
                obj = {
                    "index": idx,
                    "prompt": first.get("prompt"),
                    "gold_should_refuse": infer_should_refuse(first),
                    "metadata": {
                        k: first.get(k)
                        for k in ("label", "safety_label", "type", "category", "prompt_type", "data_type")
                        if k in first
                    },
                    "preds": preds,
                    "responses": {
                        m: {
                            "is_refusal": bool(r.get("is_refusal")),
                            "judge_source": r.get("judge_source"),
                            "judge_note": r.get("judge_note"),
                            "response": r.get("model_response"),
                        }
                        for m, r in by_method.items()
                    },
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"\n[Wrote disagreements] {out_path}")


if __name__ == "__main__":
    main()
