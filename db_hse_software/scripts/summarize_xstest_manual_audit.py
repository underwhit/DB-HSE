import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from export_xstest_safe_false_refusals import infer_should_refuse, method_from_path


TRUE_VALUES = {"true", "reasonable", "yes", "1"}
FALSE_VALUES = {"false", "unreasonable", "no", "0", "fasle"}
UNCLEAR_VALUES = {"unclear", "unknown", "maybe", "null", "none", ""}


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_manual(value) -> str:
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return "true"
    if text in FALSE_VALUES:
        return "false"
    if text in UNCLEAR_VALUES:
        return "unclear"
    return "unknown"


def load_judged_dir(path: Path) -> Dict[str, List[Dict]]:
    by_method: Dict[str, List[Dict]] = {}
    for file in sorted(path.glob("*.jsonl")):
        method = method_from_path(file)
        by_method[method] = list(read_jsonl(file))
    return by_method


def load_manual(path: Path) -> Dict[Tuple[str, int], str]:
    labels: Dict[Tuple[str, int], str] = {}
    for row in read_jsonl(path):
        method = str(row.get("method", "")).strip()
        if not method:
            continue
        idx = row.get("index", row.get("sample_index"))
        if idx is None:
            continue
        labels[(method, int(idx))] = normalize_manual(row.get("manual_judgment"))
    return labels


def pct(num: int, den: int) -> str:
    if den == 0:
        return "NA"
    return f"{num / den * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize XSTest refusal results with human-corrected safe false-refusal "
            "counts. The main table omits the automatic safe-refusal column."
        )
    )
    parser.add_argument("--judged-dir", required=True, help="Directory containing judged XSTest JSONL files.")
    parser.add_argument("--manual", required=True, help="Manual audit JSONL from make_xstest_simple_manual_review_jsonl.py.")
    args = parser.parse_args()

    judged = load_judged_dir(Path(args.judged_dir))
    manual = load_manual(Path(args.manual))

    print("| Method | Unsafe refusal ↑ | Safe refusal, human-audited ↓ |")
    print("|---|---:|---:|")

    diagnostics = []
    for method in sorted(judged):
        rows = judged[method]
        safe_rows = []
        unsafe_rows = []
        unknown_rows = []
        for pos, row in enumerate(rows):
            should_refuse = infer_should_refuse(row)
            if should_refuse is True:
                unsafe_rows.append((pos, row))
            elif should_refuse is False:
                safe_rows.append((pos, row))
            else:
                unknown_rows.append((pos, row))

        unsafe_refusals = sum(1 for _, row in unsafe_rows if bool(row.get("is_refusal")))

        safe_auto_refusals = 0
        manual_true = 0
        manual_false = 0
        manual_unclear = 0
        manual_missing = 0
        for pos, row in safe_rows:
            if not bool(row.get("is_refusal")):
                continue
            safe_auto_refusals += 1
            idx = int(row.get("index", row.get("source_index", pos)))
            label = manual.get((method, idx))
            if label == "true":
                manual_true += 1
            elif label == "false":
                manual_false += 1
            elif label == "unclear":
                manual_unclear += 1
            else:
                manual_missing += 1

        print(
            f"| {method} | {unsafe_refusals}/{len(unsafe_rows)} ({pct(unsafe_refusals, len(unsafe_rows))}) "
            f"| {manual_true}/{len(safe_rows)} ({pct(manual_true, len(safe_rows))}) |"
        )
        diagnostics.append(
            {
                "method": method,
                "safe_total": len(safe_rows),
                "unsafe_total": len(unsafe_rows),
                "unknown_total": len(unknown_rows),
                "safe_auto_refusals": safe_auto_refusals,
                "manual_true": manual_true,
                "manual_false": manual_false,
                "manual_unclear": manual_unclear,
                "manual_missing": manual_missing,
            }
        )

    print("\nDiagnostics, not for main table:")
    for item in diagnostics:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
