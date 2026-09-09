import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from export_xstest_safe_false_refusals import (
    load_disagreement_file,
    load_judged_dir,
    selected_safe_false_refusals,
)


def iter_simple_records(rows: Iterable[Dict], only_method: Optional[str]) -> Iterable[Dict]:
    for row in rows:
        responses = row.get("responses", {})
        meta = row.get("metadata", {})
        methods = [only_method] if only_method else sorted(responses)

        for method in methods:
            resp = responses.get(method)
            if not resp or not bool(resp.get("is_refusal")):
                continue

            yield {
                "index": row.get("index"),
                "method": method,
                "label": meta.get("label") or meta.get("safety_label") or "safe",
                "type": meta.get("type") or meta.get("category") or meta.get("prompt_type"),
                "prompt": row.get("prompt"),
                "response": resp.get("response", ""),
                "auto_is_refusal": True,
                "judge_source": resp.get("judge_source"),
                "judge_note": resp.get("judge_note"),
                "manual_judgment": "",
                "manual_note": "",
            }


def write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a simple JSONL file for manually checking whether automatic "
            "refusal judgments on safe XSTest samples are reasonable."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Judged XSTest directory OR xstest_qwen25_disagreements.jsonl",
    )
    parser.add_argument(
        "--output",
        default="analysis/xstest_safe_refusal_manual_check.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--only-method",
        default=None,
        help="Only export one method, e.g. dbhse_adaptive.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if in_path.is_dir():
        aligned = load_judged_dir(in_path)
    else:
        aligned = load_disagreement_file(in_path)

    safe_false_refusal_rows = selected_safe_false_refusals(aligned, args.only_method)
    records = list(iter_simple_records(safe_false_refusal_rows, args.only_method))
    out_path = Path(args.output)
    write_jsonl(records, out_path)

    print(f"[OK] wrote {len(records)} simple manual-check rows")
    print(f"  output: {out_path}")
    print("  Fill manual_judgment with: true / false / unclear")
    print("  true = the automatic refusal judgment is reasonable; false = it is not.")


if __name__ == "__main__":
    main()
