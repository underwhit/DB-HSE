import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


GROUP_KEYS = ["method", "setting", "label", "type", "category", "safety_label", "dataset"]


def iter_rows(path: Path) -> Iterable[Dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_file"] = file.name
                    yield row


def print_summary(name: str, rows: List[Dict]) -> None:
    total = len(rows)
    refusal = sum(1 for r in rows if r.get("is_refusal"))
    print(f"{name}: n={total} refusal={refusal} rate={refusal / total * 100 if total else 0:.2f}%")
    notes = Counter(str(r.get("judge_note", "")) for r in rows)
    if notes:
        print(f"  judge_notes={dict(notes)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    rows = list(iter_rows(Path(args.input)))
    print_summary("overall", rows)

    by_file = defaultdict(list)
    for row in rows:
        by_file[row["_file"]].append(row)
    for key, vals in sorted(by_file.items()):
        print_summary(f"file={key}", vals)

    for group_key in GROUP_KEYS:
        groups = defaultdict(list)
        for row in rows:
            if group_key in row and row[group_key] not in (None, ""):
                groups[str(row[group_key])].append(row)
        if groups:
            print(f"\nBy {group_key}:")
            for key, vals in sorted(groups.items()):
                print_summary(f"{group_key}={key}", vals)


if __name__ == "__main__":
    main()
