import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROMPT_COLUMNS = [
    "prompt",
    "goal",
    "behavior",
    "instruction",
    "question",
    "text",
    "user_prompt",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"{path} has no CSV header.")
        if any("429" in str(x) or "Too Many Requests" in str(x) for x in reader.fieldnames):
            raise RuntimeError(
                f"{path} looks like a failed download (429 Too Many Requests), not a dataset CSV."
            )
        return [dict(row) for row in reader]


def choose_prompt_column(rows: List[Dict[str, str]], preferred: Optional[str]) -> str:
    if not rows:
        raise RuntimeError("CSV has no rows.")
    fields = list(rows[0].keys())
    if preferred:
        if preferred not in fields:
            raise RuntimeError(f"Requested prompt column {preferred!r} not in {fields}")
        return preferred
    lowered = {f.lower(): f for f in fields}
    for name in PROMPT_COLUMNS:
        if name in lowered:
            return lowered[name]
    raise RuntimeError(
        f"Could not infer prompt column. Available columns: {fields}. "
        f"Pass --prompt-column explicitly."
    )


def iter_selected(rows: List[Dict[str, str]], limit: Optional[int]) -> Iterable[Dict[str, str]]:
    count = 0
    for row in rows:
        if limit is not None and count >= limit:
            break
        count += 1
        yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--prompt-column", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = read_csv(in_path)
    prompt_col = choose_prompt_column(rows, args.prompt_column)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(iter_selected(rows, args.limit)):
            prompt = (row.get(prompt_col) or "").strip()
            if not prompt:
                continue
            obj = {
                "index": written,
                "source_index": idx,
                "dataset": args.dataset_name,
                "prompt": prompt,
            }
            for key, value in row.items():
                if key == prompt_col:
                    continue
                obj[key] = value
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"[OK] {args.dataset_name}: wrote {written} prompts to {out_path}")
    print(f"     prompt_column={prompt_col!r}, input_rows={len(rows)}")


if __name__ == "__main__":
    main()
