#!/usr/bin/env python3
"""Calibrate DB-HSE gate bounds on an XSTest calibration split.

This script estimates tau_low/tau_high from a fixed calibration set. It does not
select DB-HSE layer or alpha/scale. Use the same layer/alpha as the main
operating point, and run the resulting bounds on the held-out XSTest split.

By default, the harmful prototype is computed from unsafe calibration prompts.
For a cleaner transfer-style ablation, pass --proto-input to reuse the same
prototype source as the main DB-HSE experiment, and use the XSTest calibration
split only to choose gate bounds.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj or "label" not in obj:
                raise RuntimeError(f"{path}:{line_no} must contain prompt and label")
            obj["label"] = str(obj["label"]).strip().lower()
            rows.append(obj)
    if not rows:
        raise RuntimeError(f"{path}: no rows")
    return rows


def load_prompt_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "prompt" not in obj:
                    raise RuntimeError(f"{path}:{line_no} missing prompt")
                rows.append(obj)
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"{path}: expected top-level list")
    for idx, obj in enumerate(data):
        if not isinstance(obj, dict) or "prompt" not in obj:
            raise RuntimeError(f"{path}:{idx} missing prompt")
    return data


def chunked(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def compute_similarities(
    tokenizer,
    model,
    rows: List[Dict[str, Any]],
    layer_id: int,
    proto_vec: torch.Tensor,
    batch_size: int,
) -> List[Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    from run_dbhse_adaptive import prefill

    out: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for batch_rows in chunked(rows, batch_size):
            prompts = [str(row["prompt"]) for row in batch_rows]
            hs_with, span_with, _ = prefill(tokenizer, model, prompts, True)
            hs_without, span_without, _ = prefill(tokenizer, model, prompts, False)
            for b, row in enumerate(batch_rows):
                item = dict(row)
                item["span_found"] = False
                item["similarity"] = None
                if span_with[b] and span_without[b]:
                    sw, ew = span_with[b]
                    sb, eb = span_without[b]
                    token_len = min(ew - sw, eb - sb)
                    if token_len > 0:
                        diff = (
                            hs_with[layer_id][b, sw : sw + token_len]
                            - hs_without[layer_id][b, sb : sb + token_len]
                        )
                        direction = diff.mean(dim=0).float()
                        sim = F.cosine_similarity(direction, proto_vec, dim=0).item()
                        item["span_found"] = True
                        item["similarity"] = float(sim)
                out.append(item)
    return out


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise RuntimeError("cannot compute percentile over empty values")
    if q < 0 or q > 100:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * (q / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(vals) - 1)
    weight = pos - lower
    return vals[lower] * (1.0 - weight) + vals[upper] * weight


def choose_bounds(
    safe_sims: List[float],
    unsafe_sims: List[float],
    *,
    safe_quantile: float,
    unsafe_quantile: float,
    margin_ratio: float,
    fallback_gap: float,
) -> Tuple[float, float, Dict[str, Any]]:
    safe_q = percentile(safe_sims, safe_quantile)
    unsafe_q = percentile(unsafe_sims, unsafe_quantile)
    metadata: Dict[str, Any] = {
        "safe_quantile": safe_quantile,
        "unsafe_quantile": unsafe_quantile,
        "safe_q_value": safe_q,
        "unsafe_q_value": unsafe_q,
        "margin_ratio": margin_ratio,
        "fallback_gap": fallback_gap,
    }

    if safe_q < unsafe_q:
        margin = (unsafe_q - safe_q) * margin_ratio
        tau_low = safe_q + margin
        tau_high = unsafe_q - margin
        metadata["mode"] = "separated_percentiles"
        metadata["margin"] = margin
    else:
        tau_low = max(safe_sims)
        unsafe_mean = sum(float(v) for v in unsafe_sims) / len(unsafe_sims)
        tau_high = max(tau_low + fallback_gap, unsafe_mean)
        metadata["mode"] = "overlap_conservative"
        metadata["margin"] = None
    return float(tau_low), float(tau_high), metadata


def describe(values: List[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {
        "min": min(vals),
        "p05": percentile(vals, 5),
        "p25": percentile(vals, 25),
        "mean": sum(vals) / len(vals),
        "median": percentile(vals, 50),
        "p75": percentile(vals, 75),
        "p95": percentile(vals, 95),
        "max": max(vals),
    }


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    lines = [
        "# XSTest-Calibrated DB-HSE Gate",
        "",
        f"- Model identifier: `{result['model_id']}`",
        f"- Calibration input: `{result['input_file']}`",
        f"- Layer: `{result['layer']}`",
        f"- Prototype label/source: `{result['prototype_label']}`",
        f"- Prototype size: `{result['prototype_size']}`",
        f"- Bounds: `tau_low={result['tau_low']:.8f}`, `tau_high={result['tau_high']:.8f}`",
        f"- Bound mode: `{result['bound_metadata']['mode']}`",
        "",
        "## Similarity Summary",
        "",
        "| Label | n | min | p05 | p25 | mean | median | p75 | p95 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("safe", "unsafe"):
        stats = result["similarity_summary"][label]
        lines.append(
            "| {label} | {n} | {min:.4f} | {p05:.4f} | {p25:.4f} | {mean:.4f} | {median:.4f} | {p75:.4f} | {p95:.4f} | {max:.4f} |".format(
                label=label,
                n=result["label_counts"][label],
                **stats,
            )
        )
    lines.extend(
        [
            "",
            "## Recommended Held-Out Run Arguments",
            "",
            "```bash",
            f"--tau-low {result['tau_low']:.10f} \\",
            f"--tau-high {result['tau_high']:.10f}",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", type=Path, required=True, help="Calibration JSONL from make_xstest_calibration_split.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--prototype-label",
        choices=["unsafe", "all"],
        default="unsafe",
        help="If --proto-input is omitted, use unsafe calibration prompts for the harmful prototype, or all calibration prompts.",
    )
    parser.add_argument("--proto-input", type=Path, default=None, help="Optional external JSON/JSONL prompt file for computing the harmful prototype.")
    parser.add_argument("--proto-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--proto-batch-size", type=int, default=8)
    parser.add_argument("--safe-quantile", type=float, default=95.0)
    parser.add_argument("--unsafe-quantile", type=float, default=5.0)
    parser.add_argument("--margin-ratio", type=float, default=0.2)
    parser.add_argument("--fallback-gap", type=float, default=0.1)
    parser.add_argument("--model-tag", default=None, help="Optional filename tag, e.g. qwen25")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model_tag or Path(args.model_path).name.replace("/", "_")
    json_path = args.output_dir / f"{tag}_xstest_gate_calibration.json"
    md_path = args.output_dir / f"{tag}_xstest_gate_calibration.md"
    sim_path = args.output_dir / f"{tag}_xstest_gate_similarities.jsonl"
    for path in (json_path, md_path, sim_path):
        if path.exists() and not args.overwrite:
            raise RuntimeError(f"output exists: {path}; pass --overwrite")

    rows = load_jsonl(args.input)
    label_counts = Counter(row["label"] for row in rows)
    if label_counts["safe"] == 0 or label_counts["unsafe"] == 0:
        raise RuntimeError(f"calibration split must contain both safe and unsafe rows; got {dict(label_counts)}")

    if args.proto_input is not None:
        proto_rows = load_prompt_rows(args.proto_input)
        prototype_label = "external"
        prototype_source = args.proto_input.name
    elif args.prototype_label == "unsafe":
        proto_rows = [row for row in rows if row["label"] == "unsafe"]
        prototype_label = args.prototype_label
        prototype_source = args.input.name
    else:
        proto_rows = list(rows)
        prototype_label = args.prototype_label
        prototype_source = args.input.name
    proto_rows = proto_rows[: args.proto_size]
    if not proto_rows:
        raise RuntimeError("no prototype rows selected")

    import torch
    from transformers import AutoTokenizer

    from edit_qwen import EditQwen2ForCausalLM
    from run_dbhse_adaptive import compute_proto

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"[Load] {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = EditQwen2ForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    print(f"[Proto] label={prototype_label} source={prototype_source} size={len(proto_rows)} layer={args.layer}")
    proto_vec = compute_proto(
        tokenizer,
        model,
        [str(row["prompt"]) for row in proto_rows],
        args.layer,
        args.proto_batch_size,
    )

    print(f"[Sims] total={len(rows)} batch_size={args.batch_size}")
    sim_rows = compute_similarities(tokenizer, model, rows, args.layer, proto_vec, args.batch_size)
    found = [row for row in sim_rows if row["similarity"] is not None]
    safe_sims = [float(row["similarity"]) for row in found if row["label"] == "safe"]
    unsafe_sims = [float(row["similarity"]) for row in found if row["label"] == "unsafe"]
    if not safe_sims or not unsafe_sims:
        raise RuntimeError(f"need non-empty safe/unsafe sims; got safe={len(safe_sims)} unsafe={len(unsafe_sims)}")

    tau_low, tau_high, bound_metadata = choose_bounds(
        safe_sims,
        unsafe_sims,
        safe_quantile=args.safe_quantile,
        unsafe_quantile=args.unsafe_quantile,
        margin_ratio=args.margin_ratio,
        fallback_gap=args.fallback_gap,
    )
    for row in sim_rows:
        sim = row["similarity"]
        row["gate_at_calibrated_bounds"] = None if sim is None else max(0.0, min(1.0, (float(sim) - tau_low) / (tau_high - tau_low)))

    result = {
        "model_id": Path(args.model_path).name,
        "input_file": args.input.name,
        "layer": args.layer,
        "prototype_label": prototype_label,
        "prototype_source": prototype_source,
        "prototype_size": len(proto_rows),
        "proto_size_requested": args.proto_size,
        "label_counts": {"safe": label_counts["safe"], "unsafe": label_counts["unsafe"]},
        "span_found_counts": {
            "safe": sum(1 for row in found if row["label"] == "safe"),
            "unsafe": sum(1 for row in found if row["label"] == "unsafe"),
        },
        "tau_low": tau_low,
        "tau_high": tau_high,
        "bound_metadata": bound_metadata,
        "similarity_summary": {
            "safe": describe(safe_sims),
            "unsafe": describe(unsafe_sims),
        },
    }

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(md_path, result)
    with sim_path.open("w", encoding="utf-8") as f:
        for row in sim_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] tau_low={tau_low:.8f} tau_high={tau_high:.8f} mode={bound_metadata['mode']}")
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(f"[OK] wrote {sim_path}")


if __name__ == "__main__":
    main()
