#!/usr/bin/env python3
"""Run DB-HSE with unconditional per-prompt hidden-state injection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer

from edit_qwen import EditQwen2ForCausalLM
from hs_method import HiddenStatesEditingMethod
from run_dbhse_adaptive import (
    chunked,
    get_input_device,
    load_jsonl_or_json,
    prefill,
)


def make_edit_method(
    inject_layer_idx: int,
    edit_tensor: torch.Tensor,
    seq_len: int,
) -> HiddenStatesEditingMethod:
    device = edit_tensor.device
    method = HiddenStatesEditingMethod(
        if_edit=True,
        edit_layer_idx=[inject_layer_idx],
        edit_mask=torch.zeros(seq_len, dtype=torch.bool, device=device),
        edit_tensor={inject_layer_idx: edit_tensor},
    )
    method.prompt_only = True
    return method


@torch.inference_mode()
def generate_dbhse_direct_batch(
    tokenizer,
    model,
    rows: List[Dict[str, Any]],
    hs_layer_id: int,
    alpha: float,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    prompts = [str(row["prompt"]) for row in rows]
    hs_with, span_with, _ = prefill(tokenizer, model, prompts, True)
    hs_without, span_without, enc_without = prefill(tokenizer, model, prompts, False)

    inject_layer_idx = hs_layer_id - 1
    batch_size, seq_len = enc_without["input_ids"].shape
    edit_tensor = torch.zeros(
        (batch_size, seq_len, model.config.hidden_size),
        dtype=model.dtype,
        device=get_input_device(model),
    )
    injection_info: List[Dict[str, Optional[float]]] = []

    for batch_idx in range(batch_size):
        info: Dict[str, Optional[float]] = {
            "span_found": 0.0,
            "gate": None,
            "final_alpha": 0.0,
            "delta_l2_mean_over_user_tokens": None,
        }
        if span_with[batch_idx] and span_without[batch_idx]:
            start_with, end_with = span_with[batch_idx]
            start_without, end_without = span_without[batch_idx]
            token_len = min(end_with - start_with, end_without - start_without)
            if token_len > 0:
                diff_matrix = (
                    hs_with[hs_layer_id][
                        batch_idx, start_with : start_with + token_len
                    ]
                    - hs_without[hs_layer_id][
                        batch_idx, start_without : start_without + token_len
                    ]
                )
                scaled_diff = diff_matrix * float(alpha)
                edit_tensor[
                    batch_idx, start_without : start_without + token_len, :
                ] = scaled_diff.to(
                    device=edit_tensor.device,
                    dtype=model.dtype,
                )
                info = {
                    "span_found": 1.0,
                    "gate": 1.0,
                    "final_alpha": float(alpha),
                    "delta_l2_mean_over_user_tokens": float(
                        torch.norm(scaled_diff.float(), dim=-1).mean().item()
                    ),
                }
        injection_info.append(info)

    edit_method = make_edit_method(inject_layer_idx, edit_tensor, seq_len)
    out_ids = model.generate(
        input_ids=enc_without["input_ids"],
        attention_mask=enc_without.get("attention_mask"),
        edit_method=edit_method,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    input_length = enc_without["input_ids"].shape[1]

    out_rows: List[Dict[str, Any]] = []
    for row, sequence, info in zip(rows, out_ids, injection_info):
        obj = dict(row)
        obj.update(
            {
                "method": "DB-HSE-Direct",
                "model_response": tokenizer.decode(
                    sequence[input_length:], skip_special_tokens=True
                ).strip(),
                "target_layer": hs_layer_id,
                "hs_layer_id": hs_layer_id,
                "inject_layer_idx": inject_layer_idx,
                "alpha": float(alpha),
                "gate_mode": "direct",
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
        )
        obj.update(info)
        out_rows.append(obj)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="1-based hidden-state id; hidden_states[k] is injected into decoder block k-1.",
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    rows = load_jsonl_or_json(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.overwrite:
            args.output.unlink()
        else:
            raise RuntimeError(f"Output exists: {args.output}; pass --overwrite")

    print(f"[Load] {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True
    )
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

    num_layers = int(model.config.num_hidden_layers)
    if not 1 <= args.layer <= num_layers:
        raise RuntimeError(
            f"Invalid --layer {args.layer}; expected 1..{num_layers} for hidden-state ids"
        )

    print(
        f"[Run] total={len(rows)} hs_layer={args.layer} "
        f"inject_block={args.layer - 1} alpha={args.alpha} gate=disabled"
    )
    done = 0
    with args.output.open("w", encoding="utf-8") as f:
        for batch_rows in chunked(rows, args.batch_size):
            out_rows = generate_dbhse_direct_batch(
                tokenizer,
                model,
                batch_rows,
                args.layer,
                args.alpha,
                args.max_new_tokens,
            )
            for obj in out_rows:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()
            done += len(batch_rows)
            print(f"[Progress] {done}/{len(rows)}")

    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
