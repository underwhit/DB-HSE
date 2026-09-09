import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from edit_qwen import EditQwen2ForCausalLM
from hs_method import HiddenStatesEditingMethod


SYSTEM_PROMPT = (
    "You are a helpful assistant. Before answering, carefully assess whether the user's "
    "request is safe and harmless. If the request involves wrongdoing, unsafe instructions, "
    "or illegal activity, refuse and instead offer safe, legal, and preventive guidance. "
    "If it is harmless, answer normally."
)


def load_jsonl_or_json(path: Path) -> List[Dict[str, Any]]:
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
    for i, obj in enumerate(data):
        if "prompt" not in obj:
            raise RuntimeError(f"{path}:{i} missing prompt")
    return data


def chunked(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_texts(tokenizer, prompts: List[str], with_system: bool) -> List[str]:
    texts = []
    for prompt in prompts:
        if with_system:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        texts.append(apply_chat_template(tokenizer, messages))
    return texts


def get_input_device(model) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_user_span_by_offsets(
    full_text: str,
    user_text: str,
    offsets_bt2: torch.Tensor,
    attn_mask_bt: torch.Tensor,
) -> Optional[Tuple[int, int]]:
    start_char = full_text.rfind(user_text)
    if start_char < 0:
        return None
    end_char = start_char + len(user_text)
    offsets = offsets_bt2.cpu().numpy()
    attn = attn_mask_bt.detach().cpu().numpy()
    indices = [
        i
        for i, (a, b) in enumerate(offsets)
        if attn[i] != 0 and (a != 0 or b != 0) and a < end_char and b > start_char
    ]
    if not indices:
        return None
    return int(min(indices)), int(max(indices)) + 1


@torch.inference_mode()
def prefill(tokenizer, model, prompts: List[str], with_system: bool):
    texts = build_texts(tokenizer, prompts, with_system)
    enc_cpu = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    )
    offsets = enc_cpu.pop("offset_mapping").cpu()
    device = get_input_device(model)
    enc = {k: v.to(device) for k, v in enc_cpu.items()}
    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc.get("attention_mask"),
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
        edit_method=None,
    )
    spans = [
        find_user_span_by_offsets(texts[b], prompts[b], offsets[b], enc["attention_mask"][b])
        for b in range(len(prompts))
    ]
    return out.hidden_states, spans, enc


@torch.inference_mode()
def compute_proto(tokenizer, model, prompts: List[str], layer_id: int, batch_size: int) -> torch.Tensor:
    deltas: List[torch.Tensor] = []
    for batch_prompts in chunked(prompts, batch_size):
        hs_with, span_with, _ = prefill(tokenizer, model, batch_prompts, True)
        hs_without, span_without, _ = prefill(tokenizer, model, batch_prompts, False)
        for b in range(len(batch_prompts)):
            if not (span_with[b] and span_without[b]):
                continue
            sw, ew = span_with[b]
            sb, eb = span_without[b]
            token_len = min(ew - sw, eb - sb)
            if token_len <= 0:
                continue
            diff = (
                hs_with[layer_id][b, sw : sw + token_len]
                - hs_without[layer_id][b, sb : sb + token_len]
            )
            deltas.append(diff.mean(dim=0).float())
    if not deltas:
        raise RuntimeError("No prototype deltas could be extracted.")
    return torch.stack(deltas, dim=0).mean(dim=0)


def make_edit_method(
    layer_id: int,
    edit_tensor: torch.Tensor,
    seq_len: int,
) -> HiddenStatesEditingMethod:
    device = edit_tensor.device
    method = HiddenStatesEditingMethod(
        if_edit=True,
        edit_layer_idx=[layer_id],
        edit_mask=torch.zeros(seq_len, dtype=torch.bool, device=device),
        edit_tensor={layer_id: edit_tensor},
    )
    method.prompt_only = True
    return method


@torch.inference_mode()
def generate_dbhse_batch(
    tokenizer,
    model,
    rows: List[Dict[str, Any]],
    layer_id: int,
    alpha: float,
    tau_low: float,
    tau_high: float,
    proto_vec: torch.Tensor,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    prompts = [str(r["prompt"]) for r in rows]
    hs_with, span_with, _ = prefill(tokenizer, model, prompts, True)
    hs_without, span_without, enc_without = prefill(tokenizer, model, prompts, False)

    batch_size, seq_len = enc_without["input_ids"].shape
    edit_tensor = torch.zeros(
        (batch_size, seq_len, model.config.hidden_size),
        dtype=model.dtype,
        device=get_input_device(model),
    )
    gate_info: List[Dict[str, Optional[float]]] = []

    for b in range(batch_size):
        info: Dict[str, Optional[float]] = {
            "span_found": 0.0,
            "similarity": None,
            "gate": None,
            "final_alpha": 0.0,
        }
        if span_with[b] and span_without[b]:
            sw, ew = span_with[b]
            sb, eb = span_without[b]
            token_len = min(ew - sw, eb - sb)
            if token_len > 0:
                diff_matrix = (
                    hs_with[layer_id][b, sw : sw + token_len]
                    - hs_without[layer_id][b, sb : sb + token_len]
                )
                current_direction = diff_matrix.mean(dim=0).float()
                sim = F.cosine_similarity(current_direction, proto_vec, dim=0).item()
                gate = max(0.0, min(1.0, (sim - tau_low) / (tau_high - tau_low)))
                final_alpha = alpha * gate
                if final_alpha > 0:
                    edit_tensor[b, sb : sb + token_len, :] = diff_matrix.to(
                        device=edit_tensor.device,
                        dtype=model.dtype,
                    ) * final_alpha
                info = {
                    "span_found": 1.0,
                    "similarity": float(sim),
                    "gate": float(gate),
                    "final_alpha": float(final_alpha),
                }
        gate_info.append(info)

    edit_method = make_edit_method(layer_id, edit_tensor, seq_len)
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
    for row, seq, info in zip(rows, out_ids, gate_info):
        obj = dict(row)
        obj.update(
            {
                "method": "DB-HSE-Adaptive",
                "model_response": tokenizer.decode(seq[input_length:], skip_special_tokens=True).strip(),
                "target_layer": layer_id,
                "alpha": alpha,
                "tau_low": tau_low,
                "tau_high": tau_high,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
        )
        obj.update(info)
        out_rows.append(obj)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--proto-input", default="dataset.json")
    parser.add_argument("--proto-size", type=int, default=100)
    parser.add_argument("--layer", type=int, required=True, help="HF hidden-state layer id, e.g. 17")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--tau-low", type=float, required=True)
    parser.add_argument("--tau-high", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--proto-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    rows = load_jsonl_or_json(Path(args.input))
    proto_rows = load_jsonl_or_json(Path(args.proto_input))[: args.proto_size]
    proto_prompts = [str(r["prompt"]) for r in proto_rows]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if args.overwrite:
            output_path.unlink()
        else:
            raise RuntimeError(f"Output exists: {output_path}; pass --overwrite")

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

    print(f"[Proto] size={len(proto_prompts)} layer={args.layer}")
    proto_vec = compute_proto(tokenizer, model, proto_prompts, args.layer, args.proto_batch_size)

    print(f"[Run] total={len(rows)} alpha={args.alpha} tau=[{args.tau_low},{args.tau_high}]")
    done = 0
    with output_path.open("w", encoding="utf-8") as f:
        for batch_rows in chunked(rows, args.batch_size):
            out_rows = generate_dbhse_batch(
                tokenizer,
                model,
                batch_rows,
                args.layer,
                args.alpha,
                args.tau_low,
                args.tau_high,
                proto_vec,
                args.max_new_tokens,
            )
            for obj in out_rows:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            done += len(batch_rows)
            if done % (args.batch_size * 10) == 0:
                print(f"[Progress] {done}/{len(rows)}")

    print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
