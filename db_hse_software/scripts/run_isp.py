import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer

from edit_qwen import EditQwen2ForCausalLM


SYSTEM_PROMPT = (
    "You are a helpful assistant. Before answering, carefully assess whether the user's request is safe and harmless. "
    "If the request involves wrongdoing, unsafe instructions, or illegal activity, refuse and instead offer safe, "
    "legal, and preventive guidance. If it is harmless, answer normally."
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


def build_texts_user_only(tokenizer, prompts: List[str]) -> List[str]:
    return [
        apply_chat_template(tokenizer, [{"role": "user", "content": prompt}])
        for prompt in prompts
    ]


def build_texts_with_system(tokenizer, prompts: List[str]) -> List[str]:
    return [
        apply_chat_template(
            tokenizer,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        for prompt in prompts
    ]


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
    indices = []
    for i, (a, b) in enumerate(offsets):
        if attn[i] == 0 or (a == 0 and b == 0):
            continue
        if a < end_char and b > start_char:
            indices.append(i)
    if not indices:
        return None
    return int(min(indices)), int(max(indices)) + 1


@torch.inference_mode()
def get_hidden_states_last_token(tokenizer, model, prompts: List[str], with_system: bool, layer: int):
    texts = build_texts_with_system(tokenizer, prompts) if with_system else build_texts_user_only(tokenizer, prompts)
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
        edit_method=None,
    )
    hidden = out.hidden_states[layer]
    last_indices = (enc["attention_mask"].sum(dim=1) - 1).to(hidden.device)
    batch_indices = torch.arange(enc["input_ids"].size(0), device=hidden.device)
    return hidden[batch_indices, last_indices, :]


@torch.inference_mode()
def compute_steering_params(tokenizer, model, prompts: List[str], layer: int, batch_size: int):
    diff_acc: List[torch.Tensor] = []
    sys_acc: List[torch.Tensor] = []
    for batch_prompts in chunked(prompts, batch_size):
        vecs_sys = get_hidden_states_last_token(tokenizer, model, batch_prompts, True, layer)
        vecs_no_sys = get_hidden_states_last_token(tokenizer, model, batch_prompts, False, layer)
        diff_acc.append((vecs_sys - vecs_no_sys).cpu())
        sys_acc.append(vecs_sys.cpu())

    all_diffs = torch.cat(diff_acc, dim=0)
    all_sys = torch.cat(sys_acc, dim=0).to(model.device)
    u = all_diffs.mean(dim=0).to(model.device)
    u = u / u.norm()
    beta = torch.matmul(all_sys, u).mean()
    return {"u": u, "beta": beta}


@torch.inference_mode()
def prefill_for_generation(tokenizer, model, prompts: List[str]):
    texts = build_texts_user_only(tokenizer, prompts)
    enc_cpu = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    )
    offsets = enc_cpu["offset_mapping"].cpu()
    enc = {k: v.to(model.device) for k, v in enc_cpu.items() if k != "offset_mapping"}
    spans = [
        find_user_span_by_offsets(texts[b], prompts[b], offsets[b], enc["attention_mask"][b])
        for b in range(len(prompts))
    ]
    return enc, spans


class DynamicSteeringHook:
    def __init__(
        self,
        u_vector: torch.Tensor,
        beta: torch.Tensor,
        span_list: List[Optional[Tuple[int, int]]],
        scale: float,
    ) -> None:
        self.u = u_vector
        self.beta = beta
        self.span_list = span_list
        self.scale = scale
        self.handle = None

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # With device_map="auto", the selected layer may not be on the input
        # embedding device. Move the small steering parameters to the layer
        # output device instead of assuming every decoder block is on cuda:0.
        u = self.u.to(device=hidden.device, dtype=hidden.dtype)
        beta = self.beta.to(device=hidden.device, dtype=hidden.dtype)
        proj = torch.matmul(hidden, u)
        coeff = torch.clamp((beta - proj) * self.scale, min=0.0)

        if hidden.shape[1] > 1:
            mask = torch.zeros_like(coeff)
            for batch_idx, span in enumerate(self.span_list):
                if span is None:
                    continue
                start, end = span
                span_len = min(end - start, max(hidden.shape[1] - start, 0))
                if span_len > 0:
                    mask[batch_idx, start : start + span_len] = 1.0
        else:
            mask = torch.ones_like(coeff)

        steered = hidden + (coeff * mask).unsqueeze(-1) * u.view(1, 1, -1)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    def register(self, module) -> None:
        self.handle = module.register_forward_hook(self)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()


@torch.inference_mode()
def generate_batch(tokenizer, model, rows: List[Dict[str, Any]], layer: int, params, scale: float, max_new_tokens: int):
    prompts = [str(row["prompt"]) for row in rows]
    enc, spans = prefill_for_generation(tokenizer, model, prompts)
    target_module = model.model.layers[layer - 1]
    hook = DynamicSteeringHook(params["u"], params["beta"], spans, scale)
    hook.register(target_module)
    try:
        out_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    finally:
        hook.remove()

    # Keep the same decoding convention as the existing ISP sweep script for comparability.
    input_lens = enc["attention_mask"].sum(dim=1).tolist()
    return [
        tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
        for seq, input_len in zip(out_ids, input_lens)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-input", default="dataset.json")
    parser.add_argument("--calibration-size", type=int, default=200)
    parser.add_argument("--layer", type=int, required=True, help="1-based hidden-state layer id.")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--calibration-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    rows = load_jsonl_or_json(Path(args.input))
    calibration_rows = load_jsonl_or_json(Path(args.calibration_input))[: args.calibration_size]
    calibration_prompts = [str(row["prompt"]) for row in calibration_rows]

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

    print(f"[ISP] layer={args.layer} scale={args.scale} calibration={len(calibration_prompts)}")
    params = compute_steering_params(
        tokenizer,
        model,
        calibration_prompts,
        args.layer,
        args.calibration_batch_size,
    )
    print(f"[ISP] beta={params['beta'].item():.4f}")

    print(f"[Run] total={len(rows)} batch={args.batch_size}")
    done = 0
    with output_path.open("w", encoding="utf-8") as f:
        for batch_rows in chunked(rows, args.batch_size):
            responses = generate_batch(
                tokenizer,
                model,
                batch_rows,
                args.layer,
                params,
                args.scale,
                args.max_new_tokens,
            )
            for row, response in zip(batch_rows, responses):
                obj = dict(row)
                obj.update(
                    {
                        "method": "contrastive_steering_dynamic",
                        "model_response": response,
                        "hs_layer_id": args.layer,
                        "delta_scale": args.scale,
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False,
                    }
                )
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            done += len(batch_rows)
            if done % (args.batch_size * 10) == 0:
                print(f"[Progress] {done}/{len(rows)}")

    print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
