import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer

from edit_qwen import EditQwen2ForCausalLM
from hs_method import HiddenStatesEditingMethod


POS_PROMPT = "Safe"
NEG_PROMPT = "Unsafe"


@dataclass
class ActivationAddition:
    prompt: str
    coeff: float
    layer: int
    act: torch.Tensor

    def __post_init__(self) -> None:
        assert len(self.act.shape) == 3, f"act must be 3D, got {self.act.shape}"


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


@torch.inference_mode()
def get_actadd_additions(
    model,
    tokenizer,
    prompt1: str,
    prompt2: str,
    coeff: float,
    layer_id: int,
) -> List[ActivationAddition]:
    def get_hidden(text: str) -> torch.Tensor:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model(**inputs, output_hidden_states=True, edit_method=None)
        hidden = out.hidden_states[layer_id]
        return hidden.mean(dim=1, keepdim=True)

    act_pos = get_hidden(prompt1)
    act_neg = get_hidden(prompt2)
    return [
        ActivationAddition(
            prompt=f"{prompt1}-{prompt2}",
            coeff=coeff,
            layer=layer_id,
            act=act_pos - act_neg,
        )
    ]


def bridge_additions_to_edit_method(
    additions: List[ActivationAddition],
    batch_size: int,
    seq_len: int,
    model_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> HiddenStatesEditingMethod:
    edit_tensors: Dict[int, torch.Tensor] = {}
    for add in additions:
        layer_idx = add.layer - 1
        if layer_idx not in edit_tensors:
            edit_tensors[layer_idx] = torch.zeros(
                (batch_size, seq_len, model_dim),
                dtype=dtype,
                device=device,
            )

        vec = add.act.to(device=device, dtype=dtype) * add.coeff
        target_len = min(vec.shape[1], seq_len)
        edit_tensors[layer_idx][:, :target_len, :] += vec.squeeze(0)

    method = HiddenStatesEditingMethod(
        if_edit=True,
        edit_layer_idx=list(edit_tensors.keys()),
        edit_mask=torch.zeros(seq_len, dtype=torch.bool, device=device),
        edit_tensor=edit_tensors,
    )
    method.prompt_only = True
    return method


@torch.inference_mode()
def generate_batch(tokenizer, model, rows: List[Dict[str, Any]], additions, max_new_tokens: int):
    prompts = [str(row["prompt"]) for row in rows]
    texts = [
        apply_chat_template(tokenizer, [{"role": "user", "content": prompt}])
        for prompt in prompts
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    batch_size, seq_len = enc["input_ids"].shape
    edit_method = bridge_additions_to_edit_method(
        additions,
        batch_size,
        seq_len,
        model.config.hidden_size,
        model.device,
        model.dtype,
    )
    out_ids = model.generate(
        **enc,
        edit_method=edit_method,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    responses = [
        tokenizer.decode(seq[seq_len:], skip_special_tokens=True).strip()
        for seq in out_ids
    ]
    return responses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, required=True, help="1-based hidden-state layer id.")
    parser.add_argument("--coeff", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    rows = load_jsonl_or_json(Path(args.input))
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

    print(f"[ActAdd] layer={args.layer} coeff={args.coeff}")
    additions = get_actadd_additions(
        model, tokenizer, POS_PROMPT, NEG_PROMPT, args.coeff, args.layer
    )

    print(f"[Run] total={len(rows)} batch={args.batch_size}")
    done = 0
    with output_path.open("w", encoding="utf-8") as f:
        for batch_rows in chunked(rows, args.batch_size):
            responses = generate_batch(tokenizer, model, batch_rows, additions, args.max_new_tokens)
            for row, response in zip(batch_rows, responses):
                obj = dict(row)
                obj.update(
                    {
                        "method": "actadd",
                        "model_response": response,
                        "actadd_layer": args.layer,
                        "actadd_coeff": args.coeff,
                        "injection_pos": "prefix (index 0)",
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
