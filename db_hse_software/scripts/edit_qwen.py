# -*- coding: utf-8 -*-
"""
edit_qwen.py
Model-agnostic hidden-state editing wrapper for decoder-only HF CausalLMs (Qwen2/Qwen3, etc.)

- No dependency on internal docstring constants.
- Works via forward hooks on decoder blocks.
- layer_idx in edit_method refers to decoder block index (0-based).
- Default behavior: prompt-only editing (skip seq_len==1 decoding steps).
"""

import torch
from torch import nn
from typing import Any, List, Optional, Sequence
from transformers import AutoModelForCausalLM, PreTrainedModel

from hs_method import HiddenStatesEditingMethod


def _get_attr_by_path(obj: Any, path: Sequence[str]) -> Any:
    cur = obj
    for p in path:
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


def _find_decoder_blocks(model: Any) -> List[nn.Module]:
    """
    Try common paths to locate decoder blocks.
    Qwen2/Qwen3 typically: base_model.model.layers
    """
    candidates = [
        ("model", "layers"),           # Qwen2/Qwen3: ForCausalLM.model.layers
        ("transformer", "h"),          # GPT2-style
        ("gpt_neox", "layers"),        # NeoX-style
        ("layers",),                   # fallback
    ]
    for path in candidates:
        blocks = _get_attr_by_path(model, path)
        if blocks is None:
            continue
        if isinstance(blocks, (nn.ModuleList, list, tuple)) and len(blocks) > 0 and isinstance(blocks[0], nn.Module):
            return list(blocks)
    raise RuntimeError(
        "Cannot locate decoder blocks automatically. Tried paths: "
        + ", ".join([".".join(p) for p in candidates])
    )


def _replace_tuple_first(x, new_first):
    if not isinstance(x, tuple):
        return x
    if len(x) == 0:
        return (new_first,)
    return (new_first,) + tuple(x[1:])


class EditCausalLM(nn.Module):
    def __init__(self, base_model: PreTrainedModel):
        super().__init__()
        # 作为子模块注册到 _modules，必须依赖 nn.Module.__getattr__ 取回
        self.base_model = base_model

        self.blocks = _find_decoder_blocks(self.base_model)
        self.num_layers = len(self.blocks)

        # hook 读取的运行时上下文
        self._active_edit_method: Optional[HiddenStatesEditingMethod] = None
        self._prompt_only: bool = True

        self._hooks = []
        self._register_hooks()

    # ✅ 关键：先调用 nn.Module.__getattr__（找子模块/参数），找不到再委托给 base_model
    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)  # nn.Module 的查找机制（_modules/_parameters/_buffers）
        except AttributeError:
            base = super().__getattr__("base_model")  # 避免递归
            return getattr(base, name)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.base_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.base_model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _register_hooks(self):
        for layer_idx, block in enumerate(self.blocks):
            h = block.register_forward_hook(self._make_block_hook(layer_idx))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def _make_block_hook(self, layer_idx: int):
        def hook(module: nn.Module, inputs, output):
            em = self._active_edit_method
            if em is None or (not getattr(em, "if_edit", False)):
                return output
            if em.edit_layer_idx is None or layer_idx not in em.edit_layer_idx:
                return output

            # output 通常是 Tensor 或 tuple(Tensor, ...)
            if isinstance(output, tuple):
                hs = output[0]
                if not torch.is_tensor(hs):
                    return output
                # prompt-only：跳过 seq_len==1 的增量生成步
                if self._prompt_only and hs.dim() >= 2 and hs.shape[1] == 1:
                    return output
                new_hs = em.edit(hidden_states=hs, layer_idx=layer_idx)
                return _replace_tuple_first(output, new_hs)

            if torch.is_tensor(output):
                hs = output
                if self._prompt_only and hs.dim() >= 2 and hs.shape[1] == 1:
                    return output
                return em.edit(hidden_states=hs, layer_idx=layer_idx)

            return output

        return hook

    def forward(self, *args, **kwargs):
        edit_method: Optional[HiddenStatesEditingMethod] = kwargs.pop("edit_method", None)

        # 默认 prompt-only；你也可以在 edit_method 上挂 prompt_only=False 来强制每步都注入
        prompt_only = True
        if edit_method is not None and hasattr(edit_method, "prompt_only"):
            prompt_only = bool(getattr(edit_method, "prompt_only"))

        self._active_edit_method = edit_method
        self._prompt_only = prompt_only
        try:
            return self.base_model(*args, **kwargs)
        finally:
            self._active_edit_method = None
            self._prompt_only = True

    def generate(self, *args, **kwargs):
        edit_method: Optional[HiddenStatesEditingMethod] = kwargs.pop("edit_method", None)

        prompt_only = True
        if edit_method is not None and hasattr(edit_method, "prompt_only"):
            prompt_only = bool(getattr(edit_method, "prompt_only"))

        self._active_edit_method = edit_method
        self._prompt_only = prompt_only
        try:
            return self.base_model.generate(*args, **kwargs)
        finally:
            self._active_edit_method = None
            self._prompt_only = True

    @classmethod
    def from_pretrained(cls, model_path: str, **hf_kwargs) -> "EditCausalLM":
        base = AutoModelForCausalLM.from_pretrained(model_path, **hf_kwargs)
        base.eval()
        return cls(base)


# 兼容你旧脚本的 import 名
EditQwen2ForCausalLM = EditCausalLM
EditQwen3ForCausalLM = EditCausalLM
