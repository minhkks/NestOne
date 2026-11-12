"""Backbone loading and layer-by-layer execution with prefix exits and a shared coda.

The decision model has layers f_0 .. f_{n-1}. The *coda* is the last two layers plus the final
norm and the (tied) LM head restricted to the option symbols. An *exit after layer e* runs the
prefix f_0 .. f_e, then the coda:  p_e = softmax(C(h^(e))_O / T_e). The full model is the exit
after layer n-3. Because every sub-model is a prefix of the next, the states of a shallow exit
are reused when a decision continues to a deeper one.

:class:`LayerRunner` runs any sequence of decoder layers on saved full-sequence states, using
the model's own embedding, attention masks and rotary embeddings, so it works for the hybrid
(linear + full attention) Qwen3.5 stack without modifying it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .schema import DECISION_TOKEN


def find_decoder(model: nn.Module) -> nn.Module:
    """The text decoder module owning ``embed_tokens`` / ``layers`` / ``norm``."""
    for _, m in model.named_modules():
        if isinstance(getattr(m, "layers", None), nn.ModuleList) and hasattr(m, "norm") and hasattr(m, "embed_tokens"):
            return m
    raise RuntimeError("no decoder with embed_tokens / layers / norm found")


def load_model(path: str, dtype=torch.bfloat16, device: str = "cuda", add_marker: bool = True):
    """Load a Qwen3.5 checkpoint (base or trained) and its tokenizer with the ``<decision>`` marker."""
    from transformers import AutoTokenizer
    try:
        from transformers import Qwen3_5ForConditionalGeneration as Cls
    except ImportError:  # pragma: no cover - other causal LMs
        from transformers import AutoModelForCausalLM as Cls
    tok = AutoTokenizer.from_pretrained(path)
    if add_marker and DECISION_TOKEN not in tok.get_added_vocab():
        tok.add_special_tokens({"additional_special_tokens": [DECISION_TOKEN]})
    model = Cls.from_pretrained(path, dtype=dtype)
    if tok.convert_tokens_to_ids(DECISION_TOKEN) >= model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))
    return model.to(device), tok


class _Capture(nn.Module):
    """Stand-in layer that records the hidden state and the per-layer-type kwargs."""

    def __init__(self, layer_type: str, store: dict):
        super().__init__()
        self.layer_type = self.block_type = layer_type
        self.store = store

    def forward(self, hidden_states, **kw):
        self.store.setdefault("h0", hidden_states)
        self.store[self.layer_type] = kw
        return hidden_states


@dataclass
class State:
    h: torch.Tensor          # [B, L, D] residual stream after `layer` (full sequence)
    layer: int               # last layer applied (-1 = embeddings)
    kw: dict                 # layer-type -> forward kwargs (masks, rotary embeddings)


class LayerRunner:
    def __init__(self, model: nn.Module, n_coda: int = 2):
        self.model = model
        self.dec = find_decoder(model)
        self.layers = self.dec.layers
        self.n = len(self.layers)
        cfg = self.dec.config
        self.types = list(getattr(cfg, "layer_types", None) or ["full_attention"] * self.n)
        self.coda = list(range(self.n - n_coda, self.n))
        self.head = model.get_output_embeddings()

    @property
    def full_exit(self) -> int:
        """The exit that equals the full model (last layer before the coda)."""
        return self.coda[0] - 1

    def start(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> State:
        """Embeddings plus the attention masks / rotary embeddings each layer type needs."""
        store, cfg = {}, self.dec.config
        kinds = sorted(set(self.types))
        orig = (self.dec.layers, cfg.num_hidden_layers, getattr(cfg, "layer_types", None))
        try:
            self.dec.layers = nn.ModuleList([_Capture(t, store) for t in kinds])
            cfg.num_hidden_layers = len(kinds)
            if orig[2] is not None:
                cfg.layer_types = kinds
            self.dec(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            self.dec.layers, cfg.num_hidden_layers = orig[0], orig[1]
            if orig[2] is not None:
                cfg.layer_types = orig[2]
        return State(store["h0"], -1, {t: store[t] for t in kinds})

    def run(self, st: State, layer_ids) -> State:
        h = st.h
        for i in layer_ids:
            out = self.layers[i](h, **st.kw[self.types[i]])
            h = out[0] if isinstance(out, tuple) else out
        return State(h, layer_ids[-1] if len(layer_ids) else st.layer, st.kw)

    def advance(self, st: State, to: int) -> State:
        """Continue the prefix from the saved state up to and including layer `to`."""
        return self.run(st, list(range(st.layer + 1, to + 1)))

    def coda_logits(self, st: State, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits at the readout positions after running the coda on `st`."""
        if st.layer >= self.coda[0] and st.layer != self.n - 1:
            raise ValueError(f"cannot exit inside the coda (state after layer {st.layer})")
        h = st.h if st.layer == self.n - 1 else self.run(st, self.coda).h
        h = self.dec.norm(h[rows, cols])
        return self.head(h.to(self.head.weight.dtype)).float()

    def exit_logits(self, ids, mask, rows, cols, exit_layer: int) -> torch.Tensor:
        """One pass of the sub-model that exits after `exit_layer` (prefix + coda)."""
        return self.coda_logits(self.advance(self.start(ids, mask), exit_layer), rows, cols)


def option_logits(vocab_logits: torch.Tensor, syms: list[list[int]]) -> list[torch.Tensor]:
    """Restrict each decision's vocabulary logits to its option symbols."""
    return [z[s] for z, s in zip(vocab_logits, syms)]
