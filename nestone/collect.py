"""Run every exit of a checkpoint over a set of rows in one shared prefix pass per batch.

For each decision it returns the option logits of every exit and, optionally, the gate input
z_e: the residual stream right after layer e at the decision's readout position, i.e. the
state *before* the coda (what the stop gate reads at inference).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import collate, micro_batches
from .model import LayerRunner, option_logits


@dataclass
class Collected:
    exits: list[int]
    keys: list[str] = field(default_factory=list)
    suite: list[str] = field(default_factory=list)
    gold: list[int] = field(default_factory=list)
    logits: dict[int, list[np.ndarray]] = field(default_factory=dict)   # exit -> per-decision option logits
    states: dict[int, np.ndarray] = field(default_factory=dict)         # exit -> [N, D] float16

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), **{f"z{e}": s for e, s in self.states.items()})
        json.dump({"exits": self.exits, "keys": self.keys, "suite": self.suite, "gold": self.gold,
                   "logits": {str(e): [z.tolist() for z in v] for e, v in self.logits.items()}},
                  open(path.with_suffix(".json"), "w"))

    @classmethod
    def load(cls, path: str | Path) -> "Collected":
        path = Path(path)
        j = json.load(open(path.with_suffix(".json")))
        c = cls(j["exits"], j["keys"], j["suite"], j["gold"],
                {int(e): [np.asarray(z, np.float32) for z in v] for e, v in j["logits"].items()})
        npz = path.with_suffix(".npz")
        if npz.exists():
            z = np.load(npz)
            c.states = {int(k[1:]): z[k] for k in z.files}
        return c


@torch.no_grad()
def collect(runner: LayerRunner, examples, exits: list[int], pad_id: int, max_tokens: int = 16384,
            state_exits: list[int] | None = None, device: str = "cuda") -> Collected:
    exits = sorted(exits)
    state_exits = sorted(state_exits or [])
    out = Collected(exits, logits={e: [] for e in exits})
    states = {e: [] for e in state_exits}
    runner.model.eval()
    for b in micro_batches(examples, max_tokens):
        batch = collate(examples, b, pad_id, device)
        for i in b:
            out.keys += examples[i].keys
            out.suite += [examples[i].suite] * len(examples[i].keys)
            out.gold += examples[i].gold
        st = runner.start(batch.ids, batch.mask)
        for e in sorted(set(exits) | set(state_exits)):
            st = runner.advance(st, e)
            if e in states:
                states[e].append(st.h[batch.rows, batch.cols].float().cpu().numpy().astype(np.float16))
            if e in out.logits:
                z = option_logits(runner.coda_logits(st, batch.rows, batch.cols), batch.syms)
                out.logits[e] += [zj.float().cpu().numpy() for zj in z]
    out.states = {e: np.concatenate(v) for e, v in states.items()}
    return out


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, np.float64) / T
    p = np.exp(z - z.max())
    return p / p.sum()
