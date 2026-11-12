"""Make decisions with a NestOne checkpoint, at a fixed exit or with the stop gate.

    # fixed depth: always answer at the exit after layer 21 (24 layers)
    python -m nestone.decide --ckpt runs/nestone/checkpoint --exit 21 --rows my_rows.jsonl --out preds.jsonl
    # adaptive depth: stop gate at every exit of gate.json (e.g. 9 -> 13 -> 21 -> 29)
    python -m nestone.decide --ckpt runs/nestone/checkpoint --gate runs/nestone/gate.npz --rows my_rows.jsonl

The model runs layer by layer. At each gated exit e, the gate reads the state after layer e
for every decision that is still open; decisions with g_e >= tau run the coda and are answered
at e, the others continue from the saved states (no recomputation). The last exit answers
whatever is left. Each output line reports the chosen option, its calibrated probabilities, the
exit used and the layers run for that decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import schema as S
from .collect import softmax
from .data import Encoder, read_rows
from .evaluate import temperatures
from .gate import ExitGate, Projection
from .model import LayerRunner, load_model, option_logits


def load_gate(path) -> tuple[list[int], float, dict[int, ExitGate]]:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    z = np.load(path.with_suffix(".npz"), allow_pickle=True)
    gates = {}
    for e in meta["exits"][:-1]:
        P = z[f"{e}_P"]
        proj = Projection(z[f"{e}_mu"], z[f"{e}_sd"], None if P.ndim == 0 else P, z[f"{e}_ps"])
        gates[e] = ExitGate(proj, z[f"{e}_w"], float(z[f"{e}_b"][0]))
    return meta["exits"], float(meta["tau"]), gates


class NestOne:
    def __init__(self, ckpt: str, exit: int | None = None, gate: str | None = None, device: str = "cuda"):
        if (exit is None) == (gate is None):
            raise ValueError("choose exactly one of a fixed exit or a gate file")
        self.model, self.tok = load_model(ckpt, dtype=torch.bfloat16, device=device)
        self.model.eval()
        self.runner = LayerRunner(self.model)
        self.enc = Encoder(self.tok)
        self.device = device
        if gate:
            self.exits, self.tau, self.gates = load_gate(gate)
        else:
            self.exits, self.tau, self.gates = [exit], 1.0, {}
        if self.exits[-1] >= self.runner.coda[0]:
            raise ValueError(f"exits must be below the coda (layers {self.runner.coda})")
        self.T = temperatures(ckpt, self.exits)

    @torch.no_grad()
    def decide(self, row: dict) -> list[dict]:
        ex = self.enc(row)
        comp = S.compile_row(row, include_targets=False)
        ids = torch.tensor([ex.ids], device=self.device)
        st = self.runner.start(ids, torch.ones_like(ids))
        open_ = list(range(len(ex.pos)))
        out = [None] * len(ex.pos)
        for e in self.exits:
            st = self.runner.advance(st, e)
            cols = torch.tensor([ex.pos[j] for j in open_], device=self.device)
            if e == self.exits[-1]:
                stop = open_
            else:
                z = st.h[0, cols].float().cpu().numpy()
                g = self.gates[e](z)
                stop = [j for j, gj in zip(open_, g) if gj >= self.tau]
            if stop:
                sc = torch.tensor([ex.pos[j] for j in stop], device=self.device)
                logits = option_logits(self.runner.coda_logits(st, torch.zeros_like(sc), sc), [ex.syms[j] for j in stop])
                for j, zj in zip(stop, logits):
                    p = softmax(zj.float().cpu().numpy(), self.T[e])
                    f = comp.fields[j]
                    k = int(np.argmax(p))
                    values = [v for v, _ in S._options(row["questions"][f])]
                    out[j] = {"key": ex.keys[j], "field": f, "symbol": comp.symbols[f][k], "value": values[k],
                              "probs": p.round(6).tolist(), "exit": e, "layers": e + 1 + len(self.runner.coda)}
            open_ = [j for j in open_ if j not in stop]
            if not open_:
                break
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--exit", type=int, help="fixed exit layer (e.g. 21 = 24-layer model; 29 = full model)")
    g.add_argument("--gate", help="gate file from `nestone.gate fit`")
    ap.add_argument("--rows", required=True, help="JSONL of rows (state + questions)")
    ap.add_argument("--out", help="output JSONL (default: stdout)")
    a = ap.parse_args()
    m = NestOne(a.ckpt, exit=a.exit, gate=a.gate)
    sink = open(a.out, "w") if a.out else None
    layers = []
    for row in read_rows(a.rows):
        for r in m.decide(row):
            layers.append(r["layers"])
            line = json.dumps(r, ensure_ascii=False)
            print(line, file=sink) if sink else print(line)
    if layers:
        print(f"{len(layers)} decisions, mean layers {np.mean(layers):.1f}", flush=True)


if __name__ == "__main__":
    main()
