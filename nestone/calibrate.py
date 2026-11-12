"""Fit one temperature per exit on a held-out calibration split (minimise NLL, T in [0.01, 100]).

    python -m nestone.calibrate --ckpt runs/nestone/checkpoint --data data/train

Writes <ckpt>/calibration.json  {"temperatures": {"9": T_9, "13": T_13, ...}, ...}.
The temperature does not change any argmax; it only calibrates the probabilities that the
stop gate's confidence baselines and the Brier score use.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .collect import collect
from .data import Encoder, encode_rows, read_rows
from .model import LayerRunner, load_model


def nll(logits, gold, T):
    return float(np.mean([np.logaddexp.reduce(np.asarray(z, np.float64) / T) - z[g] / T for z, g in zip(logits, gold)]))


def fit_temperature(logits, gold) -> float:
    lo, hi = math.log(0.01), math.log(100.0)
    for _ in range(80):           # golden-section search on log T
        m1, m2 = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        if nll(logits, gold, math.exp(m1)) < nll(logits, gold, math.exp(m2)):
            hi = m2
        else:
            lo = m1
    return math.exp((lo + hi) / 2)


def load_exits(ckpt) -> list[int]:
    return json.loads((Path(ckpt) / "nestone_config.json").read_text())["exits"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="directory with calib.jsonl")
    ap.add_argument("--micro-tokens", type=int, default=16384)
    a = ap.parse_args()
    model, tok = load_model(a.ckpt, dtype=torch.bfloat16)
    runner = LayerRunner(model)
    exits = load_exits(a.ckpt)
    ex, _ = encode_rows(Encoder(tok), read_rows(Path(a.data) / "calib.jsonl"))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    c = collect(runner, ex, exits, pad, a.micro_tokens)
    res = {"temperatures": {}, "calib_acc": {}, "calib_decisions": len(c.gold)}
    for e in exits:
        T = fit_temperature(c.logits[e], c.gold)
        res["temperatures"][str(e)] = T
        res["calib_acc"][str(e)] = float(np.mean([int(np.argmax(z) == g) for z, g in zip(c.logits[e], c.gold)]))
    (Path(a.ckpt) / "calibration.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
