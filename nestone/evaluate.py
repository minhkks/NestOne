"""Score every exit of a checkpoint on the benchmark, and cache gate inputs.

    python -m nestone.evaluate --ckpt runs/nestone/checkpoint \
        --bundle Intern-Decision/benchmarks/accuracy-v1 --calib data/train/calib.jsonl --out runs/nestone/eval

* Accuracy (%) and Brier score per suite and exit, macro-averaged over suites (the benchmark
  convention: every suite counts equally). Probabilities use the per-exit temperatures from
  calibration.json when present.
* With --cache (default on), also writes ``bench.{json,npz}`` and ``calib.{json,npz}``: per-decision
  option logits of every exit and the gate input z_e (state after layer e, before the coda) at
  every exit except the last. These caches are all ``nestone.gate`` needs.
* --exits overrides the checkpoint's exits (any layers below the coda can be read out; layers
  that were not trained as exits will usually be much less accurate).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .collect import Collected, collect, softmax
from .data import Encoder, encode_rows, read_benchmark, read_rows
from .model import LayerRunner, load_model


def temperatures(ckpt, exits) -> dict[int, float]:
    f = Path(ckpt) / "calibration.json"
    t = json.loads(f.read_text())["temperatures"] if f.exists() else {}
    return {e: float(t.get(str(e), 1.0)) for e in exits}


def summarize(c: Collected, T: dict[int, float]) -> dict:
    suites = list(dict.fromkeys(c.suite))
    su, gold = np.array(c.suite), np.array(c.gold)
    res = {}
    for e in c.exits:
        pred = np.array([int(np.argmax(z)) for z in c.logits[e]])
        bri = np.array([((softmax(z, T[e]) - np.eye(len(z))[g]) ** 2).sum() for z, g in zip(c.logits[e], gold)])
        acc = {s: 100 * float((pred[su == s] == gold[su == s]).mean()) for s in suites}
        res[e] = {"acc": acc, "macro": float(np.mean(list(acc.values()))),
                  "brier_macro": float(np.mean([bri[su == s].mean() for s in suites])),
                  "layers": e + 3, "temperature": T[e]}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundle", required=True, help="Intern-Decision benchmarks/accuracy-v1 directory")
    ap.add_argument("--calib", help="calib.jsonl, cached too so the gate can train on it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exits", help="comma-separated exit layers (default: the checkpoint's)")
    ap.add_argument("--suites", help="comma-separated subset of suites")
    ap.add_argument("--no-cache", action="store_true", help="only write the summary")
    ap.add_argument("--micro-tokens", type=int, default=16384)
    a = ap.parse_args()

    model, tok = load_model(a.ckpt, dtype=torch.bfloat16)
    runner = LayerRunner(model)
    cfg = Path(a.ckpt) / "nestone_config.json"
    exits = ([int(x) for x in a.exits.split(",")] if a.exits else json.loads(cfg.read_text())["exits"])
    exits = sorted(exits)
    T = temperatures(a.ckpt, exits)
    enc = Encoder(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    state_exits = [] if a.no_cache else exits[:-1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    examples = []
    for suite, rows in read_benchmark(a.bundle, a.suites.split(",") if a.suites else None).items():
        examples += encode_rows(enc, rows, suite=suite)[0]
    bench = collect(runner, examples, exits, pad, a.micro_tokens, state_exits)
    if not a.no_cache:
        bench.save(out / "bench")
        if a.calib:
            cal = collect(runner, encode_rows(enc, read_rows(a.calib), suite="calib")[0], exits, pad,
                          a.micro_tokens, state_exits)
            cal.save(out / "calib")
    res = summarize(bench, T)
    (out / "summary.json").write_text(json.dumps({str(e): r for e, r in res.items()}, indent=2))
    suites = list(next(iter(res.values()))["acc"])
    print("| Exit | Layers | " + " | ".join(suites) + " | Average | Brier |")
    print("|" + "---|" * (len(suites) + 4))
    for e, r in res.items():
        print(f"| {e} | {r['layers']} | " + " | ".join(f"{r['acc'][s]:.1f}" for s in suites)
              + f" | **{r['macro']:.2f}** | {r['brier_macro']:.3f} |")


if __name__ == "__main__":
    main()
