"""Nested prefix training (NestOne, no external teacher).

    python -m nestone.train --base Qwen/Qwen3.5-4B --data data/train --out runs/nestone

Objective per global step (Section 4.2 of the paper)::

    L = CE(p_32, y) + sum_{k in K_t} [ (1 - a_t) CE(p_k, y) + a_t KL(sg(p_32) || p_k) ]

* p_32 is the full model; p_k is the sub-model that exits after layer e_k (prefix + shared coda).
* CE is over the full vocabulary at the readout position (target = the gold option symbol);
  KL is over the decision's option symbols; sg = stop-gradient.
* K_t: each sub-model is included independently with its probability (default: exit 21 with 0.5,
  exits 13 and 9 always), drawn once per global step.
* a_t ramps linearly from 0 to --alpha-max over the first --alpha-ramp fraction of training.
* Recipe: full fine-tune of the language model (vision tower frozen), AdamW lr 2e-6, weight
  decay 0.01, cosine to 2e-7 with 3% warm-up, 2 epochs, ~131k tokens per step, bf16 autocast
  with fp32 master weights, gradient checkpointing.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import Encoder, collate, encode_rows, micro_batches, read_rows
from .model import LayerRunner, find_decoder, load_model, option_logits


def parse_sub_exits(spec: str) -> dict[int, float]:
    """"21:0.5,13:1,9:1" -> {21: 0.5, 13: 1.0, 9: 1.0} (exit layer -> inclusion probability)."""
    return {int(e): float(p) for e, p in (x.split(":") for x in spec.split(","))}


def kl_options(logits_k: list[torch.Tensor], q: list[torch.Tensor]) -> torch.Tensor:
    """sum_j KL(q_j || softmax(z_j)) over each decision's options."""
    tot = logits_k[0].new_zeros(())
    for z, qj in zip(logits_k, q):
        qj = qj.clamp_min(1e-8)
        tot = tot + (qj * (qj.log() - F.log_softmax(z, -1))).sum()
    return tot


def targets(batch) -> torch.Tensor:
    """Vocabulary id of each decision's gold option symbol."""
    return torch.tensor([s[g] for s, g in zip(batch.syms, batch.gold.tolist())], device=batch.gold.device)


@torch.no_grad()
def evaluate(runner, examples, exits, pad, max_tokens, device):
    """Validation accuracy per exit (one shared prefix pass per micro-batch)."""
    runner.model.eval()
    correct, n = {e: 0 for e in exits}, 0
    for b in micro_batches(examples, max_tokens):
        batch = collate(examples, b, pad, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            st = runner.start(batch.ids, batch.mask)
            for e in sorted(exits):
                st = runner.advance(st, e)
                z = option_logits(runner.coda_logits(st, batch.rows, batch.cols), batch.syms)
                correct[e] += sum(int(zj.argmax() == g) for zj, g in zip(z, batch.gold.tolist()))
        n += len(batch.gold)
    runner.model.train()
    return {e: correct[e] / max(n, 1) for e in exits}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base model path or HF id (e.g. Qwen/Qwen3.5-4B)")
    ap.add_argument("--data", required=True, help="directory with train.jsonl and val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sub-exits", default="21:0.5,13:1,9:1",
                    help="sub-model exits (layer after which they read out) with per-step inclusion probability")
    ap.add_argument("--alpha-max", type=float, default=0.5)
    ap.add_argument("--alpha-ramp", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--lr-min", type=float, default=2e-7)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--tokens-per-step", type=int, default=131072)
    ap.add_argument("--micro-tokens", type=int, default=16384, help="padded tokens per micro-batch")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, help="debug: stop after N optimizer steps")
    ap.add_argument("--limit", type=int, help="debug: use only N training rows")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    log = open(out / "train_log.jsonl", "a")

    model, tok = load_model(a.base, dtype=torch.float32, device=device)
    runner = LayerRunner(model)
    subs = parse_sub_exits(a.sub_exits)
    full = runner.full_exit
    assert all(e < full for e in subs), f"sub-model exits must be below the full exit {full}"
    exits = sorted(subs) + [full]

    # train the language model only (vision tower and projector frozen)
    lm = {id(p) for p in find_decoder(model).parameters()} | {id(model.get_output_embeddings().weight)}
    for p in model.parameters():
        p.requires_grad_(id(p) in lm)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    enc = Encoder(tok)
    rng = random.Random(a.seed)
    train_rows = read_rows(Path(a.data) / "train.jsonl")[: a.limit]
    train, skipped = encode_rows(enc, train_rows, a.max_len)
    val, _ = encode_rows(enc, read_rows(Path(a.data) / "val.jsonl"), a.max_len)
    tok_epoch = sum(len(e.ids) for e in train)
    total = a.max_steps or max(1, round(tok_epoch / a.tokens_per_step)) * a.epochs
    print(f"train {len(train)} rows ({skipped} over-length dropped), {tok_epoch / 1e6:.1f}M tokens/epoch, "
          f"{total} steps; val {len(val)} rows; exits {exits} (sub-models {subs})", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in params if p.ndim >= 2], "weight_decay": a.weight_decay},
                             {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.999), foreach=False)
    warm = max(1, int(a.warmup_ratio * total))

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    step, t0, done = 0, time.time(), False

    def fwd(batch, exit_layer):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return runner.exit_logits(batch.ids, batch.mask, batch.rows, batch.cols, exit_layer)

    for epoch in range(a.epochs):
        batches = micro_batches(train, a.micro_tokens, rng)
        group, ntok = [], 0
        for bi, b in enumerate(batches):
            group.append(b)
            ntok += sum(len(train[i].ids) for i in b)
            if ntok < a.tokens_per_step and bi < len(batches) - 1:
                continue
            n_dec = sum(len(train[i].pos) for g in group for i in g)
            alpha = a.alpha_max * min(1.0, step / max(1.0, a.alpha_ramp * total))
            picks = [e for e in sorted(subs, reverse=True) if rng.random() < subs[e]]
            stats = {"loss": 0.0, "acc": 0.0, **{f"acc{e}": 0.0 for e in picks}}
            for g in group:
                batch = collate(train, g, pad, device)
                tgt = targets(batch)
                logits = fwd(batch, full)                                   # full model
                ce = F.cross_entropy(logits, tgt, reduction="sum")
                (ce / n_dec).backward()
                with torch.no_grad():
                    q = [F.softmax(z.float(), -1) for z in option_logits(logits.detach(), batch.syms)]
                stats["loss"] += ce.item()
                stats["acc"] += (logits.argmax(-1) == tgt).sum().item()
                del logits
                for e in picks:                                             # sampled sub-models
                    lk = fwd(batch, e)
                    loss = (1 - alpha) * F.cross_entropy(lk, tgt, reduction="sum") \
                        + alpha * kl_options(option_logits(lk, batch.syms), q)
                    (loss / n_dec).backward()
                    stats["loss"] += loss.item()
                    stats[f"acc{e}"] += (lk.argmax(-1) == tgt).sum().item()
                    del lk
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": epoch, "lr": lr_at(step - 1), "alpha": alpha, "grad_norm": gn,
                   "decisions": n_dec, "sub_models": picks, "elapsed_min": (time.time() - t0) / 60,
                   **{k: v / n_dec for k, v in stats.items()}}
            if step % a.eval_every == 0 or step == total:
                rec.update({f"val_acc{e}": v for e, v in evaluate(runner, val, exits, pad, a.micro_tokens, device).items()})
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
            group, ntok = [], 0
            if step >= total:
                done = True
                break
        if done:
            break

    model.gradient_checkpointing_disable()
    ck = out / "checkpoint"
    model.to(torch.bfloat16).save_pretrained(ck, safe_serialization=True)
    tok.save_pretrained(ck)
    (ck / "nestone_config.json").write_text(json.dumps(
        {"exits": exits, "full_exit": full, "coda": runner.coda, "base": a.base, "recipe": vars(a)}, indent=2))
    print(f"saved {ck}", flush=True)


if __name__ == "__main__":
    main()
