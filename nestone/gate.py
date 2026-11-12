"""The stop gate: one logistic regression per exit on the hidden state, deciding stop / continue.

At exit e the gate reads z_e, the residual stream right after layer e at the readout position
(before the coda), and predicts whether exit e answers correctly:

    g_e(x) = sigmoid(w_e . phi_e(z_e) + b_e),   phi_e = standardise -> PCA (128) -> unit variance

Labels c_e = 1[argmax p_e = y] with class-balanced weights. One threshold tau is shared by all
exits; at inference a decision stops at the first exit with g_e >= tau, and the last exit always
answers. Passing an exit costs nothing (the gate reads the state before the readout), so a
decision that stops at exit e costs e + 1 + 2 layers (prefix + coda).

tau rule: the most aggressive value whose macro accuracy stays within --delta points of always
answering at the last exit, measured on out-of-sample gate scores (inner 2-fold split).

Two commands, both on caches written by ``nestone.evaluate``:

  fit       train a deployable gate on labelled data from your traffic (plus optional extra
            training rows such as the calibration split) and save it for ``nestone.decide``.
            python -m nestone.gate fit --fit eval/bench --extra eval/calib --exits 9,13,21,29 \
                --delta 0.25 --out runs/nestone/gate.npz
  crossfit  the paper's evaluation: suite-stratified 2-fold cross-fitting on the benchmark,
            with the confidence baselines (Table 4 of the paper).
            python -m nestone.gate crossfit --bench eval/bench --extra eval/calib --exits 9,13,21
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .collect import Collected, softmax

TAUS = np.round(np.concatenate([np.arange(0.30, 0.90, 0.05), np.arange(0.90, 0.995, 0.01)]), 3)
FEATURES = ("state", "state+conf", "conf", "maxprob")   # only "state" needs no readout per check


# ----------------------------------------------------------------------------- model pieces
@dataclass
class Projection:
    mu: np.ndarray
    sd: np.ndarray
    P: np.ndarray | None     # [D, k] PCA basis (None = standardise only)
    ps: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, k: int | None):
        X = X.astype(np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs = (X - mu) / sd
        P = None
        if k:
            P = np.linalg.svd(Xs, full_matrices=False)[2][:k].T
            Xs = Xs @ P
        return cls(mu, sd, P, Xs.std(0) + 1e-6)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Xs = (X.astype(np.float64) - self.mu) / self.sd
        if self.P is not None:
            Xs = Xs @ self.P
        return Xs / self.ps


def logistic(X: np.ndarray, y: np.ndarray, sw: np.ndarray, l2: float = 1e-2, iters: int = 200):
    """Weighted, L2-regularised logistic regression fitted with L-BFGS. Returns (w, b)."""
    import torch
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    st = torch.tensor(sw / sw.mean(), dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt, weight=st) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def balanced_weights(y: np.ndarray) -> np.ndarray:
    pos = y.mean()
    return np.where(y > 0, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))


def conf_features(logits: list[np.ndarray], T: float) -> np.ndarray:
    """Calibrated top probability, top-2 margin and entropy of an exit's distribution."""
    out = []
    for z in logits:
        p = softmax(z, T)
        s = np.sort(p)[::-1]
        out.append([s[0], s[0] - (s[1] if len(s) > 1 else 0.0), -(p * np.log(p + 1e-12)).sum()])
    return np.array(out)


@dataclass
class ExitGate:
    proj: Projection
    w: np.ndarray
    b: float

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.proj(X) @ self.w + self.b)))


def fit_exit_gate(X: np.ndarray, y: np.ndarray, pca: int | None, l2: float) -> ExitGate:
    proj = Projection.fit(X, pca)
    w, b = logistic(proj(X), y, balanced_weights(y), l2)
    return ExitGate(proj, w, b)


# ----------------------------------------------------------------------------- data + policy
class Data:
    """Per-decision arrays from one or more caches (concatenated)."""

    def __init__(self, caches: list[Collected], exits: list[int], T: dict[int, float]):
        self.exits = exits
        self.suite = np.array([s for c in caches for s in c.suite])
        self.gold = np.array([g for c in caches for g in c.gold])
        self.pred = {e: np.array([int(np.argmax(z)) for c in caches for z in c.logits[e]]) for e in exits}
        self.correct = {e: (self.pred[e] == self.gold).astype(float) for e in exits}
        self.conf = {e: conf_features([z for c in caches for z in c.logits[e]], T.get(e, 1.0)) for e in exits}
        self.state = {}
        for e in exits[:-1]:
            if all(e in c.states for c in caches):
                self.state[e] = np.concatenate([c.states[e] for c in caches]).astype(np.float32)

    def __len__(self):
        return len(self.gold)

    def features(self, e: int, kind: str, idx) -> np.ndarray:
        if kind == "state":
            return self.state[e][idx]
        if kind == "conf":
            return self.conf[e][idx]
        if kind == "state+conf":
            return np.hstack([self.state[e][idx], self.conf[e][idx]])
        raise ValueError(kind)


def macro(x: np.ndarray, suites: np.ndarray) -> float:
    return 100 * float(np.mean([x[suites == s].mean() for s in dict.fromkeys(suites)]))


def run_policy(d: Data, exits: list[int], scores: dict[int, np.ndarray], tau: float, idx: np.ndarray, paid: bool):
    """Walk the exits in order; stop at the first exit with score >= tau (the last always answers).
    Returns (exit used, layers run, correct) per decision. `paid` charges 2 layers per earlier
    readout (rules that need the exit's confidence to decide)."""
    stop = np.full(len(idx), -1)
    lay = np.zeros(len(idx))
    checks = np.zeros(len(idx))
    for e in exits:
        go = stop < 0
        lay[go] = e + 1 + 2 + (2 * checks[go] if paid else 0)
        if e == exits[-1]:
            stop[go] = e
            break
        stop[go & (scores[e][idx] >= tau)] = e
        checks[go] += 1
    ok = np.array([d.correct[e][i] for e, i in zip(stop, idx)])
    return stop, lay, ok


def choose_tau(d: Data, exits, oof: dict[int, np.ndarray], idx: np.ndarray, delta: float, paid: bool) -> float:
    target = macro(d.correct[exits[-1]][idx], d.suite[idx]) - delta
    good = [t for t in TAUS if macro(run_policy(d, exits, oof, t, idx, paid)[2], d.suite[idx]) >= target]
    return float(min(good)) if good else 1.01


def stratified_split(suites: np.ndarray, seed: int = 0):
    """Suite-stratified 2-fold split (fold) and an independent inner split (inner)."""
    rng = np.random.default_rng(seed)
    fold, inner = np.zeros(len(suites), int), np.zeros(len(suites), int)
    for s in sorted(set(suites)):
        ix = np.nonzero(suites == s)[0]
        rng.shuffle(ix)
        fold[ix[: len(ix) // 2]] = 1
        inner[ix[::2]] = 1
    return fold, inner


class Scorer:
    """Fits gates of one feature kind on (training idx [+ extra data]) and scores test idx."""

    def __init__(self, d: Data, extra: Data | None, kind: str, pca: int, l2: float):
        self.d, self.extra, self.kind, self.pca, self.l2 = d, extra, kind, pca, l2

    def fit(self, e, tr) -> ExitGate:
        X, y = self.d.features(e, self.kind, tr), self.d.correct[e][tr]
        if self.extra is not None:
            ai = np.arange(len(self.extra))
            X = np.vstack([X, self.extra.features(e, self.kind, ai)])
            y = np.concatenate([y, self.extra.correct[e]])
        return fit_exit_gate(X, y, None if self.kind == "conf" else self.pca, self.l2)

    def scores(self, exits, tr, te) -> dict[int, np.ndarray]:
        out = {}
        for e in exits[:-1]:
            if self.kind == "maxprob":
                out[e] = np.zeros(len(self.d))
                out[e][te] = self.d.conf[e][te, 0]
            else:
                out[e] = np.zeros(len(self.d))
                out[e][te] = self.fit(e, tr)(self.d.features(e, self.kind, te))
        return out

    def oof(self, exits, idx, inner) -> dict[int, np.ndarray]:
        """Out-of-sample scores on idx from a 2-fold split by `inner`."""
        out = {e: np.zeros(len(self.d)) for e in exits[:-1]}
        for j in (0, 1):
            s = self.scores(exits, idx[inner[idx] != j], idx[inner[idx] == j])
            for e in out:
                out[e][idx[inner[idx] == j]] = s[e][idx[inner[idx] == j]]
        return out


# ----------------------------------------------------------------------------- commands
def load_T(path) -> dict[int, float]:
    if not path:
        return {}
    return {int(k): float(v) for k, v in json.loads(Path(path).read_text())["temperatures"].items()}


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def cmd_fit(a):
    exits = sorted(int(x) for x in a.exits.split(","))
    T = load_T(a.calibration)
    d = Data([Collected.load(p) for p in a.fit], exits, T)
    extra = Data([Collected.load(p) for p in a.extra], exits, T) if a.extra else None
    sc = Scorer(d, extra, "state", a.pca, a.l2)
    idx = np.arange(len(d))
    _, inner = stratified_split(d.suite, a.seed)
    tau = choose_tau(d, exits, sc.oof(exits, idx, inner), idx, a.delta, paid=False)
    gates = {e: sc.fit(e, idx) for e in exits[:-1]}                 # refit on all labelled data
    arrays = {}
    for e, g in gates.items():
        arrays.update({f"{e}_mu": g.proj.mu, f"{e}_sd": g.proj.sd, f"{e}_P": g.proj.P,
                       f"{e}_ps": g.proj.ps, f"{e}_w": g.w, f"{e}_b": np.array([g.b])})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    meta = {"exits": exits, "tau": tau, "delta": a.delta, "pca": a.pca, "n_fit": len(d),
            "n_extra": len(extra) if extra else 0}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    stop, lay, ok = run_policy(d, exits, {e: g(d.state[e]) for e, g in gates.items()}, tau, idx, False)
    print(json.dumps({**meta, "in_sample_macro": macro(ok, d.suite), "last_exit_macro": macro(d.correct[exits[-1]], d.suite),
                      "in_sample_layers": float(lay.mean())}, indent=2))


def cmd_crossfit(a):
    exits = sorted(int(x) for x in a.exits.split(","))
    T = load_T(a.calibration)
    d = Data([Collected.load(a.bench)], exits, T)
    extra = Data([Collected.load(p) for p in a.extra], exits, T) if a.extra else None
    fold, inner = stratified_split(d.suite, a.seed)
    S = list(dict.fromkeys(d.suite))
    last = exits[-1]
    print(f"exits {exits}; static accuracy:", {e: round(macro(d.correct[e], d.suite), 2) for e in exits})
    rows = []
    for kind in a.features:
        sc = Scorer(d, extra if kind != "maxprob" else None, kind, a.pca, a.l2)
        paid = kind != "state"
        test = {e: np.zeros(len(d)) for e in exits[:-1]}
        inner_oof = {}
        for k in (0, 1):
            tr, te = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
            s = sc.scores(exits, tr, te)
            for e in test:
                test[e][te] = s[e][te]
            inner_oof[k] = sc.oof(exits, tr, inner)
        au = {e: auroc(test[e], d.correct[e]) for e in exits[:-1]}
        for delta in a.deltas:
            stop, lay, ok, taus = np.zeros(len(d), int), np.zeros(len(d)), np.zeros(len(d)), []
            for k in (0, 1):
                tr, te = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
                t = choose_tau(d, exits, inner_oof[k], tr, delta, paid)
                taus.append(t)
                stop[te], lay[te], ok[te] = run_policy(d, exits, test, t, te, paid)
            r = {"features": kind, "delta": delta, "taus": "/".join(f"{t:.2f}" for t in taus),
                 "macro": macro(ok, d.suite), "delta_macro": macro(ok, d.suite) - macro(d.correct[last], d.suite),
                 "layers": float(np.mean([lay[d.suite == s].mean() for s in S])),
                 "auroc": {str(e): round(v, 3) for e, v in au.items()},
                 "stop_share": {str(e): float((stop == e).mean()) for e in exits}}
            for s in S:
                m = d.suite == s
                r[f"acc|{s}"] = 100 * float(ok[m].mean())
                r[f"dacc|{s}"] = 100 * float(ok[m].mean() - d.correct[last][m].mean())
                r[f"layers|{s}"] = float(lay[m].mean())
            rows.append(r)
            print(f"{kind:11s} delta {delta:<5} macro {r['macro']:.2f} ({r['delta_macro']:+.2f}) "
                  f"layers {r['layers']:.1f} taus {r['taus']} | "
                  + " ".join(f"{s}:{r[f'dacc|{s}']:+.1f}@{r[f'layers|{s}']:.1f}" for s in S), flush=True)
        if kind != "maxprob":
            print(f"{'':11s} AUROC per exit {au}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "crossfit"):
        p = sub.add_parser(name)
        p.add_argument("--exits", required=True, help="exit layers in order; the last one always answers")
        p.add_argument("--extra", nargs="*", default=[], help="extra caches used only to train the gates (e.g. calib)")
        p.add_argument("--calibration", help="calibration.json (temperatures for confidence features)")
        p.add_argument("--pca", type=int, default=128)
        p.add_argument("--l2", type=float, default=1e-2)
        p.add_argument("--seed", type=int, default=0)
    f = sub.choices["fit"]
    f.add_argument("--fit", nargs="+", required=True, help="labelled caches from the target traffic")
    f.add_argument("--delta", type=float, default=0.25)
    f.add_argument("--out", required=True, help="gate file (.npz; settings go to the matching .json)")
    c = sub.choices["crossfit"]
    c.add_argument("--bench", required=True)
    c.add_argument("--features", nargs="+", default=["state", "state+conf", "conf", "maxprob"], choices=FEATURES)
    c.add_argument("--deltas", type=float, nargs="+", default=[0.25, 2.0])
    c.add_argument("--out", help="write all results as JSON")
    a = ap.parse_args()
    cmd_fit(a) if a.cmd == "fit" else cmd_crossfit(a)


if __name__ == "__main__":
    main()
