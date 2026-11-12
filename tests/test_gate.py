"""CPU tests of the stop-gate logic on synthetic data (no model needed)."""
import numpy as np

from nestone.collect import Collected
from nestone.gate import Data, Projection, Scorer, choose_tau, fit_exit_gate, run_policy, stratified_split


def synthetic(n=600, d=32, seed=0):
    rng = np.random.default_rng(seed)
    exits = [9, 13, 21]
    suite = np.array(["easy"] * (n // 2) + ["hard"] * (n - n // 2))
    gold = rng.integers(0, 4, n)
    c = Collected(exits, [f"r{i}/f" for i in range(n)], list(suite), gold.tolist(), {}, {})
    for e, p_ok in zip(exits, (0.6, 0.75, 0.9)):
        ok = rng.random(n) < np.where(suite == "easy", p_ok + 0.08, p_ok - 0.2)
        logits = []
        for g, o in zip(gold, ok):
            z = rng.normal(0, 1, 4)
            z[g if o else (g + 1) % 4] += 3
            logits.append(z.astype(np.float32))
        c.logits[e] = logits
        if e != exits[-1]:   # state carries a noisy copy of correctness
            c.states[e] = (rng.normal(0, 1, (n, d)) + ok[:, None] * 0.8).astype(np.float16)
    return c, exits


def test_projection_roundtrip():
    X = np.random.default_rng(1).normal(size=(200, 20))
    p = Projection.fit(X, 8)
    assert p(X).shape == (200, 8)
    assert np.allclose(p(X).std(0), 1, atol=1e-6)


def test_gate_learns_and_policy_costs():
    c, exits = synthetic()
    d = Data([c], exits, {})
    g = fit_exit_gate(d.state[9], d.correct[9], pca=8, l2=1e-2)
    s = g(d.state[9])
    assert s[d.correct[9] > 0].mean() > s[d.correct[9] == 0].mean()
    idx = np.arange(len(d))
    stop, lay, ok = run_policy(d, exits, {9: s, 13: np.zeros(len(d))}, 0.5, idx, paid=False)
    assert set(np.unique(stop)) <= set(exits)
    assert np.all(lay[stop == 9] == 12) and np.all(lay[stop == 21] == 24)   # no readout cost


def test_tau_rule_respects_delta():
    c, exits = synthetic()
    d = Data([c], exits, {})
    idx = np.arange(len(d))
    _, inner = stratified_split(d.suite)
    oof = Scorer(d, None, "state", 8, 1e-2).oof(exits, idx, inner)
    tau = choose_tau(d, exits, oof, idx, delta=1.0, paid=False)
    assert 0.3 <= tau <= 1.01
