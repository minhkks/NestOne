# NestOne

Code for **"From Large Language Models to Efficient System One Models via Adaptive Computation"**.

NestOne turns one LLM into a System One decision model that can answer at several depths:

1. **Prefix sub-models with a shared coda.** The sub-model that exits after layer *e* runs layers
   0..*e*, then the model's last two layers and decision head (the *coda*). With Qwen3.5-4B the
   default exits are after layers 9, 13 and 21 (12-, 16- and 24-layer models) plus the full model.
   Each sub-model is a prefix of the next, so a decision that continues to a deeper exit reuses
   the states it already computed.
2. **Nested training without an external teacher.** The full model and sampled sub-models are
   trained jointly in one run:

   `L = CE(p_full, y) + Σ_{k∈K_t} [(1 − α_t)·CE(p_k, y) + α_t·KL(sg(p_full) ‖ p_k)]`

   Each sub-model learns from the gold option and from the full model's own distribution.
3. **A hidden-state stop gate.** At each exit, a logistic regression on the state right after the
   exit layer (before the coda) predicts whether that exit answers correctly. A decision stops at
   the first exit whose gate fires. Passing an exit costs nothing extra, so hard decisions never
   run deeper than the last exit.

## Install

```bash
pip install -e .            # torch, transformers >= 5, numpy
pip install -e ".[data]"    # + pandas / pyarrow, only to rebuild the training data
```

The benchmark is read directly from the [Intern-Decision](https://github.com/InternLM/Intern-Decision)
repository (`benchmarks/accuracy-v1`, seven suites, 12,351 decisions). Clone it next to this repo.
Models are Qwen3.5 checkpoints (`Qwen/Qwen3.5-4B` in the paper).

## Pipeline

`scripts/run_pipeline.sh` runs every step. The steps one by one:

```bash
# 1. training data (44,941 decisions from public train splits, deduplicated against the benchmark)
python scripts/build_train_data.py --raw data/raw --bundle Intern-Decision/benchmarks/accuracy-v1 --out data/train

# 2. nested training (one H200: ~80 min for Qwen3.5-4B)
python -m nestone.train --base Qwen/Qwen3.5-4B --data data/train --out runs/nestone

# 3. one temperature per exit on the held-out calibration split
python -m nestone.calibrate --ckpt runs/nestone/checkpoint --data data/train

# 4. benchmark accuracy of every exit; also caches exit logits and gate inputs
python -m nestone.evaluate --ckpt runs/nestone/checkpoint --bundle Intern-Decision/benchmarks/accuracy-v1 \
    --calib data/train/calib.jsonl --out runs/nestone/eval

# 5a. paper evaluation of the gate (cross-fitted on the benchmark, with confidence baselines)
python -m nestone.gate crossfit --bench runs/nestone/eval/bench --extra runs/nestone/eval/calib \
    --calibration runs/nestone/checkpoint/calibration.json --exits 9,13,21

# 5b. a deployable gate, fitted on labelled data that looks like your traffic
python -m nestone.gate fit --fit runs/nestone/eval/bench --extra runs/nestone/eval/calib \
    --exits 9,13,21,29 --delta 2 --out runs/nestone/gate.npz

# 6. decisions
python -m nestone.decide --ckpt runs/nestone/checkpoint --exit 21 --rows rows.jsonl              # fixed depth
python -m nestone.decide --ckpt runs/nestone/checkpoint --gate runs/nestone/gate.npz --rows rows.jsonl  # adaptive
```

Rows use Intern-Decision's record format: a `state` and an ordered dict of `questions`
(`choice` / `score` / `noul`); every question field is one decision.

```json
{"id": "r1", "state": {"text": "Fed raises rates again"},
 "questions": {"decision": {"type": "choice", "instructions": "Which news topic is this?",
   "criteria": {"world": "World news", "sports": "Sports", "business": "Business", "sci_tech": "Science and technology"}}}}
```

## Choosing exits

Exits are configuration everywhere, given as the layer after which a sub-model reads out (a
*k*-layer sub-model exits after layer *k* − 3; the full 32-layer model exits after layer 29).

| Where | Option | Example |
|---|---|---|
| Training | `--sub-exits` exit:inclusion-probability per step | `21:0.5,13:1,9:1` (paper), `9:1,13:1,17:1,21:0.5` |
| Evaluation | `--exits` (default: the checkpoint's) | `9,13,17,21,29` |
| Gate | `--exits`; a gate is trained at every listed exit except the last, which always answers | `9,13,21` (stop at 24 layers), `9,13,21,29` (full model last) |
| Decisions | `--exit` for one fixed depth, or `--gate` for adaptive depth | `--exit 21` |

Any layer below the coda can be read out, but layers not trained as exits are usually much
less accurate. Train the exits you plan to use.

**Gate threshold.** `--delta` sets how much benchmark-average accuracy you accept to lose against
always answering at the last exit. The gate uses one threshold for all exits: the most aggressive
value that stays within `delta`, chosen on out-of-sample gate scores. Small values (0.25) are
safe on every suite. Larger values (2) save more layers on easy traffic.

## Results (Qwen3.5-4B, 7 suites, macro average)

One NestOne checkpoint against models trained separately at each depth (mean ± sd over 3 seeds):

| Layers | Separate | NestOne |
|---:|---:|---:|
| 32 | 84.3 ± 0.2 | 84.0 ± 0.6 |
| 24 | 83.5 ± 0.8 | 83.7 ± 0.3 |
| 16 | 72.8 ± 0.9 | 74.7 ± 0.9 |
| 12 | 61.5 ± 1.5 | 64.4 ± 0.4 |

Stop gate over exits 9 → 13 → 21 (seed 42; fixed exit after layer 21 = 83.8 at 24 layers):

| δ | Average | Mean layers | Easy suites | Hard suites | Change on the 4 largest suites |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 82.3 | 21.6 | 20.2 | 24.0 | +0.08 |
| 2 | 82.1 | 20.2 | 17.5 | 23.9 | +0.03 |

Easy = AGNews and WildJailbreak; hard = Typed and J-Hard. The drop in the average comes from the
three smallest suites (430 of 12,351 decisions). With the full model as the last exit
(9 → 13 → 21 → 29, δ = 2), the gate keeps the full model's accuracy on the four largest suites at
23.9 layers on average instead of 32.

**Reproducibility.** `nestone.evaluate` scores decisions in length-grouped batches in bf16. On the
paper's checkpoint this agrees with one-decision-at-a-time scoring on 99.3–99.7% of decisions
per exit, so exit accuracies can differ from the table above by up to about 0.3 points. Use
`--micro-tokens 1` to score one decision at a time.

## Layout

```
nestone/
  schema.py     Intern-Decision prompt compiler (vendored, Apache-2.0)
  data.py       rows, benchmark suites, encoding, batching
  model.py      loading; LayerRunner: run any layers, continue from saved states, coda readout
  train.py      nested training
  calibrate.py  per-exit temperatures
  collect.py    one shared prefix pass per batch: exit logits + gate inputs
  evaluate.py   benchmark scores per exit, caches for the gate
  gate.py       stop gate: fit (deployable), crossfit (paper protocol and baselines)
  decide.py     inference at a fixed exit or with the gate
scripts/        training-data builder, end-to-end pipeline
tests/          CPU tests of the gate logic
```

## Citation

```bibtex
@article{nguyen2026nestone,
  title  = {From Large Language Models to Efficient System One Models via Adaptive Computation},
  author = {Nguyen, Minh-Anh},
  year   = {2026}
}
```

## License

Apache License 2.0. `nestone/schema.py` comes from Intern-Decision (Apache-2.0); see `NOTICE`.
The benchmark data and training sources keep their own licenses and are not redistributed here.
