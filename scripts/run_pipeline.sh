#!/usr/bin/env bash
# End-to-end NestOne pipeline on one GPU (paper setting: Qwen3.5-4B, exits 9 / 13 / 21 + full).
#   bash scripts/run_pipeline.sh <Intern-Decision repo> <raw data dir> [output dir]
set -euo pipefail
ID=${1:?path to the Intern-Decision repo (for benchmarks/accuracy-v1)}
RAW=${2:?directory with the downloaded training sources}
OUT=${3:-runs/nestone}
BUNDLE=$ID/benchmarks/accuracy-v1
BASE=${BASE:-Qwen/Qwen3.5-4B}
SUB_EXITS=${SUB_EXITS:-21:0.5,13:1,9:1}     # exit layer : per-step inclusion probability

python scripts/build_train_data.py --raw "$RAW" --bundle "$BUNDLE" --out data/train
python -m nestone.train --base "$BASE" --data data/train --out "$OUT" --sub-exits "$SUB_EXITS"
python -m nestone.calibrate --ckpt "$OUT/checkpoint" --data data/train
python -m nestone.evaluate --ckpt "$OUT/checkpoint" --bundle "$BUNDLE" --calib data/train/calib.jsonl --out "$OUT/eval"
# paper evaluation of the stop gate (cross-fitted on the benchmark, with confidence baselines)
python -m nestone.gate crossfit --bench "$OUT/eval/bench" --extra "$OUT/eval/calib" \
    --calibration "$OUT/checkpoint/calibration.json" --exits 9,13,21 --out "$OUT/eval/gate_crossfit.json"
# a deployable gate over all exits; fit it on labelled data that looks like your traffic
python -m nestone.gate fit --fit "$OUT/eval/bench" --extra "$OUT/eval/calib" --exits 9,13,21,29 \
    --delta 2 --out "$OUT/gate.npz"
