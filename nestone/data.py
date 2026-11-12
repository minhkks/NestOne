"""Decision rows, prompt encoding and batching.

A row follows Intern-Decision's canonical record: a ``state``, an ordered dict of ``questions``
(``choice`` / ``score`` / ``noul``), and ``targets[field].label`` for supervised rows. Every
question field is one decision. The prompt is compiled by Intern-Decision's ``compile_row``
(vendored in :mod:`nestone.schema`): the assistant turn is a JSON skeleton with one
``<decision>`` marker per field, and the decision for a field is read at the position just
before its marker, restricted to the field's option symbols (A, B, C, ...).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from . import schema as S

# Benchmark suites of Intern-Decision's accuracy-v1 bundle (name -> path inside the bundle).
SUITES = {
    "J-Easy": "jevbench/easy.jsonl",
    "J-Orig": "jevbench/original.jsonl",
    "J-Hard": "jevbench/hard.jsonl",
    "Typed": "typed_decisions/test.jsonl",
    "ToolACE": "toolace/test.jsonl",
    "AGNews": "agnews/test.jsonl",
    "WildJB": "wildjailbreak/test.jsonl",
}


@dataclass
class Example:
    """One encoded row; lists are aligned over the row's decision fields."""
    ids: list[int]
    pos: list[int]          # readout position per field (token before its marker)
    syms: list[list[int]]   # option-symbol token ids per field
    gold: list[int]         # gold option index per field (-1 if unlabelled)
    keys: list[str]         # "<row id>/<field>"
    suite: str = ""


def canonical(raw: dict) -> dict:
    """Bring a benchmark or training record into the canonical row format."""
    if "questions" in raw:
        row = {k: raw[k] for k in ("id", "state", "questions", "targets") if k in raw}
    else:  # single-question form: {"question": {...}, "expected": label}
        row = {"id": raw["id"], "state": raw["state"], "questions": {"decision": raw["question"]},
               "targets": {"decision": {"label": raw["expected"]}}}
    row["images"] = raw.get("images", [])
    if row["images"]:
        raise NotImplementedError(f"{row['id']}: image rows are not supported")
    return row


def read_rows(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [canonical(json.loads(line)) for line in f if line.strip()]


def read_benchmark(bundle: str | Path, suites: list[str] | None = None) -> dict[str, list[dict]]:
    """Rows per suite from Intern-Decision's ``benchmarks/accuracy-v1`` directory."""
    names = suites or list(SUITES)
    return {s: read_rows(Path(bundle) / SUITES[s]) for s in names}


class Encoder:
    def __init__(self, tok):
        self.tok = tok
        self.marker = tok.convert_tokens_to_ids(S.DECISION_TOKEN)
        self._sym = {}

    def symbol_id(self, s: str) -> int:
        if s not in self._sym:
            enc = self.tok.encode(s, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"answer symbol {s!r} is not a single token")
            self._sym[s] = enc[0]
        return self._sym[s]

    def __call__(self, row: dict, suite: str = "") -> Example:
        comp = S.compile_row(row, include_targets=True)
        text = self.tok.apply_chat_template(comp.messages, tokenize=False, add_generation_prompt=False,
                                            enable_thinking=False)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        markers = [i for i, t in enumerate(ids) if t == self.marker]
        if len(markers) != len(comp.fields):
            raise ValueError(f"{row['id']}: {len(markers)} markers for {len(comp.fields)} fields")
        gold = []
        for f in comp.fields:
            tgt = (comp.targets or {}).get(f)
            gold.append(comp.symbols[f].index(tgt) if tgt is not None else -1)
        return Example(ids=ids, pos=[m - 1 for m in markers],
                       syms=[[self.symbol_id(s) for s in comp.symbols[f]] for f in comp.fields],
                       gold=gold, keys=[f"{row['id']}/{f}" for f in comp.fields], suite=suite)


def encode_rows(enc: Encoder, rows: list[dict], max_len: int | None = None, suite: str = ""):
    out, skipped = [], 0
    for r in rows:
        e = enc(r, suite)
        if max_len and len(e.ids) > max_len:
            skipped += 1        # over-length rows are dropped, not truncated
            continue
        out.append(e)
    return out, skipped


def micro_batches(examples: list[Example], max_tokens: int, rng: random.Random | None = None) -> list[list[int]]:
    """Length-grouped micro-batches with batch_size * longest <= max_tokens (shuffled if rng is given)."""
    idx = list(range(len(examples)))
    if rng:
        rng.shuffle(idx)
    batches = []
    for chunk in (idx[i:i + 2000] for i in range(0, len(idx), 2000)):
        chunk.sort(key=lambda i: len(examples[i].ids))
        cur, longest = [], 0
        for i in chunk:
            n = len(examples[i].ids)
            if cur and max(longest, n) * (len(cur) + 1) > max_tokens:
                batches.append(cur)
                cur, longest = [], 0
            cur.append(i)
            longest = max(longest, n)
        if cur:
            batches.append(cur)
    if rng:
        rng.shuffle(batches)
    return batches


@dataclass
class Batch:
    ids: torch.Tensor       # [B, L], right-padded
    mask: torch.Tensor      # [B, L]
    rows: torch.Tensor      # [N] batch row of each decision
    cols: torch.Tensor      # [N] readout position of each decision
    syms: list[list[int]]   # [N] option-symbol ids
    gold: torch.Tensor      # [N] gold option index


def collate(examples: list[Example], batch: list[int], pad_id: int, device) -> Batch:
    L = max(len(examples[i].ids) for i in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    rows, cols, syms, gold = [], [], [], []
    for b, i in enumerate(batch):
        e = examples[i]
        ids[b, :len(e.ids)] = torch.tensor(e.ids)
        mask[b, :len(e.ids)] = 1
        rows += [b] * len(e.pos)
        cols += e.pos
        syms += e.syms
        gold += e.gold
    return Batch(ids.to(device), mask.to(device), torch.tensor(rows, device=device),
                 torch.tensor(cols, device=device), syms, torch.tensor(gold, device=device))
