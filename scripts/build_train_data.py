#!/usr/bin/env python
"""Build the training data (train / val / calib JSONL) from public train splits.

    python scripts/build_train_data.py --raw data/raw --bundle Intern-Decision/benchmarks/accuracy-v1 --out data/train

Rows follow Intern-Decision's canonical record (state, ordered questions, targets[field].label).
Nothing is taken from the benchmark test suites, and training texts that also occur in them are
dropped (exact match after whitespace/case normalisation). Download each source first with
`hf download <repo> --repo-type dataset --local-dir <raw>/<org>_<name>`:

  fancyzhx/ag_news            AG News train: choice (same question as the AGNews suite)
  Team-ACE/ToolACE            choice: which single tool to call (same format as the ToolACE suite)
  lmsys/toxic-chat, PKU-Alignment/BeaverTails
                              choice harmful / benign (stand-in for WildJailbreak, which is gated)
  PolyAI/banking77            choice, 20 sampled intents (77 > 62 answer symbols)
  mteb/amazon_massive_intent  choice, 20 sampled intents
  google/boolq                noul
  Yelp/yelp_review_full       multi-field rows: score (stars) + choice (sentiment) + noul (recommend)

The paper's split has 44,941 training decisions; val (2%) is for loss monitoring and calib (3%)
for temperature fitting and as extra gate training data.
"""
import argparse
import gzip
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nestone import schema as S  # noqa: E402

BENCH = None  # set from --bundle
N_OPTIONS = 20  # sampled candidate set for BANKING77 / MASSIVE


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def test_texts() -> set[str]:
    """Every state string in the benchmark test suites (to drop overlaps)."""
    out = set()
    for f in BENCH.rglob("*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            st = r.get("state")
            vals = [st] if isinstance(st, str) else [v for v in (st or {}).values() if isinstance(v, str)]
            out.update(norm(v) for v in vals)
    return out


def bench_question(rel: str) -> dict:
    return json.loads(open(BENCH / rel).readline())["questions"]["decision"]


def row(rid, state, questions, targets, source):
    r = {"id": rid, "state": state, "images": [], "questions": questions,
         "targets": {k: {"label": v} for k, v in targets.items()}, "source": source}
    S.compile_row(r)  # validates options/labels with the official compiler
    return r


def ag_news(raw, rng, n, seen):
    q = bench_question("agnews/test.jsonl")
    labels = ["world", "sports", "business", "sci_tech"]  # HF label ids 0..3
    df = pd.read_parquet(raw / "fancyzhx_ag_news/data/train-00000-of-00001.parquet").sample(frac=1, random_state=1)
    out = []
    for i, r in enumerate(df.itertuples()):
        if norm(r.text) in seen:
            continue
        out.append(row(f"agnews-tr-{i}", {"text": r.text}, {"decision": q}, {"decision": labels[int(r.label)]}, "agnews"))
        if len(out) >= n:
            break
    return out


def toolace(raw, rng, n, seen):
    instr = bench_question("toolace/test.jsonl")["instructions"]
    data = json.load(open(raw / "Team-ACE_ToolACE/data.json"))
    rng.shuffle(data)
    out = []
    for i, d in enumerate(data):
        sysmsg = d.get("system", "")
        j = sysmsg.find("[", sysmsg.find("functions in JSON format"))
        try:
            tools, _ = json.JSONDecoder().raw_decode(sysmsg[j:])
        except ValueError:
            continue
        conv = d.get("conversations", [])
        if len(conv) < 2 or conv[0]["from"] != "user" or conv[1]["from"] != "assistant":
            continue
        names = [t.get("name") for t in tools if isinstance(t, dict)]
        if len(names) < 2 or len(set(names)) != len(names) or len(names) > 62 or not all(names):
            continue
        call = conv[1]["value"]
        called = [nm for nm in names if f"{nm}(" in call]
        if len(called) != 1 or not call.startswith("[") or call.count("), ") > 0:
            continue  # exactly one tool call
        request = conv[0]["value"]
        if norm(request) in seen:
            continue
        crit = {t["name"]: (t.get("description") or t["name"]).strip() for t in tools}
        state = {"request": request, "tools": [{"name": k, "description": v} for k, v in crit.items()]}
        q = {"type": "choice", "instructions": instr, "criteria": crit}
        out.append(row(f"toolace-tr-{i}", state, {"decision": q}, {"decision": called[0]}, "toolace"))
        if len(out) >= n:
            break
    return out


def safety(raw, rng, n, seen):
    q = bench_question("wildjailbreak/test.jsonl")
    tc = pd.read_csv(raw / "lmsys_toxic-chat/data/0124/toxic-chat_annotation_train.csv")
    harm, ben = [], []
    for r in tc.itertuples():
        (harm if (r.toxicity == 1 or r.jailbreaking == 1) else ben).append(r.user_input)
    # BeaverTails labels the (prompt, response) pair; keep prompts whose every response is unsafe
    by_prompt: dict[str, list[bool]] = {}
    for line in gzip.open(raw / "PKU-Alignment_BeaverTails/round0/30k/train.jsonl.gz"):
        b = json.loads(line)
        by_prompt.setdefault(b["prompt"], []).append(str(b["is_safe"]) == "True")
    bt = [p for p, safe in by_prompt.items() if len(safe) >= 2 and not any(safe)]
    rng.shuffle(bt)
    harm += bt
    rng.shuffle(harm); rng.shuffle(ben)
    n_h = min(len(harm), n // 2)
    picked = [(p, "harmful") for p in harm[:n_h]] + [(p, "benign") for p in ben[: n - n_h]]
    rng.shuffle(picked)
    out = []
    for i, (p, lab) in enumerate(picked):
        if not isinstance(p, str) or not p.strip() or norm(p) in seen or "<decision>" in p:
            continue
        out.append(row(f"safety-tr-{i}", {"prompt": p}, {"decision": q}, {"decision": lab}, "safety"))
    return out


def intents(texts_labels, names, rng, n, seen, prefix, instr, key):
    out = []
    for i, (t, lab) in enumerate(texts_labels):
        if norm(t) in seen:
            continue
        cands = [lab] + rng.sample([x for x in names if x != lab], N_OPTIONS - 1)
        rng.shuffle(cands)
        crit = {c: c.replace("_", " ") for c in cands}
        q = {"type": "choice", "instructions": instr, "criteria": crit}
        out.append(row(f"{prefix}-tr-{i}", {key: t}, {"decision": q}, {"decision": lab}, prefix))
        if len(out) >= n:
            break
    return out


def banking77(raw, rng, n, seen):
    readme = open(raw / "PolyAI_banking77/README.md").read()
    names = re.findall(r"^\s+'?(\d+)'?: (\S+)\s*$", readme, flags=re.M)
    names = [nm for _, nm in sorted(names, key=lambda x: int(x[0]))]
    assert len(names) == 77, len(names)
    df = pd.read_parquet(raw / "PolyAI_banking77/data/train-00000-of-00001.parquet").sample(frac=1, random_state=2)
    tl = [(r.text, names[int(r.label)]) for r in df.itertuples()]
    return intents(tl, names, rng, n, seen, "banking77",
                   "Which banking intent does this customer message express?", "message")


def massive(raw, rng, n, seen):
    m = [json.loads(l) for l in gzip.open(raw / "mteb_amazon_massive_intent/train/en.json.gz")]
    rng.shuffle(m)
    names = sorted({x["label_text"] for x in m})
    return intents([(x["text"], x["label_text"]) for x in m], names, rng, n, seen, "massive",
                   "Which intent does this voice-assistant command express?", "utterance")


def boolq(raw, rng, n, seen):
    df = pd.read_parquet(raw / "google_boolq/data/train-00000-of-00001.parquet").sample(frac=1, random_state=3)
    crit = {"true": "The passage supports answering yes.", "false": "The passage supports answering no."}
    out = []
    for i, r in enumerate(df.itertuples()):
        if norm(r.passage) in seen:
            continue
        q = {"type": "noul", "instructions": r.question.strip().capitalize() + "?", "criteria": crit}
        out.append(row(f"boolq-tr-{i}", {"passage": r.passage}, {"decision": q},
                       {"decision": "yes" if str(r.answer) == "True" else "no"}, "boolq"))
        if len(out) >= n:
            break
    return out


def yelp(raw, rng, n, seen):
    df = pd.read_parquet(raw / "Yelp_yelp_review_full/yelp_review_full/train-00000-of-00001.parquet")
    df = df[df.text.str.len() < 2500].sample(n=n * 3, random_state=4)
    stars = {"type": "score", "instructions": "How many stars did the reviewer give?",
             "criteria": ["1 star: very negative experience.", "2 stars: mostly negative experience.",
                          "3 stars: mixed or average experience.", "4 stars: mostly positive experience.",
                          "5 stars: excellent experience."]}
    senti = {"type": "choice", "instructions": "What is the overall sentiment of the review?",
             "criteria": {"negative": "The reviewer is unhappy.", "neutral": "The reviewer is neither clearly happy nor unhappy.",
                          "positive": "The reviewer is happy."}}
    rec = {"type": "noul", "instructions": "The reviewer would recommend this business to others.",
           "criteria": {"true": "The review indicates a recommendation.", "false": "The review indicates no recommendation."}}
    out = []
    for i, r in enumerate(df.itertuples()):
        if norm(r.text) in seen:
            continue
        s = int(r.label)  # 0..4
        qs = {"stars": stars, "sentiment": senti}
        tg = {"stars": str(s), "sentiment": ["negative", "negative", "neutral", "positive", "positive"][s]}
        if s != 2:
            qs["recommend"] = rec
            tg["recommend"] = "yes" if s >= 3 else "no"
        out.append(row(f"yelp-tr-{i}", {"review": r.text}, qs, tg, "yelp"))
        if len(out) >= n:
            break
    return out


def main():
    global BENCH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="directory with the downloaded source datasets")
    ap.add_argument("--bundle", required=True, help="Intern-Decision benchmarks/accuracy-v1 directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--calib-frac", type=float, default=0.03)
    a = ap.parse_args()
    BENCH = Path(a.bundle)
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    seen = test_texts()
    # rows per source; Yelp rows carry 2-3 decisions each -> about 50k decisions in total
    plan = [(ag_news, 8000), (toolace, 6000), (safety, 8000), (banking77, 6000),
            (massive, 6000), (boolq, 6000), (yelp, 3500)]
    splits = {"train": [], "val": [], "calib": []}
    stats = {}
    for fn, n in plan:
        rows = fn(raw, rng, n, seen)
        rng.shuffle(rows)
        nv, nc = int(len(rows) * a.val_frac), int(len(rows) * a.calib_frac)
        splits["val"] += rows[:nv]
        splits["calib"] += rows[nv:nv + nc]
        splits["train"] += rows[nv + nc:]
        dec = sum(len(r["questions"]) for r in rows)
        stats[fn.__name__] = {"rows": len(rows), "decisions": dec,
                              "labels": dict(Counter(t["label"] for r in rows for t in r["targets"].values()).most_common(6))}
        print(f"{fn.__name__:10s} rows={len(rows):6d} decisions={dec:6d}")
    for name, rows in splits.items():
        rng.shuffle(rows)
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = {"rows": len(rows), "decisions": sum(len(r["questions"]) for r in rows)}
        print(f"{name:6s} rows={len(rows):6d} decisions={stats[name]['decisions']:6d}")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
