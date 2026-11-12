"""Stage 03: build unified dataset (§8.2 schema) -> data/processed/{train,val,test}.jsonl

Phase 1 stand-in: civil_comments only, draft-only (empty context).

Label mapping (civil_comments float score -> WakeItUp label):
    risk         <- toxicity
    hate_slur    <- identity_attack
    insensitive  <- identity_attack   (weak proxy; no better public signal yet)
    harassment   <- insult
    threat       <- threat
    sexual       <- sexual_explicit
    misread_risk <- 0 always          (no data; effectively disabled)

Each record carries
    "labels":      binary labels (score >= 0.5) -- used for evaluation / threshold tuning
    "soft_labels": the raw rater-fraction floats   -- used as BCE targets in training

Sampling: comments <= MAX_CHARS characters (drafts are short), whitespace-normalised
and de-duplicated (case-insensitive). Keep all toxic (toxicity >= 0.5) and a random
sample of non-toxic at NONTOX_RATIO : 1, total capped at --cap. Random 90/5/5 split.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "public" / "civil_comments"
OUT = ROOT / "data" / "processed"

LABELS = ["risk", "hate_slur", "insensitive", "harassment", "threat", "sexual", "misread_risk"]
SRC_COL = {
    "risk": "toxicity",
    "hate_slur": "identity_attack",
    "insensitive": "identity_attack",
    "harassment": "insult",
    "threat": "threat",
    "sexual": "sexual_explicit",
    "misread_risk": None,
}
MAX_CHARS = 300
MIN_CHARS = 2
NONTOX_RATIO = 3
SEED = 1234


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=200_000, help="max total records (train+val+test)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    args = ap.parse_args()

    import pandas as pd

    frames = [pd.read_parquet(RAW / f"{s}.parquet") for s in ("train", "validation", "test")]
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df["text"] = df["text"].astype(str).map(lambda t: re.sub(r"\s+", " ", t).strip())
    df = df[(df["text"].str.len() >= MIN_CHARS) & (df["text"].str.len() <= MAX_CHARS)]
    df["_key"] = df["text"].str.lower()
    # conflicting duplicates: keep the max toxicity row (conservative)
    df = df.sort_values("toxicity", ascending=False).drop_duplicates("_key", keep="first")
    print(f"[03] {n0:,} raw -> {len(df):,} after length filter (<= {MAX_CHARS} chars) + dedupe")

    tox = df[df["toxicity"] >= 0.5]
    non = df[df["toxicity"] < 0.5]
    n_tox = min(len(tox), args.cap // (NONTOX_RATIO + 1))
    n_non = min(len(non), n_tox * NONTOX_RATIO)
    tox = tox.sample(n=n_tox, random_state=SEED)
    non = non.sample(n=n_non, random_state=SEED)
    sel = pd.concat([tox, non]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"[03] selected {len(sel):,} (toxic {n_tox:,}, non-toxic {n_non:,})")

    n = len(sel)
    n_val, n_test = int(n * args.val_frac), int(n * args.test_frac)
    splits = {
        "test": sel.iloc[:n_test],
        "val": sel.iloc[n_test:n_test + n_val],
        "train": sel.iloc[n_test + n_val:],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, part in splits.items():
        path = OUT / f"{name}.jsonl"
        pos = {l: 0 for l in LABELS}
        with open(path, "w", encoding="utf-8") as f:
            for i, row in enumerate(part.itertuples(index=False)):
                soft = {}
                for l in LABELS:
                    col = SRC_COL[l]
                    soft[l] = round(float(getattr(row, col)), 4) if col else 0.0
                hard = {l: int(v >= 0.5) for l, v in soft.items()}
                for l in LABELS:
                    pos[l] += hard[l]
                rec = {
                    "id": f"cc-{name}-{i:06d}",
                    "source": "civil_comments",
                    "context": [],
                    "reply_to": None,
                    "draft": row.text,
                    "labels": hard,
                    "soft_labels": soft,
                    "scenario": "public_single_comment",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[name] = {"n": len(part), "positives": pos}
        print(f"[03] {name}: {len(part):,} -> {path}  positives={pos}")
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
