"""Stage 02a: pick borderline civil_comments for re-labeling under OUR definition of offensive.

Why: civil_comments marks many comments toxic only for mild swearing, which taught the
stand-in "swearing = bad" (SOURCE_OF_TRUTH.md §6.1, D10). These comments get re-labeled
(by Claude, following training/synthetic/SPEC.md) and the new labels override the
rater scores in stage 03.

Buckets (sizes are targets):
    swear_only   toxicity >= 0.5 but insult/identity/threat/sexual all low     (likely mislabeled)
    borderline   0.3 <= toxicity < 0.6                                          (ambiguous)
    identity_ok  identity_attack < 0.1 and toxicity < 0.2, mentions an identity group
    clear_toxic  insult >= 0.6 (sanity anchor; should mostly stay risky)

Output: data/raw/relabel/to_label_<k>.jsonl  (chunks of --chunk records: id, bucket, text)
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "public" / "civil_comments"
OUT = ROOT / "data" / "raw" / "relabel"
MAX_CHARS = 300
SEED = 2026

IDENTITY = re.compile(
    r"\b(black|white|asian|latino|hispanic|jewish|jews?|muslims?|christians?|catholic|hindu|gay|lesbian|"
    r"trans(gender)?|immigrants?|mexicans?|women|men|girls|disabled|autistic|african|arab|indian|chinese)\b",
    re.IGNORECASE,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--swear-only", type=int, default=1000)
    ap.add_argument("--borderline", type=int, default=600)
    ap.add_argument("--identity-ok", type=int, default=250)
    ap.add_argument("--clear-toxic", type=int, default=150)
    args = ap.parse_args()

    import pandas as pd

    df = pd.concat([pd.read_parquet(RAW / f"{s}.parquet").assign(split=s)
                    for s in ("train", "validation", "test")], ignore_index=True)
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[(df.text.str.len() >= 8) & (df.text.str.len() <= MAX_CHARS)]
    df = df[~df.text.str.lower().duplicated()]
    df["id"] = "cc-" + df.index.astype(str)

    low = (df.insult < 0.3) & (df.identity_attack < 0.2) & (df.threat < 0.2) & (df.sexual_explicit < 0.2)
    buckets = {
        "swear_only": df[(df.toxicity >= 0.5) & low],
        "borderline": df[(df.toxicity >= 0.3) & (df.toxicity < 0.6) & ~((df.toxicity >= 0.5) & low)],
        "identity_ok": df[(df.identity_attack < 0.1) & (df.toxicity < 0.2) & df.text.str.contains(IDENTITY)],
        "clear_toxic": df[df.insult >= 0.6],
    }
    sizes = {"swear_only": args.swear_only, "borderline": args.borderline,
             "identity_ok": args.identity_ok, "clear_toxic": args.clear_toxic}

    picked, seen = [], set()
    for name, frame in buckets.items():
        frame = frame[~frame.id.isin(seen)]
        sample = frame.sample(n=min(sizes[name], len(frame)), random_state=SEED)
        seen.update(sample.id)
        print(f"[02a] {name:12s} available {len(frame):7d}  picked {len(sample)}")
        for r in sample.itertuples():
            picked.append({"id": r.id, "bucket": name, "text": r.text,
                           "orig": {"toxicity": round(r.toxicity, 3), "insult": round(r.insult, 3),
                                    "identity_attack": round(r.identity_attack, 3)}})

    import random
    random.Random(SEED).shuffle(picked)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("to_label_*.jsonl"):
        old.unlink()
    for k in range(0, len(picked), args.chunk):
        path = OUT / f"to_label_{k // args.chunk + 1}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for rec in picked[k:k + args.chunk]:
                # The labeler sees only id + text; bucket/orig stay for analysis in stage 03.
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[02a] wrote {path}")
    print(f"[02a] total {len(picked)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
