"""Stage 03b: build the Phase 2 combined dataset -> data/processed_p2/{train,val,test}.jsonl

Sources (SOURCE_OF_TRUTH.md §8, D10):
  1. civil_comments, CLEANED: comments that were "toxic" only because of mild swearing
     (toxicity >= 0.5 but insult/identity/threat/sexual all low) are dropped unless re-labeled.
  2. civil_comments RE-LABELED by Claude under SPEC.md (data/raw/relabel/labeled_*.jsonl).
     Their labels replace the rater scores.
  3. SYNTHETIC chat snippets with context (data/raw/synthetic/*.jsonl), written under SPEC.md.

Label masks: public rater data has no real signal for `insensitive` or `misread_risk`, so those
labels are masked out (label_mask = 0) for plain civil_comments records; the training loss and
threshold tuning ignore masked labels. Re-labeled and synthetic records have every label.

Splits: synthetic records are split by contrastive pair (pair_id, else id) so the two halves of
a pair never straddle train/test. Public records are split randomly.

Upsampling: synthetic + re-labeled records are repeated in TRAIN ONLY so they make up about
--syn-frac of each epoch (val/test are never upsampled).

Usage: .venv/Scripts/python.exe scripts/03b_build_phase2.py
"""
import argparse
import glob
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed_p2"
LABELS = ["risk", "hate_slur", "insensitive", "harassment", "threat", "sexual", "misread_risk"]
CATEGORIES = LABELS[1:]
PUBLIC_COL = {"risk": "toxicity", "hate_slur": "identity_attack", "harassment": "insult",
              "threat": "threat", "sexual": "sexual_explicit"}
PUBLIC_MASKED = {"insensitive", "misread_risk"}
SEED = 2026
MAX_CHARS = 300


def synth_labels(risk: float, categories: list[str]):
    soft = {"risk": float(risk)}
    for c in CATEGORIES:
        soft[c] = 1.0 if c in categories else 0.0
    hard = {l: int(v >= 0.5) for l, v in soft.items()}
    return hard, soft


def load_synthetic():
    recs, bad = [], 0
    for path in sorted(glob.glob(str(RAW / "synthetic" / "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                risk = float(r["risk"])
                cats = [c for c in r.get("categories", []) if c in CATEGORIES]
                ctx = [{"speaker": str(m["speaker"]), "text": str(m["text"])} for m in r.get("context", [])]
                reply = r.get("reply_to")
                if reply is not None and not (isinstance(reply, int) and 0 <= reply < len(ctx)):
                    reply = None
                hard, soft = synth_labels(risk, cats)
                recs.append({"id": r["id"], "source": "synthetic", "scenario": r.get("scenario", "unknown"),
                             "group": r.get("pair_id") or r["id"], "context": ctx, "reply_to": reply,
                             "draft": str(r["draft"]), "labels": hard, "soft_labels": soft,
                             "label_mask": {l: 1 for l in LABELS}})
            except Exception:  # noqa: BLE001
                bad += 1
    return recs, bad


def load_relabeled():
    texts = {}
    for path in glob.glob(str(RAW / "relabel" / "to_label_*.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            texts[r["id"]] = (r["text"], r["bucket"])
    recs = []
    for path in sorted(glob.glob(str(RAW / "relabel" / "labeled_*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] not in texts:
                continue
            text, bucket = texts[r["id"]]
            hard, soft = synth_labels(float(r["risk"]), [c for c in r.get("categories", []) if c in CATEGORIES])
            recs.append({"id": r["id"], "source": "civil_relabeled", "scenario": f"relabel_{bucket}",
                         "group": r["id"], "context": [], "reply_to": None, "draft": text,
                         "labels": hard, "soft_labels": soft, "label_mask": {l: 1 for l in LABELS}})
    return recs, set(texts)


def load_public(exclude_ids: set[str], n_public: int, toxic_frac: float):
    import pandas as pd

    df = pd.concat([pd.read_parquet(RAW / "public" / "civil_comments" / f"{s}.parquet")
                    for s in ("train", "validation", "test")], ignore_index=True)
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[(df.text.str.len() >= 8) & (df.text.str.len() <= MAX_CHARS)]
    df = df[~df.text.str.lower().duplicated()]
    df["id"] = "cc-" + df.index.astype(str)  # same ids as stage 02a
    df = df[~df.id.isin(exclude_ids)]
    low = (df.insult < 0.3) & (df.identity_attack < 0.2) & (df.threat < 0.2) & (df.sexual_explicit < 0.2)
    swear_only = (df.toxicity >= 0.5) & low
    n_drop = int(swear_only.sum())
    df = df[~swear_only]
    tox, non = df[df.toxicity >= 0.5], df[df.toxicity < 0.5]
    n_tox = min(len(tox), int(n_public * toxic_frac))
    n_non = min(len(non), n_public - n_tox)
    sel = pd.concat([tox.sample(n=n_tox, random_state=SEED), non.sample(n=n_non, random_state=SEED)])
    recs = []
    for r in sel.itertuples():
        soft = {l: (round(float(getattr(r, PUBLIC_COL[l])), 4) if l in PUBLIC_COL else 0.0) for l in LABELS}
        hard = {l: int(v >= 0.5) for l, v in soft.items()}
        recs.append({"id": r.id, "source": "civil_comments", "scenario": "public_single_comment",
                     "group": r.id, "context": [], "reply_to": None, "draft": r.text,
                     "labels": hard, "soft_labels": soft,
                     "label_mask": {l: (0 if l in PUBLIC_MASKED else 1) for l in LABELS}})
    return recs, n_drop


def noisy_text(text: str, rng: random.Random) -> str:
    """Same meaning, messier surface: real chat drops punctuation, capitals and apostrophes.
    Added after the round-2 model's score for the same message moved 35% -> 73% with one comma."""
    t = text
    if rng.random() < 0.5:
        t = re.sub(r"[,.;:]", "" if rng.random() < 0.5 else " ", t)
    if rng.random() < 0.5:
        t = t.lower()
    if rng.random() < 0.4:
        t = t.replace("'", "").replace("’", "")
    if rng.random() < 0.3:
        t = re.sub(r"[!?]+", "", t) if rng.random() < 0.5 else re.sub(r"([!?])", r"\1\1", t)  # drop or double !/?
    if rng.random() < 0.25 and len(t) > 6:  # one keyboard typo: swap two adjacent letters
        i = rng.randrange(1, len(t) - 2)
        if t[i].isalpha() and t[i + 1].isalpha():
            t = t[:i] + t[i + 1] + t[i] + t[i + 2:]
    t = re.sub(r"\s+", " ", t).strip()
    return t or text


def noisy_copy(rec: dict, rng: random.Random, k: int) -> dict:
    r = dict(rec)
    r["id"] = f"{rec['id']}~aug{k}"
    r["draft"] = noisy_text(rec["draft"], rng)
    r["context"] = [{"speaker": m["speaker"], "text": noisy_text(m["text"], rng)} for m in rec["context"]]
    return r


def split_by_group(recs, val_frac, test_frac, rng):
    groups = sorted({r["group"] for r in recs})
    rng.shuffle(groups)
    n_test, n_val = int(len(groups) * test_frac), int(len(groups) * val_frac)
    test_g, val_g = set(groups[:n_test]), set(groups[n_test:n_test + n_val])
    out = {"train": [], "val": [], "test": []}
    for r in recs:
        out["test" if r["group"] in test_g else "val" if r["group"] in val_g else "train"].append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-public", type=int, default=40_000)
    ap.add_argument("--public-toxic-frac", type=float, default=0.25)
    ap.add_argument("--syn-frac", type=float, default=0.40, help="target share of synthetic+relabeled in train epochs")
    ap.add_argument("--val-frac", type=float, default=0.075)
    ap.add_argument("--test-frac", type=float, default=0.075)
    ap.add_argument("--no-augment", action="store_true", help="plain repeats instead of noisy variants")
    args = ap.parse_args()
    rng = random.Random(SEED)

    syn, bad = load_synthetic()
    # The golden set must stay unseen: drop training records whose draft matches a golden draft.
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()  # noqa: E731
    golden_path = ROOT / "golden" / "golden_set.jsonl"
    if golden_path.exists():
        golden_drafts = {norm(json.loads(l)["draft"]) for l in open(golden_path, encoding="utf-8") if l.strip()}
        before = len(syn)
        syn = [r for r in syn if norm(r["draft"]) not in golden_drafts]
        print(f"[03b] removed {before - len(syn)} synthetic records whose draft appears in the golden set")
    rel, relabel_ids = load_relabeled()
    pub, n_swear_dropped = load_public(relabel_ids, args.n_public, args.public_toxic_frac)
    print(f"[03b] synthetic {len(syn)} (unparseable {bad}), relabeled {len(rel)}, public {len(pub)} "
          f"(dropped {n_swear_dropped} unrelabeled swear-only comments)")
    if not syn:
        raise SystemExit("no synthetic data found in data/raw/synthetic/")

    parts = {k: [] for k in ("train", "val", "test")}
    for recs in (syn, rel, pub):
        for k, v in split_by_group(recs, args.val_frac, args.test_frac, rng).items():
            parts[k] += v

    # Upsample synthetic + relabeled in train only.
    focus = [r for r in parts["train"] if r["source"] != "civil_comments"]
    public = [r for r in parts["train"] if r["source"] == "civil_comments"]
    repeat = max(1, math.ceil(args.syn_frac * len(public) / max(1, (1 - args.syn_frac) * len(focus))))
    # Repeat 0 is the original; repeats 1..k-1 are noisy-surface variants (same label), so the
    # upsampling also teaches robustness to punctuation/case/apostrophes/typos.
    if args.no_augment:
        parts["train"] = public + focus * repeat
    else:
        parts["train"] = public + focus + [noisy_copy(r, rng, k) for k in range(1, repeat) for r in focus]
    rng.shuffle(parts["train"])
    print(f"[03b] train: {len(public)} public + {len(focus)} focus x{repeat} = {len(parts['train'])} "
          f"(focus share {len(focus) * repeat / len(parts['train']):.0%})")

    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"repeat": repeat, "swear_only_dropped": n_swear_dropped, "splits": {}}
    for name, recs in parts.items():
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps({k: v for k, v in r.items() if k != "group"}, ensure_ascii=False) + "\n")
        by_src = Counter(r["source"] for r in recs)
        risky = Counter(r["source"] for r in recs if r["labels"]["risk"])
        stats["splits"][name] = {"n": len(recs), "by_source": dict(by_src), "risky_by_source": dict(risky)}
        print(f"[03b] {name}: {len(recs)}  by source {dict(by_src)}  risky {dict(risky)}")
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
