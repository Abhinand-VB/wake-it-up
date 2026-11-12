"""Stage 01: download public datasets -> data/raw/public/

Phase 1 stand-in: only google/civil_comments (Hugging Face).
Columns: text, toxicity, severe_toxicity, obscene, threat, insult,
identity_attack, sexual_explicit (floats in [0, 1]).

Idempotent: skips splits whose parquet file already exists (use --force to redo).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "public" / "civil_comments"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    splits = ["train", "validation", "test"]
    if not args.force and all((OUT / f"{s}.parquet").exists() for s in splits):
        print(f"[01] already present in {OUT}, skipping (use --force)")
        return 0

    ds = load_dataset("google/civil_comments")
    for s in splits:
        path = OUT / f"{s}.parquet"
        ds[s].to_parquet(str(path))
        print(f"[01] {s}: {len(ds[s]):,} rows -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
