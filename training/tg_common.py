"""Shared paths/helpers for the training scripts.

Environment overrides (so every stage works for Phase 1 and Phase 2 data):
    WIU_DATA           processed dataset folder under data/   (default "processed" = Phase 1)
    WIU_MODEL_VERSION  model_version written into model_config.json (default Phase 1 stand-in)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROCESSED = DATA / os.environ.get("WIU_DATA", "processed")
RUNS = ROOT / "runs"
CONFIGS = ROOT / "configs"
GOLDEN = ROOT / "golden" / "golden_set.jsonl"
REPO = ROOT.parent
ANDROID_MODEL_DIR = REPO / "android" / "app" / "src" / "main" / "assets" / "model"
ANDROID_FIXTURES_DIR = REPO / "android" / "app" / "src" / "test" / "resources" / "fixtures"

MODEL_VERSION = os.environ.get("WIU_MODEL_VERSION", "standin-2026.10.0")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_run(run: str | None) -> Path:
    """Return runs/<run>, or the most recent run that has a checkpoint."""
    if run:
        p = Path(run)
        return p if p.is_absolute() else RUNS / run
    cands = [d for d in RUNS.glob("*") if (d / "checkpoint").exists()]
    if not cands:
        raise SystemExit("no runs with a checkpoint found in runs/")
    return max(cands, key=lambda d: (d / "checkpoint").stat().st_mtime)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record_inputs(rec: dict) -> tuple[list, int | None, str]:
    """(messages, reply_index, draft) for tg_format.encode from a §8.2 record."""
    return rec.get("context") or [], rec.get("reply_to"), rec["draft"]
