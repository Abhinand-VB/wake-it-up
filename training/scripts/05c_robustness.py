"""Stage 05c: robustness of decisions to messy surface form (punctuation, case, apostrophes, typos).

For every golden record, makes N noisy variants (same meaning, same label) with a fixed seed and
reports how often the app decision flips vs. the clean text, plus recall / false alarms on the noisy
set. Added after a single comma moved the round-2 model's score for one message from 35% to 73%.

Usage: .venv/Scripts/python.exe scripts/05c_robustness.py --run <run-id> [--variants 3]
"""
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import tg_format  # noqa: E402
from tg_common import GOLDEN, read_jsonl, record_inputs, resolve_run  # noqa: E402

_spec = importlib.util.spec_from_file_location("b03b", Path(__file__).with_name("03b_build_phase2.py"))
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)
LABELS = tg_format.LABELS


@torch.no_grad()
def probs(model, recs, device):
    out = []
    for r in recs:
        e = tg_format.encode(*record_inputs(r))
        t = {k: torch.tensor([e[k]], device=device) for k in ("input_ids", "attention_mask", "token_type_ids")}
        out.append(torch.sigmoid(model(**t).logits[0]).cpu().numpy())
    return np.array(out)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--variants", type=int, default=3)
    args = ap.parse_args()
    run = resolve_run(args.run)
    th = json.loads((run / "thresholds.json").read_text(encoding="utf-8"))
    from transformers import AutoModelForSequenceClassification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(run / "checkpoint").to(device).eval()

    g = read_jsonl(GOLDEN)
    rng = random.Random(99)
    noisy = [_b.noisy_copy(r, rng, k) for k in range(args.variants) for r in g]
    flag = lambda P: np.any(np.stack([P[:, i] >= th[l] for i, l in enumerate(LABELS)]), 0)  # noqa: E731
    fc, fn = flag(probs(model, g, device)), flag(probs(model, noisy, device))
    y = np.array([r["risk"] >= 0.5 for r in g])
    yn = np.tile(y, args.variants)
    flips = float((fn != np.tile(fc, args.variants)).mean())
    res = {"variants": args.variants, "decision_flip_rate": round(flips, 4),
           "clean": {"recall": round(float(fc[y].mean()), 4), "false_alarms": round(float(fc[~y].mean()), 4)},
           "noisy": {"recall": round(float(fn[yn].mean()), 4), "false_alarms": round(float(fn[~yn].mean()), 4)}}
    (run / "robustness.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
