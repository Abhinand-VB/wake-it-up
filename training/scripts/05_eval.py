"""Stage 05: evaluate + tune thresholds -> runs/<id>/report.md, thresholds.json, eval.json

Threshold rule (on val):
  * risk: highest threshold whose val recall >= --risk-recall (default 0.85), i.e. the
    lowest-FPR operating point that meets the recall target (§8.5).
  * other trained labels: threshold maximising val F1 (they only choose the category
    shown in the detail card; a recall target on rare labels gives unusable precision).
  * misread_risk: fixed 0.99 (no training signal in this stand-in).
Metrics on test use binary labels (rater fraction >= 0.5).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (f1_score, precision_recall_curve, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)

import tg_format  # noqa: E402
from tg_common import PROCESSED, read_jsonl, record_inputs, resolve_run  # noqa: E402

LABELS = tg_format.LABELS
FIXED = {"misread_risk": 0.99}


def encode_all(recs, tok):
    feats = []
    for r in recs:
        e = tg_format.encode(*record_inputs(r), tok=tok)
        feats.append((e["input_ids"], e["token_type_ids"]))
    y = np.array([[r["labels"][l] for l in LABELS] for r in recs], dtype=np.int64)
    return feats, y


@torch.no_grad()
def predict(model, feats, device, bs=256):
    order = sorted(range(len(feats)), key=lambda i: len(feats[i][0]))
    out = np.zeros((len(feats), len(LABELS)), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(feats[i][0]) for i in idx)
        ids = torch.zeros(len(idx), L, dtype=torch.long)
        tt = torch.zeros_like(ids)
        am = torch.zeros_like(ids)
        for j, i in enumerate(idx):
            a, t = feats[i]
            ids[j, :len(a)] = torch.tensor(a); tt[j, :len(t)] = torch.tensor(t); am[j, :len(a)] = 1
        logits = model(input_ids=ids.to(device), token_type_ids=tt.to(device), attention_mask=am.to(device)).logits
        out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def tune_thresholds(y, p, risk_recall):
    th = {}
    for i, l in enumerate(LABELS):
        if l in FIXED or y[:, i].sum() == 0:
            th[l] = FIXED.get(l, 0.99)
            continue
        pr, rc, t = precision_recall_curve(y[:, i], p[:, i])
        pr, rc = pr[:-1], rc[:-1]  # align with thresholds
        if l == "risk":
            ok = np.where(rc >= risk_recall)[0]
            th[l] = float(t[ok.max()]) if len(ok) else float(t[0])
        else:
            f = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
            th[l] = float(t[int(f.argmax())])
        th[l] = round(th[l], 4)
    return th


def metrics(y, p, th):
    res = {}
    for i, l in enumerate(LABELS):
        yi, pi = y[:, i], p[:, i]
        pred = (pi >= th[l]).astype(int)
        neg = yi == 0
        res[l] = {
            "n_pos": int(yi.sum()),
            "precision": round(float(precision_score(yi, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(yi, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(yi, pred, zero_division=0)), 4),
            "fpr": round(float(pred[neg].mean()) if neg.any() else 0.0, 4),
            "auc": round(float(roc_auc_score(yi, pi)), 4) if 0 < yi.sum() < len(yi) else None,
            "threshold": th[l],
        }
    yr, pr = y[:, 0], (p[:, 0] >= th["risk"]).astype(int)
    res["risk"]["macro_f1"] = round(float(f1_score(yr, pr, average="macro")), 4)
    return res


def table(m):
    rows = ["| label | thr | n_pos | precision | recall | F1 | FPR | AUC |", "|---|---|---|---|---|---|---|---|"]
    for l in LABELS:
        r = m[l]
        rows.append(f"| {l} | {r['threshold']:.3f} | {r['n_pos']} | {r['precision']:.3f} | {r['recall']:.3f} | "
                    f"{r['f1']:.3f} | {r['fpr']:.3f} | {r['auc'] if r['auc'] is not None else 'n/a'} |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--risk-recall", type=float, default=0.85)
    args = ap.parse_args()
    run = resolve_run(args.run)

    from transformers import AutoModelForSequenceClassification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(run / "checkpoint").to(device).eval()
    tok = tg_format.get_tokenizer()

    val_f, yv = encode_all(read_jsonl(PROCESSED / "val.jsonl"), tok)
    test_f, yt = encode_all(read_jsonl(PROCESSED / "test.jsonl"), tok)
    pv, pt = predict(model, val_f, device), predict(model, test_f, device)
    ev = run / "eval"
    ev.mkdir(exist_ok=True)
    np.save(ev / "val_probs.npy", pv); np.save(ev / "test_probs.npy", pt)
    np.save(ev / "test_labels.npy", yt)

    th = tune_thresholds(yv, pv, args.risk_recall)
    mv, mt = metrics(yv, pv, th), metrics(yt, pt, th)
    (run / "thresholds.json").write_text(json.dumps(th, indent=2), encoding="utf-8")
    (run / "eval.json").write_text(json.dumps({"thresholds": th, "val": mv, "test": mt}, indent=2), encoding="utf-8")

    hist = {}
    if (run / "train_history.json").exists():
        hist = json.loads((run / "train_history.json").read_text(encoding="utf-8"))
    report = f"""# WakeItUp stand-in model report — `{run.name}`

Draft-only stand-in fine-tuned on **civil_comments** (no conversational context; segment A is
always `none`). Evaluation uses binary labels (rater fraction >= 0.5). There is **no golden set
yet**, so the §8.6 golden-set gates are not evaluated.

- Train/val/test sizes: {hist.get('n_train', '?')} / {len(yv)} / {len(yt)}
- Training: {hist.get('total_minutes', '?')} min on {hist.get('device', '?')}, torch {hist.get('torch', '?')}
- Threshold rule: risk = highest threshold with val recall >= {args.risk_recall}; other labels = val F1-max;
  misread_risk fixed at 0.99 (untrained, always-0 targets).
- **Test `risk` macro-F1 (both classes): {mt['risk']['macro_f1']:.4f}** (§8.6 gate >= 0.80)

## Test set (tuned thresholds)

{table(mt)}

## Validation set (thresholds tuned here)

{table(mv)}

## Label mapping caveats

- `hate_slur` and `insensitive` are both mapped from civil_comments `identity_attack` (identical targets).
- `harassment` <- `insult`, `threat` <- `threat`, `sexual` <- `sexual_explicit`, `risk` <- `toxicity`.
- `misread_risk` has no data; its threshold (0.99) effectively disables it.
"""
    (run / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
