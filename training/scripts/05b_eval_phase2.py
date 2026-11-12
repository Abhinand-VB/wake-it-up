"""Stage 05b: Phase 2 evaluation -> runs/<id>/{thresholds.json, eval.json, report.md}

- Thresholds tuned on VAL, ignoring masked labels:
    risk       = F1-max on the focus sources (synthetic + re-labeled), where the hard cases live
    categories = F1-max on records where the label is unmasked; labels without positives -> 0.99
- Test metrics reported overall and per source.
- Golden set (training/golden/golden_set.jsonl, never trained on): §8.6 gates
    recall on should-warn (risk >= 0.5) and false-positive rate on should-not-warn,
    plus breakdowns by scenario and difficulty.
- Probe sentences (fixed list) for a quick human-readable sanity check.

Usage: WIU_DATA=processed_p2 .venv/Scripts/python.exe scripts/05b_eval_phase2.py --run <run-id>
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score  # noqa: E402

import tg_format  # noqa: E402
from tg_common import GOLDEN, PROCESSED, read_jsonl, record_inputs, resolve_run  # noqa: E402

LABELS = tg_format.LABELS
FOCUS = {"synthetic", "civil_relabeled"}
GATES = {"golden_recall_risky": 0.85, "golden_fpr_benign": 0.05}

PROBES = [
    ([], None, "I like slaves", 1),
    ([{"speaker": "A", "text": "he never hangs out anymore, we just make him slave away"}], 0, "I like slaves", 1),
    ([], None, "slavery wasn't that bad honestly", 1),
    ([], None, "the holocaust was kind of funny", 1),
    ([], None, "women are too emotional to be leaders", 1),
    ([], None, "go back to where you came from", 1),
    ([], None, "you are an idiot", 1),
    ([], None, "I will find you and hurt you", 1),
    ([], None, "damn that game was crazy", 0),
    ([], None, "holy shit that's amazing", 0),
    ([], None, "we learned about slavery in history class today", 0),
    ([], None, "he's been slaving away at work all week", 0),
    ([], None, "lmao you're such an idiot, love you bro", 0),
    ([], None, "see you at practice", 0),
]


def encode(recs, tok):
    return [tg_format.encode(*record_inputs(r), tok=tok) for r in recs]


@torch.no_grad()
def predict(model, encs, device, bs=128):
    order = sorted(range(len(encs)), key=lambda i: len(encs[i]["input_ids"]))
    out = np.zeros((len(encs), len(LABELS)), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(encs[i]["input_ids"]) for i in idx)
        ids = torch.zeros(len(idx), L, dtype=torch.long)
        tt, am = torch.zeros_like(ids), torch.zeros_like(ids)
        for j, i in enumerate(idx):
            e = encs[i]
            n = len(e["input_ids"])
            ids[j, :n] = torch.tensor(e["input_ids"]); tt[j, :n] = torch.tensor(e["token_type_ids"]); am[j, :n] = 1
        lg = model(input_ids=ids.to(device), token_type_ids=tt.to(device), attention_mask=am.to(device)).logits
        out[idx] = torch.sigmoid(lg.float()).cpu().numpy()
    return out


def arrays(recs):
    y = np.array([[r["labels"][l] for l in LABELS] for r in recs], dtype=np.int64)
    m = np.array([[(r.get("label_mask") or {}).get(l, 1) for l in LABELS] for r in recs], dtype=np.int64)
    src = np.array([r.get("source", "?") for r in recs])
    return y, m, src


def f1_max_threshold(y, p):
    if y.sum() == 0:
        return 0.99
    pr, rc, t = precision_recall_curve(y, p)
    f = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    return round(float(t[int(f.argmax())]), 4)


def binary_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    pos, neg = y == 1, y == 0
    out = {"n": int(len(y)), "n_pos": int(pos.sum()),
           "recall": round(float(pred[pos].mean()), 4) if pos.any() else None,
           "fpr": round(float(pred[neg].mean()), 4) if neg.any() else None,
           "precision": round(float(y[pred == 1].mean()), 4) if pred.any() else None,
           "f1": round(float(f1_score(y, pred, zero_division=0)), 4)}
    if 0 < pos.sum() < len(y):
        out["auc"] = round(float(roc_auc_score(y, p)), 4)
    return out


def golden_flag(probs, th):
    """App-equivalent decision (DecisionEngine, NORMAL, not showing): risk OR any category over threshold."""
    flags = probs[:, 0] >= th["risk"]
    for i, l in enumerate(LABELS[1:], start=1):
        flags |= probs[:, i] >= th[l]
    return flags


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = resolve_run(args.run)

    from transformers import AutoModelForSequenceClassification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(run / "checkpoint").to(device).eval()
    tok = tg_format.get_tokenizer()

    val, test = read_jsonl(PROCESSED / "val.jsonl"), read_jsonl(PROCESSED / "test.jsonl")
    pv, pt = predict(model, encode(val, tok), device), predict(model, encode(test, tok), device)
    yv, mv, sv = arrays(val)
    yt, mt, st = arrays(test)

    th = {}
    focus = np.isin(sv, list(FOCUS))
    th["risk"] = f1_max_threshold(yv[focus, 0], pv[focus, 0]) if focus.any() else f1_max_threshold(yv[:, 0], pv[:, 0])
    for i, l in enumerate(LABELS[1:], start=1):
        keep = mv[:, i] == 1
        th[l] = f1_max_threshold(yv[keep, i], pv[keep, i])

    test_m = {"overall": {}, "by_source": {}}
    for i, l in enumerate(LABELS):
        keep = mt[:, i] == 1
        test_m["overall"][l] = binary_metrics(yt[keep, i], pt[keep, i], th[l]) | {"threshold": th[l]}
    for s in sorted(set(st)):
        k = st == s
        test_m["by_source"][s] = binary_metrics(yt[k, 0], pt[k, 0], th["risk"])

    golden = {}
    if GOLDEN.exists():
        g = read_jsonl(GOLDEN)
        pg = predict(model, encode(g, tok), device)
        should = np.array([float(r["risk"]) >= 0.5 for r in g])
        flag = golden_flag(pg, th)
        golden = {"n": len(g), "n_should_warn": int(should.sum()),
                  "recall_risky": round(float(flag[should].mean()), 4),
                  "fpr_benign": round(float(flag[~should].mean()), 4)}
        for key in ("scenario", "difficulty"):
            groups = defaultdict(list)
            for r, f, s in zip(g, flag, should):
                groups[r.get(key, "?")].append((f, s))
            golden[f"by_{key}"] = {k: {"n": len(v), "correct": int(sum(f == s for f, s in v))}
                                   for k, v in sorted(groups.items())}
        golden["misses"] = [{"id": r["id"], "draft": r["draft"], "risk_p": round(float(p[0]), 3)}
                            for r, p, f, s in zip(g, pg, flag, should) if s and not f]
        golden["false_alarms"] = [{"id": r["id"], "draft": r["draft"], "risk_p": round(float(p[0]), 3)}
                                  for r, p, f, s in zip(g, pg, flag, should) if f and not s]
        golden["gates"] = {"recall_ok": golden["recall_risky"] >= GATES["golden_recall_risky"],
                           "fpr_ok": golden["fpr_benign"] <= GATES["golden_fpr_benign"]}

    probes = [{"messages": m, "reply_index": ri, "draft": d} for m, ri, d, _ in PROBES]
    pp = predict(model, [tg_format.encode(p["messages"], p["reply_index"], p["draft"], tok=tok) for p in probes], device)
    pf = golden_flag(pp, th)
    probe_rows = [{"draft": p["draft"], "context": bool(p["messages"]), "should_warn": bool(s),
                   "warn": bool(f), "risk_p": round(float(q[0]), 3)}
                  for p, (_, _, _, s), f, q in zip(probes, PROBES, pf, pp)]

    (run / "thresholds.json").write_text(json.dumps(th, indent=2), encoding="utf-8")
    # 07_quantize reads eval["test"]["risk"]["auc"]; keep that key path.
    ev = {"thresholds": th, "test": {"risk": test_m["overall"]["risk"]}, "test_detail": test_m,
          "golden": golden, "probes": probe_rows}
    (run / "eval.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Phase 2 eval — `{run.name}`", "", "## Thresholds (tuned on val)", "",
             "| label | threshold |", "|---|---|"] + [f"| {l} | {th[l]} |" for l in LABELS]
    lines += ["", "## Test — risk by source", "", "| source | n | n_pos | recall | FPR | precision | F1 | AUC |",
              "|---|---|---|---|---|---|---|---|"]
    for s, m in test_m["by_source"].items():
        lines.append(f"| {s} | {m['n']} | {m['n_pos']} | {m['recall']} | {m['fpr']} | {m['precision']} | {m['f1']} | {m.get('auc')} |")
    if golden:
        lines += ["", f"## Golden set (n={golden['n']}, should-warn {golden['n_should_warn']})", "",
                  f"- Recall on should-warn: **{golden['recall_risky']:.3f}** (gate ≥ {GATES['golden_recall_risky']}) "
                  f"{'✅' if golden['gates']['recall_ok'] else '❌'}",
                  f"- False-alarm rate on harmless: **{golden['fpr_benign']:.3f}** (gate ≤ {GATES['golden_fpr_benign']}) "
                  f"{'✅' if golden['gates']['fpr_ok'] else '❌'}", "", "| scenario | n | correct |", "|---|---|---|"]
        lines += [f"| {k} | {v['n']} | {v['correct']} |" for k, v in golden["by_scenario"].items()]
    lines += ["", "## Probes", "", "| draft | context | should warn | warns | risk p |", "|---|---|---|---|---|"]
    lines += [f"| {r['draft']} | {'yes' if r['context'] else ''} | {r['should_warn']} | {'✅' if r['warn'] == r['should_warn'] else '❌'} {r['warn']} | {r['risk_p']} |"
              for r in probe_rows]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
