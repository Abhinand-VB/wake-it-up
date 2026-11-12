"""Stage 04: fine-tune MiniLM for 7-label multi-label BCE -> runs/<id>/checkpoint

Inputs are encoded with tg_format.encode (§6.2.1); for public single comments the
context is empty, so segment A is the literal "none".

Phase 2 additions (backwards compatible with Phase 1 data):
  - per-record `label_mask`: masked labels contribute nothing to the loss or pos_weight
  - checkpoint selection = mean val risk best-F1 across sources (so the small synthetic
    set counts as much as the large public one); falls back to overall F1
  - config keys `run_prefix` (default "standin") and data dir via WIU_DATA (tg_common)
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

import tg_format  # noqa: E402
from tg_common import CONFIGS, PROCESSED, RUNS, read_jsonl, record_inputs  # noqa: E402

LABELS = tg_format.LABELS


def encode_records(recs, tok, max_len, soft):
    feats = []
    for r in recs:
        m, ri, d = record_inputs(r)
        e = tg_format.encode(m, ri, d, max_len=max_len, tok=tok)
        src = r.get("soft_labels") if soft and r.get("soft_labels") else r["labels"]
        mask = r.get("label_mask") or {}
        feats.append((e["input_ids"], e["token_type_ids"],
                      [float(src[l]) for l in LABELS], [float(r["labels"][l]) for l in LABELS],
                      [float(mask.get(l, 1)) for l in LABELS]))
    return feats


def collate(batch):
    L = max(len(b[0]) for b in batch)
    ids = torch.zeros(len(batch), L, dtype=torch.long)
    tt = torch.zeros(len(batch), L, dtype=torch.long)
    am = torch.zeros(len(batch), L, dtype=torch.long)
    for i, (a, t, *_rest) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a)
        tt[i, :len(t)] = torch.tensor(t)
        am[i, :len(a)] = 1
    y = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    yb = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    mk = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    return ids, tt, am, y, yb, mk


@torch.no_grad()
def predict(model, feats, bs, device, fp16):
    model.eval()
    # sort by length for speed, restore order afterwards
    order = sorted(range(len(feats)), key=lambda i: len(feats[i][0]))
    out = np.zeros((len(feats), len(LABELS)), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        ids, tt, am, *_rest = collate([feats[i] for i in idx])
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
            logits = model(input_ids=ids.to(device), token_type_ids=tt.to(device),
                           attention_mask=am.to(device)).logits
        out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def best_f1(y, p):
    if y.sum() == 0:
        return 0.0
    pr, rc, _ = precision_recall_curve(y, p)
    f = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    return float(f.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "train.yaml"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: limit train records")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    run_id = args.run_id or datetime.now().strftime(cfg.get("run_prefix", "standin") + "-%Y%m%d-%H%M")
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "train.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = bool(cfg["fp16"]) and device.type == "cuda"
    print(f"[04] device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}) fp16={fp16}")

    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    tok = tg_format.get_tokenizer()
    t0 = time.time()
    train_recs = read_jsonl(PROCESSED / "train.jsonl")
    if args.limit:
        train_recs = train_recs[:args.limit]
    val_recs = read_jsonl(PROCESSED / "val.jsonl")
    train = encode_records(train_recs, tok, cfg["max_len"], cfg["soft_targets"])
    val = encode_records(val_recs, tok, cfg["max_len"], cfg["soft_targets"])
    print(f"[04] encoded train={len(train):,} val={len(val):,} in {time.time() - t0:.0f}s")

    yb = np.array([f[3] for f in train])
    mk = np.array([f[4] for f in train])
    P = (yb * mk).sum(0)
    N = mk.sum(0) - P
    pos_weight = np.where(P > 0, np.minimum(N / np.maximum(P, 1), cfg["pos_weight_cap"]), 1.0)
    print("[04] pos_weight", dict(zip(LABELS, np.round(pos_weight, 2).tolist())))

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"], num_labels=len(LABELS), problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    assert model.config.vocab_size == len(tok.vocab) == 30522, "expected BERT uncased vocab (30522)"
    model.to(device)

    loader = torch.utils.data.DataLoader(train, batch_size=cfg["batch_size"], shuffle=True,
                                         collate_fn=collate, num_workers=0, drop_last=False)
    steps = len(loader) * cfg["epochs"]
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]))
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_frac"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device),
                                         reduction="none")

    best, history = -1.0, []
    yval = np.array([f[3] for f in val])
    mval = np.array([f[4] for f in val])
    val_src = np.array([r.get("source", "all") for r in val_recs])
    for ep in range(cfg["epochs"]):
        model.train()
        t_ep, run_loss = time.time(), 0.0
        for step, (ids, tt, am, y, _, mask) in enumerate(loader):
            ids, tt, am, y, mask = ids.to(device), tt.to(device), am.to(device), y.to(device), mask.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                logits = model(input_ids=ids, token_type_ids=tt, attention_mask=am).logits
            loss = (loss_fn(logits.float(), y) * mask).sum() / mask.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            if (step + 1) % 500 == 0:
                el = time.time() - t_ep
                print(f"[04] ep{ep + 1} step {step + 1}/{len(loader)} loss {run_loss / 500:.4f} "
                      f"({el / (step + 1) * 1000:.0f} ms/step)", flush=True)
                run_loss = 0.0
        pv = predict(model, val, cfg["eval_batch_size"], device, fp16)
        f1s = {l: round(best_f1(yval[mval[:, i] > 0, i], pv[mval[:, i] > 0, i]), 4) for i, l in enumerate(LABELS)}
        # Sources with no risky val examples have undefined F1; leave them out of the selection mean.
        by_src = {str(s): round(best_f1(yval[val_src == s, 0], pv[val_src == s, 0]), 4)
                  for s in sorted(set(val_src)) if yval[val_src == s, 0].sum() > 0}
        score = float(np.mean(list(by_src.values()))) if len(by_src) > 1 else f1s["risk"]
        history.append({"epoch": ep + 1, "val_best_f1": f1s, "val_risk_f1_by_source": by_src,
                        "select_score": round(score, 4), "minutes": round((time.time() - t_ep) / 60, 2)})
        print(f"[04] epoch {ep + 1} done in {(time.time() - t_ep) / 60:.1f} min; val best-F1 {f1s}; "
              f"risk F1 by source {by_src}; select {score:.4f}", flush=True)
        if score > best:
            best = score
            model.save_pretrained(run_dir / "checkpoint")
            tok.save_vocabulary(str(run_dir / "checkpoint"))
            print(f"[04] saved checkpoint (select score {best:.4f})")
    (run_dir / "train_history.json").write_text(json.dumps(
        {"history": history, "pos_weight": dict(zip(LABELS, pos_weight.tolist())),
         "n_train": len(train), "n_val": len(val), "total_minutes": round((time.time() - t0) / 60, 1),
         "device": str(device), "torch": torch.__version__}, indent=2), encoding="utf-8")
    print(f"[04] done -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
