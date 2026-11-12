"""Stage 07: dynamic int8 quantization -> runs/<id>/export/model.int8.onnx
Then assembles the §7 bundle (runs/<id>/export/bundle/) and, unless --no-install,
copies it to android/app/src/main/assets/model/ (model.onnx = the int8 model).

* Weights are quantized as QUInt8 (U8U8 kernels). On x86 CPUs without AVX-VNNI
  (e.g. the dev PC's Ryzen 4800H) ORT's U8S8 path can saturate (VPMADDUBSW), which
  would make PC-generated parity fixtures differ from ARM devices; U8U8 avoids that.
* Both fp32 and int8 are evaluated on the test set with batch size 1 (as on device;
  dynamic activation quantization ranges depend on the whole batch).
* CPU latency (informational): 128-token input, p50/p95 over 200 runs.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402

import tg_format  # noqa: E402
from tg_common import ANDROID_MODEL_DIR, MODEL_VERSION, PROCESSED, read_jsonl, record_inputs, resolve_run  # noqa: E402

LABELS = tg_format.LABELS
NAMES = ["input_ids", "attention_mask", "token_type_ids"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_probs(sess, encs):
    out = np.zeros((len(encs), len(LABELS)), dtype=np.float32)
    for i, e in enumerate(encs):
        feeds = {k: np.array([e[k]], dtype=np.int64) for k in NAMES}
        out[i] = sigmoid(sess.run(["logits"], feeds)[0][0])
    return out


def label_metrics(y, p, th):
    res = {}
    for i, l in enumerate(LABELS):
        pred = (p[:, i] >= th[l]).astype(int)
        res[l] = {"f1": round(float(f1_score(y[:, i], pred, zero_division=0)), 4),
                  "precision": round(float(precision_score(y[:, i], pred, zero_division=0)), 4),
                  "recall": round(float(recall_score(y[:, i], pred, zero_division=0)), 4)}
    pr = (p[:, 0] >= th["risk"]).astype(int)
    res["risk"]["macro_f1"] = round(float(f1_score(y[:, 0], pr, average="macro")), 4)
    res["risk"]["fpr"] = round(float(pr[y[:, 0] == 0].mean()), 4)
    return res


def latency(sess, enc, n=200):
    feeds = {k: np.array([enc[k]], dtype=np.int64) for k in NAMES}
    for _ in range(20):
        sess.run(["logits"], feeds)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(["logits"], feeds)
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 2), "p95_ms": round(float(np.percentile(ts, 95)), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--no-install", action="store_true")
    args = ap.parse_args()
    run = resolve_run(args.run)
    exp = run / "export"
    fp32, int8 = exp / "model.onnx", exp / "model.int8.onnx"

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = fp32
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = exp / "model.pre.onnx"
        quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
        src = pre
    except Exception as ex:
        print(f"[07] quant_pre_process skipped: {ex!r}")
    # MatMul-only, per-channel (Phase 2 decision, SOURCE_OF_TRUTH.md §6.4): quantizing the word-embedding
    # Gather as well (per-tensor, as in Phase 1) changed 11/275 golden decisions and cut recall 84% -> 77%;
    # MatMul-only per-channel changed 1/275, at the cost of a ~70 MB file (embeddings stay fp32).
    quantize_dynamic(str(src), str(int8), weight_type=QuantType.QUInt8, per_channel=True,
                     op_types_to_quantize=["MatMul"])
    print(f"[07] fp32 {fp32.stat().st_size / 1e6:.1f} MB -> int8 {int8.stat().st_size / 1e6:.1f} MB")

    th = json.loads((run / "thresholds.json").read_text(encoding="utf-8"))
    recs = read_jsonl(PROCESSED / "test.jsonl")
    tok = tg_format.get_tokenizer()
    encs = [tg_format.encode(*record_inputs(r), tok=tok) for r in recs]
    y = np.array([[r["labels"][l] for l in LABELS] for r in recs])

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2  # same as the app (§5.7)
    s32 = ort.InferenceSession(str(fp32), so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(int8), so, providers=["CPUExecutionProvider"])
    t = time.time()
    p32 = run_probs(s32, encs)
    p8 = run_probs(s8, encs)
    print(f"[07] evaluated {len(encs)} test cases x2 in {time.time() - t:.0f}s")
    m32, m8 = label_metrics(y, p32, th), label_metrics(y, p8, th)
    drop = round((m32["risk"]["f1"] - m8["risk"]["f1"]) * 100, 2)

    long_draft = " ".join(["word"] * 200)
    e128 = tg_format.encode([{"speaker": "A", "text": "hey"}], None, long_draft)
    assert len(e128["input_ids"]) == 128
    lat = {"fp32_threads2": latency(s32, e128), "int8_threads2": latency(s8, e128)}
    so1 = ort.SessionOptions(); so1.intra_op_num_threads = 1
    lat["int8_threads1"] = latency(ort.InferenceSession(str(int8), so1, providers=["CPUExecutionProvider"]), e128)

    q = {"int8_size_mb": round(int8.stat().st_size / 1e6, 2), "fp32_size_mb": round(fp32.stat().st_size / 1e6, 2),
         "weight_type": "QUInt8 per-channel, MatMul only", "test_fp32": m32, "test_int8": m8, "risk_f1_drop_points": drop,
         "max_abs_prob_diff_int8_vs_fp32": round(float(np.abs(p8 - p32).max()), 4),
         "mean_abs_prob_diff_int8_vs_fp32": round(float(np.abs(p8 - p32).mean()), 5),
         "pc_cpu_latency_128tok": lat, "ort_version": ort.__version__}
    (exp / "quant_eval.json").write_text(json.dumps(q, indent=2), encoding="utf-8")
    print(json.dumps(q, indent=2))

    # ---- assemble §7 bundle
    ev_all = json.loads((run / "eval.json").read_text(encoding="utf-8"))
    ev = ev_all["test"]["risk"]
    golden = ev_all.get("golden") or {}
    base_model = "MiniLM-L6-H384-uncased"
    if (run / "train.yaml").exists():
        import yaml
        base_model = str(yaml.safe_load(open(run / "train.yaml", encoding="utf-8"))["base_model"]).split("/")[-1]
    notes = (f"Phase 1 draft-only stand-in fine-tuned on civil_comments (no conversational context; "
             f"golden set not evaluated, golden_* fields are placeholders). Test set (n={len(y)}, int8): "
             f"risk recall={m8['risk']['recall']:.3f}, FPR={m8['risk']['fpr']:.3f}, F1={m8['risk']['f1']:.3f}, "
             f"macro-F1={m8['risk']['macro_f1']:.3f}, AUC(fp32)={ev['auc']}. int8 vs fp32 risk F1 drop={drop:.2f} pts. "
             f"hate_slur and insensitive both proxy identity_attack; misread_risk untrained (threshold 0.99).")
    if golden:
        notes = (f"Phase 2 model ({base_model}) trained on synthetic chat data + cleaned/re-labeled civil_comments "
                 f"(SOURCE_OF_TRUTH.md D9/D10). Golden set n={golden['n']}: recall {golden['recall_risky']:.3f}, "
                 f"false alarms {golden['fpr_benign']:.3f}. Test (int8): risk recall={m8['risk']['recall']:.3f}, "
                 f"FPR={m8['risk']['fpr']:.3f}. int8 vs fp32 risk F1 drop={drop:.2f} pts.")
    cfg = {
        "contract_version": 1,
        "model_version": MODEL_VERSION,
        "base_model": base_model,
        "do_lower_case": True,
        "max_len": tg_format.MAX_LEN,
        "context_max_messages": tg_format.CONTEXT_MAX_MESSAGES,
        "context_separator": tg_format.CONTEXT_SEPARATOR,
        "reply_prefix": tg_format.REPLY_PREFIX,
        "empty_context_token": tg_format.EMPTY_CONTEXT_TOKEN,
        "labels": LABELS,
        "thresholds": {l: round(float(th[l]), 3) for l in LABELS},
        "eval": {"golden_recall_risky": golden.get("recall_risky", 0.0),
                 "golden_fpr_benign": golden.get("fpr_benign", 0.0), "notes": notes},
    }
    bundle = exp / "bundle"
    bundle.mkdir(exist_ok=True)
    shutil.copy(int8, bundle / "model.onnx")
    shutil.copy(exp / "vocab.txt", bundle / "vocab.txt")
    with open(bundle / "model_config.json", "w", encoding="utf-8", newline="") as f:  # LF endings
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write(chr(10))
    print(f"[07] bundle -> {bundle}")
    if not args.no_install:
        ANDROID_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for f in ("model.onnx", "vocab.txt", "model_config.json"):
            shutil.copy(bundle / f, ANDROID_MODEL_DIR / f)
        print(f"[07] installed -> {ANDROID_MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
