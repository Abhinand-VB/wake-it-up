"""Export candidate base models to int8 ONNX (same recipe as stages 06/07) and time them on the PC CPU.

Speed only: heads are untrained (7 random outputs). Used to choose a Phase 2 base model
under the on-device latency budget (SOURCE_OF_TRUTH.md §6.1, §9).

Output: runs/bench-candidates/<name>.int8.onnx + pc_latency.json
Usage:  .venv/Scripts/python.exe scripts/bench_export_candidates.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "bench-candidates"
NAMES = ["input_ids", "attention_mask", "token_type_ids"]

CANDIDATES = {
    "minilm-l6": "nreimers/MiniLM-L6-H384-uncased",
    "all-minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "deberta-v3-xsmall": "microsoft/deberta-v3-xsmall",
    "deberta-v3-small": "microsoft/deberta-v3-small",
    "deberta-v3-base": "microsoft/deberta-v3-base",
}
LENGTHS = [32, 64, 128]


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits


def export(name: str, model_id: str) -> Path:
    int8 = OUT / f"{name}.int8.onnx"
    if int8.exists():
        return int8
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=7, attn_implementation="eager").float().eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    dummy = tuple(torch.ones((1, 16), dtype=torch.long) for _ in NAMES)
    fp32 = OUT / f"{name}.fp32.onnx"
    dyn = {n: {0: "batch", 1: "seq"} for n in NAMES} | {"logits": {0: "batch"}}
    torch.onnx.export(Wrapper(model), dummy, str(fp32), input_names=NAMES, output_names=["logits"],
                      dynamic_axes=dyn, opset_version=17, do_constant_folding=True, dynamo=False)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8, per_channel=False)
    fp32.unlink()
    for extra in OUT.glob(f"{name}.fp32.onnx*"):
        extra.unlink()
    print(f"  exported {name}: {params:.0f}M params, int8 {int8.stat().st_size / 1e6:.0f} MB", flush=True)
    return int8


def bench(path: Path, threads: int, length: int, runs: int = 60) -> float:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    wanted = {i.name for i in sess.get_inputs()}
    rng = np.random.default_rng(0)
    feeds = {
        "input_ids": rng.integers(1000, 29000, size=(1, length), dtype=np.int64),
        "attention_mask": np.ones((1, length), dtype=np.int64),
        "token_type_ids": np.zeros((1, length), dtype=np.int64),
    }
    feeds = {k: v for k, v in feeds.items() if k in wanted}
    for _ in range(5):
        sess.run(None, feeds)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, model_id in CANDIDATES.items():
        try:
            path = export(name, model_id)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: export failed: {type(e).__name__}: {e}")
            continue
        row = {"model_id": model_id, "int8_mb": round(path.stat().st_size / 1e6, 1)}
        for threads in (1, 2):
            for length in LENGTHS:
                row[f"t{threads}_len{length}_ms"] = round(bench(path, threads, length), 2)
        results[name] = row
        print(f"  {name:18s} {row['int8_mb']:6.0f} MB | 2 threads: " +
              "  ".join(f"{L} tok {row[f't2_len{L}_ms']:6.1f} ms" for L in LENGTHS) +
              " | 1 thread @128: " + f"{row['t1_len128_ms']:.1f} ms", flush=True)
    (OUT / "pc_latency.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
