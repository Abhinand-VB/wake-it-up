"""Stage 06: export fp32 ONNX -> runs/<id>/export/model.onnx (+ vocab.txt)

Inputs : input_ids, attention_mask, token_type_ids   int64 [batch, L] (dynamic; app uses batch=1)
Output : logits                                       float32 [batch, 7]
Verifies ONNX Runtime output vs PyTorch on a set of encoded examples.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import tg_format  # noqa: E402
from tg_common import PROCESSED, read_jsonl, record_inputs, resolve_run  # noqa: E402

OPSET = 17


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits


def sample_inputs(tok, n=200):
    encs = [tg_format.encode([], None, "hello there"),
            tg_format.encode([{"speaker": "A", "text": "x " * 300}], None, "y " * 300),
            tg_format.encode([], None, "")]
    for r in read_jsonl(PROCESSED / "test.jsonl")[:n]:
        encs.append(tg_format.encode(*record_inputs(r), tok=tok))
    return encs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = resolve_run(args.run)
    out = run / "export"
    out.mkdir(exist_ok=True)

    from transformers import AutoModelForSequenceClassification
    import onnx
    import onnxruntime as ort

    model = AutoModelForSequenceClassification.from_pretrained(
        run / "checkpoint", attn_implementation="eager").eval()
    w = Wrapper(model).eval()
    tok = tg_format.get_tokenizer()
    e = tg_format.encode([{"speaker": "A", "text": "anyone up for food later"}], 0, "only if you pay lol")
    dummy = tuple(torch.tensor([e[k]], dtype=torch.long) for k in ("input_ids", "attention_mask", "token_type_ids"))
    path = out / "model.onnx"
    names = ["input_ids", "attention_mask", "token_type_ids"]
    dyn = {n: {0: "batch", 1: "seq"} for n in names}
    dyn["logits"] = {0: "batch"}
    try:
        torch.onnx.export(w, dummy, str(path), input_names=names, output_names=["logits"],
                          dynamic_axes=dyn, opset_version=OPSET, do_constant_folding=True, dynamo=False)
        exporter = "torchscript"
    except Exception as ex:  # legacy exporter unavailable -> dynamo exporter
        print(f"[06] legacy exporter failed ({ex!r}); trying dynamo exporter")
        B, S = torch.export.Dim("batch"), torch.export.Dim("seq", max=512)
        shapes = {n: {0: B, 1: S} for n in names}
        prog = torch.onnx.export(w, dummy, input_names=names, output_names=["logits"],
                                 dynamic_shapes=shapes, opset_version=max(OPSET, 18), dynamo=True)
        prog.save(str(path))
        exporter = "dynamo"

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    print(f"[06] exported ({exporter}) opset={m.opset_import[0].version} -> {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)")
    for i in m.graph.input:
        print("    in ", i.name, i.type.tensor_type.elem_type, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim])
    for o in m.graph.output:
        print("    out", o.name, o.type.tensor_type.elem_type, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim])

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    with torch.no_grad():
        for enc in sample_inputs(tok):
            feeds = {k: np.array([enc[k]], dtype=np.int64) for k in names}
            o = sess.run(["logits"], feeds)[0]
            assert o.shape == (1, len(tg_format.LABELS)) and o.dtype == np.float32
            t = w(*(torch.tensor(feeds[k]) for k in names)).numpy()
            max_diff = max(max_diff, float(np.abs(o - t).max()))
    print(f"[06] ONNX vs torch max |logit diff| = {max_diff:.2e}")
    assert max_diff < 1e-3, "ONNX export mismatch"
    shutil.copy(run / "checkpoint" / "vocab.txt", out / "vocab.txt")
    (out / "export_check.txt").write_text(f"exporter={exporter}\nopset={m.opset_import[0].version}\n"
                                          f"max_abs_logit_diff={max_diff:.3e}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
