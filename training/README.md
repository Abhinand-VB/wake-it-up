# WakeItUp training pipeline

Offline pipeline that produces the model bundle consumed by the Android app
(`SOURCE_OF_TRUTH.md` §7, §8). The current output is the **Phase 1 stand-in**: a
draft-only MiniLM-L6 classifier fine-tuned on `google/civil_comments`
(context segment is always `none`).

## Setup (Windows, Git Bash)

Use Python 3.13 (not 3.14) and a CUDA build of torch (RTX 2060 = sm_75, cu126 wheels):

```bash
cd training
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt   # torch line uses the cu126 index (see file)
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"   # must print True
```

`requirements.txt` holds an `--extra-index-url https://download.pytorch.org/whl/cu126`
line so the `+cu126` torch wheel resolves.

## Run (from `training/`)

```bash
PY=.venv/Scripts/python.exe
$PY tg_format.py                      # encoder self-test (§6.2.1)
$PY scripts/01_download_public.py     # -> data/raw/public/civil_comments/{train,validation,test}.parquet
$PY scripts/03_build_dataset.py       # -> data/processed/{train,val,test}.jsonl (+ stats.json)
$PY scripts/04_train.py               # -> runs/<id>/checkpoint  (config: configs/train.yaml; ~20-40 min on a 2060)
$PY scripts/05_eval.py                # -> runs/<id>/report.md, thresholds.json, eval.json
$PY scripts/06_export_onnx.py         # -> runs/<id>/export/model.onnx (fp32) + vocab.txt
$PY scripts/07_quantize.py            # -> runs/<id>/export/model.int8.onnx, quant_eval.json, bundle/
                                      #    and installs bundle into android/app/src/main/assets/model/
$PY scripts/08_export_fixtures.py     # -> android/app/src/test/resources/fixtures/*.json
```

Scripts 05-07 take `--run <id>` (default: the most recent run with a checkpoint).
`07_quantize.py --no-install` builds the bundle without copying it into the app.
`02_generate_synthetic.py` (LLM distillation, §8.4) does not exist yet (Phase 2).

## Outputs

| Path | Content |
|---|---|
| `data/raw/public/` | downloaded datasets (gitignored) |
| `data/processed/` | unified §8.2 JSONL; `labels` = binary (score >= 0.5), `soft_labels` = rater fractions used as training targets |
| `runs/<id>/checkpoint/` | HF checkpoint + vocab |
| `runs/<id>/report.md` | per-label test/val metrics, tuned thresholds |
| `runs/<id>/export/` | fp32 + int8 ONNX, `quant_eval.json` (int8 vs fp32, PC latency), `bundle/` |
| `android/app/src/main/assets/model/` | shipped bundle: `model.onnx` (int8), `vocab.txt`, `model_config.json` |
| `android/app/src/test/resources/fixtures/` | `wordpiece_fixtures.json`, `encoder_fixtures.json`, `parity_fixtures.json` |

## Modules

- `tg_format.py` — normative §6.2.1 encoder. `wordpiece()` = HF slow (pure-Python)
  BERT tokenizer (basic tokenizer + WordPiece, do_lower_case=True). In transformers >= 5 the
  slow class is `transformers.models.bert.tokenization_bert_legacy.BertTokenizerLegacy`
  (`transformers.BertTokenizer` is now the Rust-backed fast one). Literal `[CLS]`/`[SEP]`
  text in a message is tokenized as ordinary text, never as a special token.
  Vocab: `assets/vocab.txt` (vendored from `nreimers/MiniLM-L6-H384-uncased`; byte-identical
  to `bert-base-uncased`, 30522 entries).
- `tg_common.py` — shared paths, run resolution, JSONL helpers.

## Stand-in label mapping (civil_comments)

| WakeItUp label | source column |
|---|---|
| risk | toxicity |
| hate_slur | identity_attack |
| insensitive | identity_attack (weak proxy) |
| harassment | insult |
| threat | threat |
| sexual | sexual_explicit |
| misread_risk | always 0 (threshold 0.99 = disabled) |

## Notes

- Quantization uses `QUInt8` weights (U8U8). On x86 CPUs without AVX-VNNI, ORT's default
  U8S8 kernels can saturate, so PC-generated parity fixtures could differ from ARM devices.
- The ONNX graph has a dynamic batch axis as well as a dynamic sequence axis; the app feeds `[1, L]`.
