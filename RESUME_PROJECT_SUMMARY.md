# Wake It Up: on-device AI tone checker for Android

**Type:** Personal project (Android app + ML training pipeline)
**Working name:** ToneGuard (renamed to Wake It Up on 2026-10-06, per `SOURCE_OF_TRUTH.md` header)
**Status:** Working prototype, sideloaded on a Samsung Galaxy S25. Not published. Accuracy release gates not yet met.

Evidence sources used below: `SOURCE_OF_TRUTH.md` (design doc and status log), `training/runs/*/report.md`, `training/runs/*/eval.json`, `robustness.json`, `quant_eval.json`, `training/data/processed_p2/stats.json`, `training/data/raw/crosscheck/agreement.json`, `android/core/build/test-results/`, and the Kotlin/Python source.

---

## 1. One-line pitch

An Android app that reads the message you're typing in a chat app (Discord first), looks at the last few messages, and shows a small "Might land badly" warning before you send something likely to be misread or offensive. A fine-tuned 33M-parameter transformer runs the check on the phone in about 4 ms, and nothing leaves the device.

## 2. Problem and approach

In fast group chats, a joke that riffs on someone else's words can read as endorsing something harmful once it's out of context. The intent is harmless but the group reads it differently, and a private heads-up before sending would prevent it. Wake It Up uses an Android AccessibilityService to read the user's own draft plus the visible conversation, runs a small multi-label classifier on-device, and shows a non-blocking overlay. It warns and never blocks, never sends anything for the user, and the app has no internet permission. The classifier was trained on public toxicity data plus about 5,400 LLM-generated, LLM-labeled group-chat examples built around the cases that public data misses (context-dependent riffs, banter, sarcasm).

## 3. Architecture

```
Discord (or another enabled app)
   │  accessibility events (draft text, window content)
   ▼
WakeItUpAccessibilityService  (app/service)      main thread: event filtering, password fields skipped
   │
   ├─► context thread: AppAdapter (DiscordAdapter / GenericAdapter) + ChatLayoutParser (core)
   │       walk node tree → ordered messages → drop timestamps/UI → detect "Replying to" →
   │       rename speakers ME, A, B… → ContextCache
   │
   └─► inference thread: ToneEngine (core)
           QuickFilter + Wordlist (instant slur hits, no model call)
           InputEncoder + WordPieceTokenizer (context | draft sentence pair, max 128 tokens)
           OnnxClassifier (ONNX Runtime, int8 model.onnx, 7 sigmoid outputs)
           DecisionEngine (per-label thresholds from model_config.json, sensitivity preset, hysteresis)
   │
   ▼
OverlayController (TYPE_ACCESSIBILITY_OVERLAY pill above the input, 300 ms delay, tap for explanation card)
```

- **`:core`** (pure Kotlin/JVM, unit-tested on the PC): tokenizer, encoder, classifier wrapper, decision logic, chat layout parser, explanations.
- **`:app`** (Android only): service, adapters, overlay, settings, setup flow, test playground and benchmark screen.
- **Model contract:** `model.onnx` + `vocab.txt` + `model_config.json` with `contract_version: 1`. The training pipeline produces this bundle and the app loads it from assets; an unsupported version makes the app stay silent instead of guessing.
- **Training pipeline** (`training/scripts/`): `01_download_public` → `02a_select_relabel` → `03b_build_phase2` → `04_train` → `05b_eval_phase2` → `05c_robustness` → `06_export_onnx` → `07_quantize` → `08_export_fixtures`. Fixtures feed the Kotlin parity tests.

## 4. Tech stack

- **Android:** Kotlin 2.2.20, Gradle Kotlin DSL, AGP 8.12.1, minSdk 26 / targetSdk 36, kotlinx-coroutines 1.10.2, AccessibilityService, WindowManager overlays, Views built in code (no XML layouts).
- **On-device ML:** ONNX Runtime 1.30.0 (`onnxruntime-android` in the app, desktop `onnxruntime` for `:core` tests), int8 quantized model.
- **Training:** Python, PyTorch (CUDA, RTX 2060), Hugging Face Transformers, ONNX export and ONNX Runtime quantization. Base model `bge-small-en-v1.5`.
- **Data:** civil_comments (public), LLM-generated synthetic chat data (Claude), 275-case hand-curated golden set.
- **Testing:** JUnit via kotlin-test, Python-generated fixtures for tokenizer, encoder and model parity.

## 5. My contribution

- **No git repository**, so there's no commit history to split authorship. UNVERIFIED: solo project, and how much of the code was written with an AI coding assistant.
- **Borrowed, not built:** the base model (`bge-small-en-v1.5`, BAAI), ONNX Runtime, PyTorch/Transformers, the civil_comments dataset, and the BERT WordPiece algorithm (re-implemented in Kotlin, not invented).
- **LLM-generated data:** the 5,406 synthetic records were generated and labeled by Claude ("in-session by 9 parallel agents" for round 1, `SOURCE_OF_TRUTH.md` §Phase 2 status log). The 2,000 relabeled civil_comments rows and the 306-record cross-check were also Claude-labeled. No human annotators besides the golden set.

## 6. Hard engineering problems

1. **The model learned a shortcut.** Round 1 flagged almost any reply in a sensitive conversation ("lol same" scored 0.93). Diagnosis: training contexts were mostly one message, there were only 75 short drafts, and few kind replies in sensitive threads. Fix: 1,297 targeted records (kind/mocking replies in sensitive contexts, short reactions, long multi-speaker threads), bringing short drafts to 537. "lol same" fell to 0.02 and golden false alarms dropped from 20% to 11.5%. Evidence: `SOURCE_OF_TRUTH.md` round 1/round 2 log, `training/runs/p2-bge-v1/report.md`, `training/runs/p2-bge-v2/report.md`.
2. **Fragile to punctuation.** Removing one comma from the context moved a risk score from 73% to 35%. Added noisy-surface augmentation to upsampled repeats in `training/scripts/03b_build_phase2.py` (dropped punctuation, lowercasing, typos) and a new robustness stage `05c_robustness.py`. The same case now scores 94% either way; decision flip rate under noise fell from 4.48% to 3.64% (`training/runs/p2-bge-v2/robustness.json`, `training/runs/p2-bge-v3/robustness.json`).
3. **Quantization hurt accuracy.** Compared 5 variants on the golden set. Quantizing the 30k×384 embedding table did most of the damage (all-ops per-tensor changed 11 of 275 decisions). Shipped MatMul-only per-channel QUInt8: 133.65 MB → 69.91 MB, 1 decision changed out of 275 (`training/runs/p2-bge-v2/export/quant_eval.json`, `SOURCE_OF_TRUTH.md` quantization table, `training/scripts/07_quantize.py`).
4. **Python and Kotlin had to agree exactly.** Wrote a pure-Kotlin WordPiece tokenizer (`android/core/.../WordPieceTokenizer.kt`) and encoder, then tested them against fixtures exported from Python: 113/113 tokenizer and 32/32 encoder exact matches, model probabilities within 5e-7. Found that ONNX Runtime 1.22 on the app side made int8 outputs drift by up to 0.04; pinned both sides to 1.30.0 through `wakeitup.onnxruntimeVersion` in `android/gradle.properties`.
5. **Fitting on a phone.** Benchmarked 6 candidate models on the PC and the S25 (`training/scripts/bench_export_candidates.py`, `training/runs/bench-candidates/`). Speed was fine for all up to DeBERTa-v3-small; the real limit was memory (DeBERTa-v3-base OOMed on load with a 256 MB heap). Also fixed a cold-start issue: warming up with a tiny input left the first real check at about 170 ms, so warm-up now runs a near-max-length input.

## 7. Metrics

| Metric | Number | How measured | Source |
|---|---|---|---|
| On-device latency, stand-in MiniLM-L6 | p50 3.53 ms, p95 3.56 ms at 65 tokens | 200-run benchmark in the app's test screen, Galaxy S25, 2026-10-06 | `SOURCE_OF_TRUTH.md` Phase 1 status |
| On-device latency, bge-small int8 (v2 bundle, same size and architecture as shipped v3) | 4.5 ms per check at 19 tokens; load 354 ms, warm-up 30 ms | Test screen on the S25, 2026-10-07 | `SOURCE_OF_TRUTH.md` Phase 2 log |
| Golden-set recall on should-warn cases (n=275, 119 should-warn) | 19.3% stand-in → 84.0% v2 → 79.0% v3 (shipped). Gate ≥ 85%, **not met** | `05b_eval_phase2.py` | `runs/standin-2026.10.0/golden_baseline.json`, `runs/p2-bge-v2/report.md`, `runs/p2-bge-v3/report.md` |
| Golden-set false-alarm rate | 16.7% stand-in, 19.9% v1 → 11.5% v2, 12.2% v3. Gate ≤ 5%, **not met** | same | same, plus `runs/p2-bge-v1/report.md` |
| Core incident family (10 cases) | 6/10 stand-in → 9/10 v2 | golden-set subset | `SOURCE_OF_TRUTH.md` round 2 table |
| Probe sentences (14) | 6 → 13 (v2); 12 (v3) | probe table in eval | `runs/p2-bge-v2/report.md`, `runs/p2-bge-v3/report.md` |
| Synthetic test-set risk F1 | 0.89 (v1), 0.83 (v2), 0.85 (v3) | held-out synthetic test split (n=409) | `runs/*/report.md`, `SOURCE_OF_TRUTH.md` |
| Golden AUC | 0.917 (v2), 0.912 (v3) | same eval | `SOURCE_OF_TRUTH.md` robustness table |
| Labeling consistency | 97.7% binary agreement, Cohen's κ 0.95, n=306 | blind re-label, both labelers Claude | `data/raw/crosscheck/agreement.json` |
| Python ↔ Kotlin parity | worst abs diff 5e-7 | `ModelParityTest` with desktop ORT | `SOURCE_OF_TRUTH.md`, `android/core/src/test/` |
| Unit tests | 27/27 pass (6 suites) | Gradle `:core:test`, 2026-10-07 | `android/core/build/test-results/test/*.xml` |
| Model size | 133.65 MB fp32 → 69.91 MB int8 | file size | `runs/p2-bge-v2/export/quant_eval.json` |
| Training set | 59,164 records (34,000 civil_comments, 18,364 synthetic incl. upsampled repeats, 6,800 relabeled incl. repeats) | dataset build | `data/processed_p2/stats.json` |
| Code size | about 2,050 lines of Kotlin (1,269 app, 778 core) plus 409 test lines; about 1,960 lines of Python in `training/` | `wc -l` | source tree |

## 8. Status and honest limits

- Prototype. Sideloaded on my own Galaxy S25, not on Google Play, no outside users. UNVERIFIED: whether anyone else has used it.
- Accuracy gates are not met. Best numbers: 84% recall at 11.5% false alarms (v2) and 79% at 12.2% (v3, shipped because it's more robust to messy text and has better recall at a 5% false-alarm target).
- The golden set guided round 2 data, so it's now a dev set, not a blind test. A fresh held-out set hasn't been written yet.
- Synthetic training data was generated and labeled by an LLM (Claude), not human annotators. The 97.7% agreement figure is Claude vs. Claude.
- The Discord adapter still needs tuning on the real device, and the manual QA checklist (§12.5) hasn't been run on the phone yet (last status in `SOURCE_OF_TRUTH.md`).
- Latency on mid-range and budget phones is estimated, not measured.
- English only. Android only (iOS has no equivalent API).

## 9. Dates

Oct 2026 (no git history; dates from the design doc log and file timestamps: design approved 2026-10-04, v3 model bundled 2026-10-07, last file change 2026-10-09). UNVERIFIED: whether work is ongoing ("Oct 2026 - present") or paused.

## 10. Links

- Repo: none found (not a git repository, no remote). UNVERIFIED.
- Demo: none found. UNVERIFIED.

## 11. Resume bullets

- Built an Android app that warns users before they send a chat message likely to be misread, using an AccessibilityService, a floating overlay, and per-app conversation parsing; the app runs fully on-device and requests no permissions, not even internet.
- Fine-tuned and int8-quantized a 33M-parameter transformer (bge-small) for context-aware, 7-label tone classification, running in about 4.5 ms per check on a Galaxy S25 with ONNX Runtime.
- Trained on 34k public records plus 5.4k LLM-generated chat examples; diagnosed a context shortcut and fixed it with targeted data, cutting golden-set false alarms from 20% to 11.5% while raising recall from 19% to 84%.
- Wrote a pure-Kotlin WordPiece tokenizer and a versioned model contract, verified against the Python reference with parity tests (max error 5e-7), and halved model size (134 to 70 MB) with 1 of 275 decisions changed.

## 12. Best-fit roles

- **On-device / edge ML engineer:** model selection under phone memory limits, quantization study, ONNX Runtime on Android, latency benchmarks on real hardware.
- **Applied ML / NLP engineer:** data design, LLM-based data generation, shortcut diagnosis, robustness testing, golden-set evaluation with honest gates.
- **Android engineer:** AccessibilityService, overlays, threading, privacy-by-design manifest, setup flow for Samsung quirks.
- **ML platform / MLOps (junior):** staged re-runnable pipeline, versioned model contract, cross-language parity tests.

## 13. Interview story (STAR+R)

- **Situation:** In a group chat, a joke riffing on a friend's word ("slave away" turned into "I like slaves") read as endorsing slavery. The intent was harmless, but nothing warned me before I sent it.
- **Task:** Build a private, on-device heads-up that understands the conversation, not just the draft, without sending chats to a server.
- **Action:** Built an AccessibilityService app with a non-blocking overlay and a pure-Kotlin tokenizer, then trained a 33M-parameter classifier on public data plus LLM-generated chat examples. When round 1 flagged almost any reply in a sensitive conversation, I traced it to the data (one-message contexts, very few short drafts) and generated 1,297 targeted examples. When a single comma swung a score from 73% to 35%, I added noise augmentation and a robustness test stage.
- **Result:** Golden-set recall went from 19% to 84% and false alarms from 20% to 11.5%; the comma case now scores 94% either way; checks take about 4.5 ms on a Galaxy S25. The release gates (85% recall, 5% false alarms) are still not met.
- **Reflection:** Evaluation design mattered more than model choice. Once I used the golden set to steer data, it stopped being a blind test, so the next step is a fresh held-out set and more human-reviewed labels rather than a bigger model.

## 14. Open questions

1. **Authorship:** solo project? How much of the code and design was written with an AI coding assistant (e.g. Claude Code)? The design doc says round 1 synthetic data was generated "in-session by 9 parallel agents".
2. **Repo / demo links:** is it on GitHub (public or private)? Any demo video or screenshots?
3. **Dates:** still in progress ("Oct 2026 - present"), or paused?
4. **Users:** anyone besides you (friends, testers)?
5. **Since 2026-10-09:** has the Discord adapter been tuned on the phone, or the QA checklist run?
6. **Shortcut example number:** the old summary said the coming-out reply went 88% → 4%. The design doc gives 88% in the round 1 diagnosis but "0.58 → 0.04" in the round 2 table, so the starting number is inconsistent. I used the "lol same" example (0.93 → 0.02) instead, which is consistent.
7. **Old summary corrections (FYI):** synthetic test F1 was "0.85-0.89"; actual range across v1-v3 is 0.83-0.89. Code size grew from about 1,800 to about 2,050 Kotlin lines.
