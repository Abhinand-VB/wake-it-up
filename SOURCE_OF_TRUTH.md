# Wake It Up — Source of Truth

> **Status:** Design approved, pre-implementation  
> **Last updated:** 2026-10-04  
> **App name:** Wake It Up (code identifiers: `WakeItUp`, package `dev.wakeitup`; renamed from the working name ToneGuard on 2026-10-06)

This document is the single reference for what Wake It Up is, why it is built the way it is, and how every part fits together. If something breaks, behaves strangely, or a decision needs revisiting, start here. When a decision changes, **update this file in the same change**.

---

## Table of contents

1. [Purpose](#1-purpose)
2. [Goals and non-goals](#2-goals-and-non-goals)
3. [Decision log](#3-decision-log)
4. [System overview](#4-system-overview)
5. [Android app — component design](#5-android-app--component-design)
6. [Model specification](#6-model-specification)
7. [The model contract (app ↔ training)](#7-the-model-contract-app--training)
8. [Training pipeline](#8-training-pipeline)
9. [Performance targets and latency budget](#9-performance-targets-and-latency-budget)
10. [Privacy and security](#10-privacy-and-security)
11. [Distribution and platform policy](#11-distribution-and-platform-policy)
12. [Testing strategy](#12-testing-strategy)
13. [Failure modes and troubleshooting runbook](#13-failure-modes-and-troubleshooting-runbook)
14. [Roadmap and phase acceptance criteria](#14-roadmap-and-phase-acceptance-criteria)
15. [Repository layout](#15-repository-layout)
16. [Open questions](#16-open-questions)
17. [Glossary](#17-glossary)

---

## 1. Purpose

Some people find it hard to predict how a message will be read by a group, especially when typing fast. A common pattern:

- Someone in a group chat uses a casual phrase.
- The user replies with a quick joke that builds on a word from that phrase.
- **Out of context, the reply reads as endorsing something harmful or insensitive.**
- Doubling down ironically makes it worse, because the group cannot tell it is irony.
- The group reacts badly, and the user feels piled on without understanding why.

The intent was harmless; the problem was **how the words read to others**. A small, private heads-up *before sending* ("this might come across as X") would have prevented it.

**Wake It Up is an assistive tool for social communication.** It watches the user's own drafts in messaging apps they choose, considers the recent conversation, and quietly warns when a message is likely to be misread or cause offense. It never blocks, never sends anything for the user, and never leaves the phone.

---

## 2. Goals and non-goals

### Goals

| # | Goal |
|---|---|
| G1 | Warn about risky drafts **before** they are sent, with a pill that appears within ~350 ms of a typing pause. |
| G2 | Judge drafts **in context** (the message being replied to and the last few messages), not as isolated text. |
| G3 | Run **entirely on-device**. No server, no API keys in the app, no message data leaves the phone. |
| G4 | Low false-positive rate. Normal banter and swearing between friends should **not** trigger warnings. |
| G5 | Scale to many users at zero per-check cost. |
| G6 | Be robust: when something fails, fail **silent and safe** (no warning shown), never crash the host app or block typing. |

### Non-goals (for now)

| # | Non-goal | Why |
|---|---|---|
| N1 | Blocking or intercepting the send button | Not possible reliably via accessibility APIs; also too intrusive. Wake It Up **warns**, the user decides. |
| N2 | Generating rewrites or free-text explanations | The tiny classifier cannot do this. Planned later via on-device LLM on tap (see §14). |
| N3 | Moderating other people's messages | Wake It Up only evaluates the user's own draft. Other messages are read only as context. |
| N4 | iOS support | iOS has no equivalent of Android accessibility services for third-party apps. |
| N5 | Modifying Discord or other apps (client mods, self-bots) | Violates those apps' Terms of Service and risks account bans. |

---

## 3. Decision log

Each decision records what was chosen, the alternatives considered, and why. Revisit a decision by adding a new entry, not by deleting the old one.

### D1 — Platform: Android app (not desktop, not a Discord bot)

- **Chosen:** Android app.
- **Alternatives:** Windows keyboard hook, browser extension, Discord client mod, Discord bot with slash command, Discord AutoMod.
- **Why:** The problem mostly happens when typing fast on a phone. Bots can only act *after* a message is posted. Client mods break Discord's ToS. AutoMod is server-wide keyword matching, which has no context and is only available to server admins.

### D2 — Interception point: Accessibility service + overlay (not a custom keyboard)

- **Chosen:** An `AccessibilityService` that reads the draft and the visible conversation, plus a floating overlay pill.
- **Alternatives:** Custom keyboard (IME), "process text" context-menu action, floating bubble, share sheet.
- **Why:** Only the accessibility service can see the **conversation context** (the IME sees only the text field). It also works with the user's existing keyboard. Manual-trigger options fail the core problem: when typing fast, users won't remember to invoke them.

### D3 — Inference location: On-device (not cloud LLM)

- **Chosen:** On-device model.
- **Alternatives:** (a) cloud LLM with API key embedded in app, (b) cloud LLM behind a project backend with auth and rate limits, (c) hybrid.
- **Why:** (a) is insecure: the key can be extracted from the APK, and it does not scale (each user would need a key). (b) works but costs money per check, adds network latency, and sends private conversations to a server. On-device is free per check, private, offline-capable, and simplifies privacy policy and store review. (c) remains a future option (§14).

### D4 — Model type: Tiny fine-tuned classifier (not a small generative LLM)

- **Chosen:** A ~22M-parameter transformer classifier (MiniLM family), fine-tuned, int8-quantized, run with ONNX Runtime.
- **Alternatives:** Gemini Nano (on-device via AICore), Gemma 3 1B / Qwen3 0.6–1.7B via MediaPipe/LiteRT or llama.cpp.
- **Why:** **Latency is the top priority.** A tiny classifier runs in roughly 5–20 ms on a flagship phone, versus ~0.3–3 s for generative models. It is ~25 MB instead of 0.5–2 GB. The tradeoff is no free-text explanations or rewrites; this is mitigated by category-based explanation templates (§5.8) and a planned LLM-on-tap feature.

### D5 — Context handling: Pair input (context, draft)

- **Chosen:** The model receives two segments: recent conversation context and the draft, encoded as a BERT-style sentence pair.
- **Why:** Many risky messages contain no offensive words at all. The risk comes from how the draft interacts with the preceding message (e.g., riffing on a word in a way that changes its meaning). A draft-only classifier cannot see this.

### D6 — Teaching a small model a nuanced task: Distillation

- **Chosen:** Use a large model (Claude) **once, offline** to generate and label context-rich training examples, then fine-tune the tiny classifier on them together with public datasets.
- **Why:** Public toxicity datasets are mostly single comments without conversational context and focus on overt toxicity. The hard cases (misread jokes, irony, insensitive-but-polite phrasing) need targeted data. Paying once for labeled data makes every later check free.

### D7 — UX: Warn, never block

- **Chosen:** A small, non-focusable pill near the input field; tap for details.
- **Why:** Hard blocking is annoying with any false positive and erodes trust quickly. The user keeps full control.

### D9 — Phase 2 base model: 33M meaning-trained BERT models (decided 2026-10-06)

- **Chosen:** shortlist **`sentence-transformers/all-MiniLM-L12-v2`** and **`BAAI/bge-small-en-v1.5`** (both ~33M params, ~34 MB int8). Train both on the Phase 2 data; ship whichever scores better on the golden set.
- **Alternatives:** keep raw `nreimers/MiniLM-L6-H384-uncased` (current stand-in); DeBERTa-v3-xsmall/small/base.
- **Why:**
  - Both are pre-trained to understand **sentence meaning**. In a probe test, sentence-meaning similarity with a same-size MiniLM caught the "endorsing something horrific, no bad words" cases that the raw MiniLM fine-tuned on civil_comments missed.
  - On the S25 they take ~6 ms at 64 tokens (≈2× MiniLM-L6) and are estimated at ~30–50 ms on budget phones (§6.1 benchmark).
  - Their `vocab.txt` is **byte-identical** to the current one (BERT uncased, 30,522 tokens, `do_lower_case=true`), so the Kotlin tokenizer, the encoder and the parity tests are unchanged. Only the model bundle is swapped.
  - DeBERTa needs a SentencePiece tokenizer (Kotlin rewrite), is 87–244 MB, and the 184M base model OOMed on load. Reconsider DeBERTa-v3-xsmall only if both 33M models fail the golden set.

### D10 — Training strategy: distillation (decided 2026-10-06)

- **Step 1 (data distillation):** Claude generates and labels 5–10k chat snippets with context, including contrastive pairs (near-identical messages with opposite labels, e.g. a riff on "slave away" that endorses slavery vs. ordinary use of the phrase). The 33M student trains directly on this plus cleaned public data (§8).
- **Step 2 (only if the student plateaus): teacher → student.** Fine-tune DeBERTa-v3-base as a PC-only teacher (it scored 14/15 zero-shot on the probe set), have it soft-label 200k–1M unlabeled real chat messages, and train the student on Claude labels + teacher soft labels.
- **Guards:** a human-written golden set the teacher never grades; ≥5% manual spot-checks; messy-style generation and augmentation (typos, slang, emoji); harmless identity mentions to avoid copying over-flagging bias; never train on real private chats without consent; check the API terms on training with model outputs before public release.

### D8 — Check timing: Continuous inference, debounced display

- **Chosen:** Inference may run on every (throttled) text change because it's cheap; the pill is **displayed** only after the result is stable for a short pause (default 300 ms).
- **Why:** Fast typists hit send almost immediately. Computing continuously means a result is usually ready the moment they pause. Debouncing the display prevents flicker while typing.

---

## 4. System overview

```
 ┌──────────────── OFFLINE, ON DEVELOPER PC (run once per model version) ───────────────┐
 │                                                                                     │
 │  Public datasets ───────┐                                                           │
 │  Synthetic, LLM-labeled ├──► Build dataset ──► Fine-tune ──► Evaluate ──► Export    │
 │  context examples ──────┤    (JSONL)           MiniLM        (gates)      ONNX+int8 │
 │  Hand-written hard cases┘                                                    │      │
 │                                                                              ▼      │
 │                                         model.onnx + vocab.txt + model_config.json │
 └──────────────────────────────────────────────────────────────────────┬──────────────┘
                                                                        │ bundled in APK assets
 ┌──────────────────────────── ON PHONE (every keystroke) ──────────────▼──────────────┐
 │                                                                                     │
 │  Messaging app (e.g. Discord)                                                       │
 │        │ accessibility events                                                       │
 │        ▼                                                                            │
 │  ① WakeItUpAccessibilityService ──► ② ContextCache ◄── AppAdapter (per-app parsing) │
 │        │ draft text                       │ last N messages                         │
 │        ▼                                  │                                         │
 │  ③ QuickFilter (empty/unchanged/wordlist) │                                         │
 │        │                                  │                                         │
 │        ▼                                  ▼                                         │
 │  ④ WordPieceTokenizer (context, draft) ──► ⑤ OnnxClassifier ──► per-label probs     │
 │                                                                     │               │
 │                                                                     ▼               │
 │                                    ⑥ DecisionEngine (thresholds, hysteresis, pause) │
 │                                                                     │               │
 │                                                                     ▼               │
 │                                    ⑦ OverlayController ──► pill ⚠️ ──tap──► card    │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

**Two independent halves joined by one contract** (§7): the training pipeline produces three files, and the app consumes them. Either half can be rebuilt without touching the other as long as the contract holds.

---

## 5. Android app — component design

### 5.1 Tech stack

| Item | Choice | Notes |
|---|---|---|
| Language | Kotlin | |
| Build | Gradle (Kotlin DSL), Android Gradle Plugin 8.12.x, Gradle 8.14 | Versions already cached on the dev machine; bump deliberately. |
| JDK | 17 | |
| minSdk / targetSdk | 26 / 36 | Test device: Samsung Galaxy S25 (One UI, Android 15+). |
| Kotlin | 2.2.20 | |
| Inference | ONNX Runtime 1.22.0: `onnxruntime-android` in `:app`, desktop `onnxruntime` for `:core` compile + tests | Both pinned via `wakeitup.onnxruntimeVersion` in `android/gradle.properties`; keep them equal. |
| Async | Kotlin coroutines 1.10.2 | |
| Settings storage | Private `SharedPreferences` | Chosen over DataStore to keep dependencies minimal; stores no message text. |
| UI | Android Views, built in code (no XML layouts, no AndroidX) | Overlay must be plain Views anyway. |
| ABIs | `arm64-v8a` (phones) + `x86_64` (PC emulator) | Halves ONNX Runtime's native size. |

**Module split (as built):**
- `:core`: pure Kotlin/JVM, no Android types. Tokenizer, encoder, model config, ONNX classifier wrapper, decision engine, quick filter, wordlist, chat-layout parser, explanation templates. **All unit-tested on the PC**, including model parity through the desktop ONNX Runtime.
- `:app`: Android-only glue. Accessibility service, node-tree walking, overlay, settings, test screen.

### 5.2 Package structure (as built)

```
:core  — dev.wakeitup.core   (pure Kotlin, JVM-tested)
├── Chat.kt                     ChatMessage, ConversationSnapshot
├── WordPieceTokenizer.kt       BERT uncased WordPiece (HF slow-tokenizer parity)
├── InputEncoder.kt             §6.2.1 pair encoding + truncation
├── ModelConfig.kt              Parses/validates model_config.json (contract check)
├── OnnxClassifier.kt           ONNX Runtime session, sigmoid outputs
├── ToneEngine.kt               Encoder + classifier bundle; check(), warmUp(), load()
├── DecisionEngine.kt           Thresholds, sensitivity presets, hysteresis, levels
├── QuickFilter.kt              Skip short/duplicate drafts
├── Wordlist.kt                 Instant hard-flag terms (whole-word, normalized)
├── ChatLayoutParser.kt         Screen text → anonymized snapshot (authors, timestamps, reply)
└── Explanations.kt             Category templates for the detail card

:app   — dev.wakeitup
├── WakeItUpApp.kt             Application + EngineHolder (loads model bundle from assets once)
├── service/WakeItUpAccessibilityService.kt   Events, debounce, threading, watchdog, package filter
├── context/
│   ├── ContextCache.kt         Latest snapshot + version counter
│   └── adapters/AppAdapter.kt  AppAdapter, Adapters registry, DiscordAdapter, GenericAdapter, NodeWalker
├── overlay/OverlayController.kt  Pill + detail card (TYPE_ACCESSIBILITY_OVERLAY)
├── settings/Settings.kt        SharedPreferences: apps, sensitivity, display name, delay, feedback counters
├── ui/SettingsActivity.kt      Setup flow, status, toggles, debug tools
├── ui/TestActivity.kt          Playground + 200-run benchmark
└── util/SafeLog.kt             Content-free logging in release
```

Additions beyond the original design, made during implementation:
- **Display name setting** (optional): lets the parser label the user's own messages `ME`.
- **Visibility watchdog**: events from non-enabled apps are filtered out, so the service can't see the user switching away. While a warning is showing, it polls the active window every 750 ms and hides the warning when the app changes.
- **Debug tree dump** (debug builds only): logs the next screen's node structure to logcat (tag `WakeItUp`) for tuning adapters on a real device.

### 5.3 ① Accessibility service

**Config** (`res/xml/accessibility_service_config.xml`):

```xml
<accessibility-service xmlns:android="http://schemas.android.com/apk/res/android"
    android:accessibilityEventTypes="typeViewTextChanged|typeViewFocused|typeWindowContentChanged|typeWindowStateChanged"
    android:accessibilityFeedbackType="feedbackGeneric"
    android:accessibilityFlags="flagReportViewIds|flagRetrieveInteractiveWindows"
    android:canRetrieveWindowContent="true"
    android:notificationTimeout="50"
    android:packageNames="com.discord"
    android:description="@string/a11y_service_description"
    android:settingsActivity="dev.wakeitup.ui.SettingsActivity" />
```

- `packageNames` in XML is the default; at runtime the service calls `setServiceInfo()` with the user's enabled-app list so it **never receives events from other apps**.
- Manifest entry requires `android:permission="android.permission.BIND_ACCESSIBILITY_SERVICE"` and the `android.accessibilityservice.AccessibilityService` intent filter.

**Event handling:**

| Event | Action |
|---|---|
| `TYPE_VIEW_TEXT_CHANGED` (editable, not password) | New draft → QuickFilter → (maybe) inference. Remember the source node's screen bounds for overlay placement. |
| `TYPE_VIEW_FOCUSED` on an editable field | Mark this as the active input; trigger a context refresh. |
| `TYPE_WINDOW_CONTENT_CHANGED` / `TYPE_WINDOW_STATE_CHANGED` | Invalidate ContextCache (throttled refresh, see §5.4). Hide overlay if the input field is gone. |
| Draft becomes empty | Message was sent or cleared → hide overlay, reset decision state. |

**Rules:**
- Skip any node where `isPassword == true`.
- Skip if the package isn't in the enabled list (defense in depth, even though `setServiceInfo` filters).
- All heavy work happens off the main thread. The service callback must return quickly.
- Every handler is wrapped in try/catch. On error, log a non-content error code and do nothing (G6).

### 5.4 ② Context cache

Reading the full node tree costs roughly 20–100 ms, too slow for every keystroke. So:

- Context is **refreshed on content changes**, throttled to at most once per 250 ms, on a background dispatcher.
- Each refresh walks `rootInActiveWindow` via the app's `AppAdapter` and stores a `ConversationSnapshot`:
  ```kotlin
  data class ConversationSnapshot(
      val messages: List<ChatMessage>,   // oldest → newest, max N (default 5)
      val replyTarget: ChatMessage?,     // message explicitly being replied to, if detectable
      val capturedAtMs: Long,
  )
  data class ChatMessage(val speaker: String, val text: String, val isSelf: Boolean)
  ```
- When a draft changes, the latest snapshot is used **as-is** (≈1 ms).
- Node objects are recycled/released promptly. Never hold `AccessibilityNodeInfo` references across refreshes.

**Speaker anonymization** happens here, before anything reaches the model: speakers are mapped to `ME`, `A`, `B`, `C`… per snapshot. Real usernames are never fed to the model. This helps privacy and also generalization, since training data uses the same scheme.

**AppAdapter interface** (isolates per-app UI quirks so a UI change in one app only breaks one file):

```kotlin
interface AppAdapter {
    val packageName: String
    fun extract(root: AccessibilityNodeInfo, input: AccessibilityNodeInfo?): ConversationSnapshot?
}
```

- `DiscordAdapter`: collect text nodes inside the message list, order by on-screen top coordinate, group consecutive lines under the nearest preceding author label, drop timestamps (regex) and UI chrome (reaction counts, "edited", etc.), detect the "Replying to …" banner above the input as `replyTarget`.
- `GenericAdapter`: visible non-editable text nodes above the input field, ordered by position, last N kept. Used for any app without a dedicated adapter.
- Adapters must return `null` rather than throw when the tree is unrecognizable; the pipeline then runs **draft-only**.

### 5.5 ③ Quick filter

Runs before tokenization, costs microseconds:

- Skip if the trimmed draft is shorter than 2 characters.
- Skip if the draft is identical to the last evaluated draft (with the same context version).
- **Wordlist:** a small list of unambiguous slurs (asset file, normalized lowercase, with common obfuscation handling such as repeated letters and leetspeak). A hit produces an immediate high-risk result with category `hate_slur`, with no model call needed.

### 5.6 ④ Tokenizer

- BERT **uncased WordPiece**, implemented in pure Kotlin (no native dependency). Vocab from `vocab.txt` (§7).
- Basic tokenization: lowercase (if `do_lower_case`), strip accents, split on whitespace and punctuation; then greedy longest-match WordPiece with `##` continuation, `[UNK]` fallback.
- **Pair encoding:** `[CLS] context [SEP] draft [SEP]`, `token_type_ids` 0 for the context segment, 1 for the draft segment.
- **Context string format** (must match training exactly, see §6.2):
  ```
  A: first message | B: second message | reply_to A: message being replied to
  ```
  Messages are joined with ` | `. The reply target, if any, is placed **last** and prefixed with `reply_to `.
- **Truncation:** total length `max_len` (default 128). The draft is protected: truncate the context **from the oldest message first**. If the draft alone exceeds the budget, keep its **last** tokens. The exact algorithm is in §6.2.1.
- Emojis and unknown symbols map to `[UNK]` unless present in vocab; acceptable for v1.

### 5.7 ⑤ ONNX classifier

- One `OrtSession` per process, created lazily at service start from `assets/model/model.onnx`.
- **Warm-up:** run one dummy inference right after load so the first real check isn't slow.
- Session options: start with CPU execution provider and `intraOpNumThreads = 2`. NNAPI/XNNPACK are optional experiments; keep whichever is fastest **measured on device** and record it here.
- Inputs: `input_ids`, `attention_mask`, `token_type_ids` (int64, shape `[1, L]`). Output: `logits` (float32, shape `[1, num_labels]`). Apply **sigmoid** per label (multi-label).
- Runs on a dedicated single-thread dispatcher. A newer draft **cancels** a pending/older job (latest-wins).

### 5.8 ⑥ Decision engine

- Input: per-label probabilities + overall `risk` probability.
- **Thresholds** come from `model_config.json`, shifted by the user's sensitivity preset:

  | Preset | Effect |
  |---|---|
  | Relaxed | thresholds + 0.10 (fewer warnings) |
  | Normal (default) | thresholds as shipped |
  | Strict | thresholds − 0.10 (more warnings) |

  Shifted thresholds are clamped to **[0.05, 0.99]**, so a high tuned threshold (e.g. 0.91) under Relaxed becomes 0.99 rather than unreachable.

- **Hysteresis** (anti-flicker): show when `risk ≥ t_on`; hide only when `risk < t_off` (`t_off = t_on − 0.10`).
- **Display debounce:** a "show" decision must stay stable for `display_delay_ms` (default 300) before the pill appears. "Hide" is immediate when the draft is emptied.
- Output: `None`, or `Warning(level, topCategory)` where level is `medium`/`high`.

**Explanation templates** (shown in the detail card, one per category):

| Category | Template |
|---|---|
| `hate_slur` | "Contains a slur or language that targets a group. This is likely to hurt people even as a joke." |
| `insensitive` | "Could read as making light of a serious or painful topic, especially out of context." |
| `harassment` | "May come across as an attack on someone in the chat rather than banter." |
| `threat` | "Could be read as a threat, even if you're joking." |
| `sexual` | "Sexual content that may not fit this conversation." |
| `misread_risk` | "The joke depends on context. Someone skimming may take it literally." |

### 5.9 ⑦ Overlay

- Window type `TYPE_ACCESSIBILITY_OVERLAY`. This is available to accessibility services with **no extra permission**.
- Flags: `FLAG_NOT_FOCUSABLE | FLAG_NOT_TOUCH_MODAL | FLAG_LAYOUT_NO_LIMITS`, so it never steals focus or input from the host app.
- **Pill:** small rounded chip "⚠️ Might land badly", positioned just above the input field's top edge using the bounds captured in §5.3. Falls back to the top-center of the screen if bounds are unknown.
- **Tap:** expand to the detail card (category template + "Got it" / "Not helpful" buttons).
- "Not helpful" records a **local, content-free** counter per category (for tuning thresholds later). It never stores message text.
- Auto-hide when: draft is emptied, input loses focus, app leaves the foreground, or the risk drops below `t_off`.

### 5.10 Settings & setup flow

The setup flow in `SettingsActivity`:
1. Explain what the app does and what it reads (prominent disclosure, required for store review).
2. Choose apps to watch. Phase 1 ships with Discord only; candidates for later: `com.whatsapp`, `com.samsung.android.messaging`, `com.google.android.apps.messaging`, `com.instagram.android`, `org.telegram.messenger`, `com.snapchat.android`.
3. Sensitivity preset.
4. Guide the user through enabling the service. **There is no runtime permission popup for accessibility services. Android forbids it by design, so the user must flip the switch in Settings.** The app shows a "Turn on Wake It Up" dialog when the service is off (once per screen visit) and deep-links straight to Wake It Up's own switch via `android.settings.ACCESSIBILITY_DETAILS_SETTINGS` + `EXTRA_COMPONENT_NAME`, falling back to the accessibility list. Steps:
   - **Sideloaded APKs on Android 13+:** App info → ⋮ → *Allow restricted settings* first.
   - Settings → Accessibility → Installed apps → Wake It Up → On.
5. **Samsung battery:** guide the user to add Wake It Up to *Never sleeping apps* (Settings → Battery → Background usage limits). Otherwise One UI may kill the service.
6. Live status indicator: service on/off, model loaded, last inference time (ms).

### 5.11 Threading model

| Thread | Work |
|---|---|
| Main | Receive accessibility events (return fast), all overlay View operations |
| `contextDispatcher` (single thread) | Node tree walks, ContextCache refresh |
| `inferenceDispatcher` (single thread) | Tokenize + ONNX run; latest-wins cancellation |

---

## 6. Model specification

### 6.1 Base model candidates

| Candidate | Params | Expected on-device latency | Notes |
|---|---|---|---|
| **MiniLM-L6-H384 (uncased)** | ~22M | lowest | **Default.** Start here. |
| MiniLM-L12-H384 (uncased) | ~33M | ~2× L6 | Use if L6 misses accuracy gates. |
| DeBERTa-v3-xsmall | ~70M (incl. embeddings) | highest | Best accuracy of the three; only if latency headroom allows. Uses a SentencePiece tokenizer, which would require a different tokenizer in the app. |

Choose by measured results on the golden set **and** on-device latency (§9). Record the chosen model and the numbers in this section.

#### Candidate speed benchmark (2026-10-06)

int8 (QUInt8) ONNX, untrained heads, synthetic inputs. Scripts: `training/scripts/bench_export_candidates.py` (PC) and the debug-only "Benchmark candidate models" button in `TestActivity` (phone; models pushed to `/sdcard/Android/data/dev.wakeitup/files/bench/`). The S25 numbers are the better of two runs: CPU frequency scaling made the first run up to ~2× slower.

| Model | Params | int8 size | Tokenizer | PC 2 thr @128 | **S25 2 thr @64 / @128** | S25 1 thr @128 |
|---|---|---|---|---|---|---|
| MiniLM-L6 (current) | 23M | 23 MB | BERT WordPiece ✅ | 18 ms | **2.9 / 6.1 ms** | 11.3 ms |
| all-MiniLM-L12-v2 | 33M | 34 MB | BERT WordPiece ✅ | 35 ms | 6.6 / 13.8 ms | 25.5 ms |
| bge-small-en-v1.5 | 33M | 34 MB | BERT WordPiece ✅ | 38 ms | 5.8 / 12.1 ms | 25.5 ms |
| DeBERTa-v3-xsmall | 71M | 87 MB | SentencePiece ❌ | 56 ms | 11.6 / 23.7 ms | 42.6 ms |
| DeBERTa-v3-small | 142M | 172 MB | SentencePiece ❌ | 83 ms | ~22 / ~27–67 ms (noisy) | 45 ms |
| DeBERTa-v3-base | 184M | 244 MB | SentencePiece ❌ | 156 ms | **failed to load: OutOfMemoryError** | — |

Findings:
- Every model up to DeBERTa-v3-small fits the S25 latency budget. **Size and memory are the real limits, not speed.** The app heap limit was 256 MB, so loading a 244 MB model as a byte array OOMs. Models above ~100 MB would need file-path/mmap loading and would bloat the APK.
- Weaker phones are **estimated** (not measured) at ~2–3× slower than the S25 for mid-range and ~5–8× for budget phones. At a typical 64 tokens: MiniLM-L6 ≈ 15–25 ms on budget phones, MiniLM-L12/bge ≈ 30–50 ms, DeBERTa-xsmall ≈ 60–100 ms, DeBERTa-small ≈ 110–180 ms. Budget phones also have less RAM, which favors the ≤ 35 MB models.
- Recommendation: the 33M BERT-tokenizer models (all-MiniLM-L12-v2, bge-small) are the sweet spot. Same tokenizer and contract, about 2× the L6 cost, fine on budget phones. Consider DeBERTa only if they fail the golden set.

#### Recorded results

**`standin-2026.10.0`** (Phase 1 stand-in, 2026-10-04): `nreimers/MiniLM-L6-H384-uncased`, draft-only (context always `none`), trained on 180k civil_comments rows (≤300 chars; all 50k toxic + 150k non-toxic), soft-label BCE, 3 epochs, ~15 min on an RTX 2060.

| Label | Threshold | Test F1 (fp32 / int8) | AUC |
|---|---|---|---|
| risk | 0.719 | 0.833 / 0.828 (P 0.845, R 0.821, FPR 0.051) | 0.969 |
| hate_slur | 0.909 | 0.605 / 0.598 | 0.984 |
| insensitive | 0.900 | 0.615 / 0.613 | 0.984 |
| harassment | 0.796 | 0.806 / 0.803 | 0.970 |
| threat | 0.880 | 0.603 / 0.573 | 0.987 |
| sexual | 0.856 | 0.596 / 0.598 | 0.987 |
| misread_risk | 0.99 (untrained) | — | — |

- int8 model: 23.0 MB. Kotlin↔Python parity on the PC: worst abs diff 5e-7 over 36 cases.
- PC latency (Ryzen 7 4800H, JVM, 53 tokens): p50 7.6 ms. Python ORT at 128 tokens: p50 16.5 ms (2 threads).
- **Known behavior: not shippable as-is.** It flags mild swearing and excited banter (e.g. "damn that's crazy" 0.90), because civil_comments labels these toxic, which violates G4. It also misses implicit/polite offense (e.g. a "go back where you came from" style remark 0.06). Both are expected for a draft-only stand-in on this dataset. Phase 2 data (benign banter, implicit cases, context) is the fix.

### 6.2 Input format (must be identical in training and app)

- Segment A (context): up to 5 prior messages, oldest → newest, each `SPEAKER: text`, joined by ` | `. If a reply target exists, it is moved to the end and prefixed `reply_to `.
- Speakers: `ME` for the user, `A`, `B`, `C`… for others, assigned in order of first appearance.
- Segment B (draft): the raw draft text (no speaker prefix).
- `max_len = 128` tokens; truncation per the algorithm below.
- Empty context is allowed (draft-only mode): segment A is the literal string `none`.

Example (illustrative, neutral):

```
Segment A: A: anyone up for food later | B: cant, still working | reply_to A: anyone up for food later
Segment B: only if someone else is paying lol
```

#### 6.2.1 Exact encoding algorithm (normative)

Python (`training/tg_format.py`) and Kotlin (`ml/InputEncoder.kt`) **must** implement exactly this. Parity is verified by fixtures (§12.1).

```
Inputs: messages: list of (speaker, text), oldest → newest (already anonymized)
        reply_index: index into messages or null
        draft: string
        max_len = 128

1. If reply_index is not null: move messages[reply_index] to the end of the list
   and mark it as the reply.
2. Keep only the last 5 messages (context_max_messages).
3. Format each message as  "{speaker}: {text}", or "reply_to {speaker}: {text}" for the reply.
4. d = wordpiece(draft)                       # BERT uncased basic tokenizer + WordPiece
   budget = max_len - 3                       # [CLS], [SEP], [SEP]
   if len(d) > budget - 1: d = last (budget - 1) tokens of d
   c_budget = budget - len(d)                 # always >= 1
5. msgs = [wordpiece(m) for m in formatted messages]
   Drop messages from the FRONT (oldest) while
       len(msgs) > 1  and  sum(len(m) for m in msgs) + (len(msgs) - 1) > c_budget
       # (len-1) = "|" separators. Never drops the last remaining message.
   If exactly one message remains and it is still longer than c_budget,
       keep only its LAST c_budget tokens.
   (A context consisting of a single long message is therefore never replaced by "none".)
6. c = msgs joined with the single token "|" between them.
   If c is empty (no messages, or all dropped): c = wordpiece("none")
7. input_ids      = [CLS] + c + [SEP] + d + [SEP]
   token_type_ids = 0 for [CLS] + c + [SEP]; 1 for d + [SEP]
   attention_mask = all 1s. No padding (dynamic length, batch size 1).
```

`wordpiece()` must reproduce the Hugging Face **slow/legacy** BERT tokenizer with `do_lower_case=True`. In transformers ≥ 5, `transformers.BertTokenizer` is the Rust-backed fast tokenizer; the reference is `transformers.models.bert.tokenization_bert_legacy.BertTokenizerLegacy` (used by `training/tg_format.py`). Steps, in order:

1. Lowercase.
2. Clean: drop U+0000, U+FFFD and control characters (`C*` categories except `\t \n \r`, including Cf zero-width chars, soft hyphen, BOM, Co private use); map whitespace (`\t \n \r`, `Zs`) to a space.
3. Pad CJK ideographs with spaces.
4. NFC-normalize.
5. Split like Python `str.split()` (this also splits on U+2028/U+2029 and other `isspace()` chars that aren't `Zs`).
6. Per word: lowercase, strip accents (NFD, drop `Mn`), split on punctuation (Unicode `P*` plus ASCII 33–47, 58–64, 91–96, 123–126).
7. Greedy longest-match WordPiece with `##` prefixes. Lengths are counted in **code points** (not UTF-16 chars). Words over 100 code points, or with no match, become `[UNK]`.

Special-token strings in user text (`[CLS]`, `[SEP]`, …) are **plain text** (e.g. `[ cl ##s ]`), never control tokens.

Known, accepted deviation: Kotlin does **not** treat unassigned code points (`Cn`) as control, because the JVM's Unicode tables can be older than Python's and newer emoji would otherwise be dropped instead of becoming `[UNK]`. On Android (current ICU) this matches the reference in practice.

Gotcha: in transformers 5, `BertTokenizerFast(vocab_file=...)` silently ignores the file. Pass `vocab=dict` if the fast tokenizer is ever needed.

### 6.3 Output labels (multi-label, sigmoid)

| Index | Label | Meaning |
|---|---|---|
| 0 | `risk` | Overall: would a reasonable group member likely be offended or misread this in context? |
| 1 | `hate_slur` | Slurs, dehumanizing language about protected groups |
| 2 | `insensitive` | Trivializing atrocities, tragedies, or identity; stereotypes, including "polite" ones |
| 3 | `harassment` | Insults/attacks aimed at a person in the chat, beyond clear mutual banter |
| 4 | `threat` | Threats of violence or harm, including joking ones that read as real |
| 5 | `sexual` | Sexual content likely inappropriate for the context |
| 6 | `misread_risk` | Irony or a context-dependent joke that reads badly when taken literally |

`risk` is trained directly (not computed as a max), so the model can learn that e.g. mild swearing among friends is *not* risky.

### 6.4 Optimization

- Export to ONNX (opset ≥ 14) with dynamic sequence axis.
- **Dynamic int8 quantization** (ONNX Runtime quantization tools). Expect a ~4× size reduction to ~25 MB for L6.
- Accuracy after quantization must stay within 1 point of the fp32 model on the golden set; otherwise ship fp16/fp32.
- Weights are quantized as **QUInt8 (U8U8)**, not the default QInt8 (U8S8): on x86 CPUs without VNNI, ORT's U8S8 kernels can saturate, which would make PC-generated fixtures disagree with ARM phones.
- **The ONNX Runtime version must be identical** in `training/requirements.txt` and `android/gradle.properties` (currently 1.30.0). With 1.22 on the app side, int8 outputs drifted by up to 0.04 from the Python reference, while the encoder matched exactly.
- int8 is numerically sensitive (max per-example int8 vs fp32 prob diff on test: 0.18). If on-device results ever differ from PC fixtures at the margins, check kernel/version differences first.

---

## 7. The model contract (app ↔ training)

**This is the most important interface in the project.** The app must work with any model bundle that satisfies it.

Location in the app: `app/src/main/assets/model/`

| File | Content |
|---|---|
| `model.onnx` | Quantized classifier. Inputs `input_ids`, `attention_mask`, `token_type_ids` (int64 `[1, L]`); output `logits` (float32 `[1, num_labels]`). |
| `vocab.txt` | WordPiece vocabulary used in training. |
| `model_config.json` | Metadata, see schema below. |

`model_config.json` schema:

```json
{
  "contract_version": 1,
  "model_version": "2026.10.0",
  "base_model": "MiniLM-L6-H384-uncased",
  "do_lower_case": true,
  "max_len": 128,
  "context_max_messages": 5,
  "context_separator": " | ",
  "reply_prefix": "reply_to ",
  "empty_context_token": "none",
  "labels": ["risk", "hate_slur", "insensitive", "harassment", "threat", "sexual", "misread_risk"],
  "thresholds": { "risk": 0.60, "hate_slur": 0.50, "insensitive": 0.55, "harassment": 0.60, "threat": 0.55, "sexual": 0.65, "misread_risk": 0.60 },
  "eval": { "golden_recall_risky": 0.0, "golden_fpr_benign": 0.0, "notes": "" }
}
```

Rules:
- The app **refuses to load** a bundle whose `contract_version` it doesn't support. It then shows "model error" in settings and stays silent (G6).
- Any change to input formatting, labels, or tokenizer **bumps `contract_version`** and requires matching app changes.
- Thresholds are tuned in training (on the validation set) and shipped with the model, so recalibration doesn't need app code changes.

---

## 8. Training pipeline

Lives in `training/` (Python). Runs on the developer PC (NVIDIA GPU).

### 8.1 Stages

```
01_download_public.py   → data/raw/public/
02_generate_synthetic.py → data/raw/synthetic/   (LLM generation + labeling, run once)
03_build_dataset.py     → data/processed/{train,val,test}.jsonl  (normalize, map labels, dedupe, split)
04_train.py             → runs/<id>/checkpoint
05_eval.py              → runs/<id>/report.md   (incl. golden set, threshold tuning)
06_export_onnx.py       → runs/<id>/export/model.onnx (+ vocab.txt, model_config.json)
07_quantize.py          → runs/<id>/export/model.int8.onnx
```

Each script is idempotent and reads/writes only its own folders, so any stage can be re-run alone.

### 8.2 Unified record schema (JSONL)

```json
{
  "id": "syn-000123",
  "source": "synthetic|jigsaw|civil_comments|toxigen|sbf|golden",
  "context": [ {"speaker": "A", "text": "..."}, {"speaker": "B", "text": "..."} ],
  "reply_to": 1,
  "draft": "...",
  "labels": { "risk": 1, "hate_slur": 0, "insensitive": 1, "harassment": 0, "threat": 0, "sexual": 0, "misread_risk": 1 },
  "rationale": "short reason (synthetic/golden only, not used for training)",
  "scenario": "riff_on_previous_word"
}
```

Public single-comment datasets have empty `context` and are mapped into the label set; unmappable labels are dropped.

### 8.3 Data sources

| Source | Provides | Notes |
|---|---|---|
| Jigsaw Toxic Comment Classification | Overt toxicity, insults, threats | Single comments. Verify license/terms before use. |
| Civil Comments | Toxicity + identity mentions | Helps reduce bias against identity terms used neutrally. Verify license. |
| ToxiGen | Implicit / subtle hate, including benign statements about groups | Key for "polite but offensive" and for *not* flagging neutral identity mentions. Verify license. |
| Social Bias Frames | Implied offensiveness and stereotypes | Verify license. |
| **Synthetic (LLM-labeled)** | **Conversational context cases**, the core of the product | See §8.4. |
| **Golden set (hand-written)** | Hard, realistic cases for evaluation only | Never used for training. See §12.2. |

> Before any commercial release, re-check each dataset's license and terms of use and record the result here.

### 8.4 Synthetic data generation (distillation)

- A large model (Claude, via the Anthropic API) generates short group-chat snippets (2–5 context messages + a draft) across defined **scenarios**, then labels each one with the §6.3 labels and a one-line rationale.
- This runs **once, offline, on the developer machine**. The API key lives only in the developer's environment, never in the app or repo.
- **Scenario list** (balanced positives and negatives):
  - Riff on a word from the previous message that changes its meaning (risky, `misread_risk`/`insensitive`)
  - Same riff structure but harmless (not risky; teaches the model not to over-flag the pattern)
  - Ironic doubling-down after a bad reaction (risky)
  - Sincere clarification or apology (not risky)
  - Friendly swearing / roasting among friends (not risky)
  - Targeted insult at one person (risky, `harassment`)
  - Neutral mention of identity, religion, or politics (not risky)
  - Stereotype phrased politely (risky, `insensitive`)
  - Dark joke about a tragedy (risky, `insensitive`)
  - Joking threat (risky, `threat`)
- Target size v1: ~5,000–10,000 synthetic records. Spot-check ≥5% by hand; fix or drop bad labels.
- **Split by scenario template**, not just randomly, to avoid near-duplicates leaking between train and test.
- The generated data necessarily contains offensive text. Keep `data/` out of version control (gitignored), and don't publish it.

### 8.5 Training configuration (starting point)

| Setting | Value |
|---|---|
| Objective | Multi-label BCE over 7 labels |
| Class weighting | `pos_weight` per label from train-set frequencies |
| Max length | 128 |
| Batch size | 32 |
| Learning rate | 3e-5, linear decay, 6% warm-up |
| Epochs | 3–5, early stopping on validation `risk` F1 |
| Mixing | Upsample synthetic + context records so they're ≥ 40% of each epoch |

Threshold tuning: on the validation set, pick per-label thresholds that hit the recall target with the lowest false-positive rate; write them into `model_config.json`.

### 8.6 Release gates (must pass before a model ships)

| Metric | Initial target |
|---|---|
| Golden set: recall on risky cases | ≥ 0.85 |
| Golden set: false-positive rate on benign banter | ≤ 0.05 |
| Test set: `risk` macro-F1 | ≥ 0.80 |
| int8 vs fp32 `risk` F1 drop | ≤ 1 point |
| On-device p95 inference latency (S25, 128 tokens) | ≤ 30 ms |

Targets are initial guesses. Revise them after the first real evaluation and record the change.

---

## 9. Performance targets and latency budget

Measured from **the moment the user stops typing** to the pill being visible:

| Step | Budget |
|---|---|
| Display debounce (deliberate) | 300 ms (configurable) |
| Context lookup (cache hit) | ≤ 2 ms |
| Quick filter | < 1 ms |
| Tokenize | ≤ 2 ms |
| Inference (MiniLM-L6 int8, S25) | ≤ 20 ms typical, ≤ 30 ms p95 (**measured: 3.5 ms p50 / 3.6 ms p95 at 65 tokens**) |
| Overlay draw | ~16 ms (one frame) |
| **Total** | **≈ 350 ms**, dominated by the deliberate debounce |

Other targets:
- APK size increase from the model: ≤ 30 MB (stand-in int8: 23 MB ✅).
- **ONNX Runtime native size is the bigger cost:** with 1.30.0, `libonnxruntime.so` is ~33 MB (arm64) and ~39 MB (x86_64); the debug APK is ~104 MB. Phones only install arm64 from an App Bundle (~57 MB total). Before public release, consider an ONNX Runtime **custom minimal build** limited to this model's operators (typically a few MB), and drop `x86_64` from release builds.
- Battery: no measurable drain while idle. Work happens only on events from enabled apps.
- Memory: model session ≤ 60 MB resident.

---

## 10. Privacy and security

Non-negotiable rules:

1. **No network permission** in the app manifest for v1. This makes "no data leaves the phone" verifiable.
2. Only events from **user-enabled packages** are processed.
3. **Password fields are never read.**
4. Message text is kept **only in memory** and only for the current snapshot/draft. It is never written to disk, logs, crash reports, or analytics.
5. `SafeLog` strips content in release builds. Only timings, error codes, and counts may be logged.
6. Usernames are anonymized before reaching the model (§5.4).
7. "Not helpful" feedback stores only counters per category, never text.
8. No API keys or secrets anywhere in the app or the repo. Synthetic-data generation reads the key from the developer's environment.

---

## 11. Distribution and platform policy

- **Phase 1–3:** sideload a debug/release APK onto the developer's S25 via USB (`adb install`).
- **Google Play (later):** apps using `AccessibilityService` receive extra review. Requirements to prepare:
  - Prominent in-app disclosure plus explicit consent before enabling (the §5.10 setup flow).
  - The Play Console accessibility declaration explaining the use: an assistive tool that helps users with social-communication difficulties anticipate how their messages will be read.
  - A privacy policy reflecting §10 (on-device only, nothing stored or transmitted).
- Re-check current Play policy wording at submission time; it changes.

---

### 11.1 Production readiness (deferred)

Because inference is fully on-device there is **no server to scale**: per-check cost is zero and does not grow with users. Most production concerns can wait until a public release.

| Concern | Status | When |
|---|---|---|
| Server scaling | Not applicable (no server) | Never |
| Forced update | Not needed: old versions keep working since there is no server API to break. Normal Play auto-updates suffice. If ever needed, use the Play In-App Updates API ("immediate" mode). | Later, only if needed |
| Shipping new models | v1 bundles the model in the APK, so new model = app update. Later option: deliver model files separately through Google Play's asset/model delivery. | Later |
| Bad model release | Use Play **staged rollouts** (e.g. 5% → 100%). | First public release |
| Host app UI changes break context parsing | **Ongoing maintenance.** Isolated to one `AppAdapter` per app; pipeline falls back to draft-only (§5.4). | Ongoing |
| Crash reporting / analytics | Conflicts with the no-network rule (§10). Opt-in only, if ever. | Later |
| Low-end devices | Expect 3–5× slower than the S25; still well under 100 ms for MiniLM-L6. Verify. | Before public release |
| Languages | English only in v1. | Later |
| Signing key | Losing it makes updates impossible. Enable **Play App Signing**; back up the upload key. | First Play upload |

Things that **must** be right from day one (already in the design): the versioned model contract (§7) and per-app adapter isolation (§5.4).

---

## 12. Testing strategy

### 12.1 Unit tests (JVM)

- `WordPieceTokenizer`: output matches the Python tokenizer's token IDs on a fixed fixture file (exported by the training pipeline). **This is the most likely source of silent accuracy loss.**
- Context formatting: matches §6.2 byte-for-byte against fixtures.
- `DecisionEngine`: thresholds, presets, hysteresis, debounce timing.
- `QuickFilter` / `Wordlist`: normalization and obfuscation handling.

### 12.2 Golden set

- ~150–300 hand-written, realistic cases in `training/golden/golden_set.jsonl`, same schema as §8.2.
- Must include: context-dependent misread jokes, the same jokes with harmless context, ironic doubling-down, sincere clarifications, friendly roasting, neutral identity mentions, and plain everyday messages.
- Used for evaluation **only**, never training. It is the final judge of whether a model ships.

### 12.3 Parity test (Python ↔ Android)

- The training pipeline exports `parity_fixtures.json`: inputs plus the expected probabilities from the exported ONNX model.
- `:core` `ModelParityTest` runs the same inputs through the Kotlin encoder + desktop ONNX Runtime **on the PC** and asserts the probabilities match within 1e-3. Because `:core` is the exact code the app runs, no device is needed for this.
- Run all PC-side tests with `cd android && ./gradlew :core:test`. Tests that need the model bundle or fixtures are skipped (not failed) when those files are missing.

### 12.4 On-device benchmark

- A debug screen runs 200 inferences on fixture inputs and reports p50/p95/max latency. Record S25 results in §6.1 / §9.

### 12.5 Manual QA checklist (per app adapter)

- [ ] Pill appears for a known risky draft within ~0.5 s of pausing.
- [ ] No pill for normal messages and friendly swearing.
- [ ] Pill disappears after sending.
- [ ] Reply target is detected when replying to a specific message.
- [ ] Nothing happens in apps that aren't enabled.
- [ ] Password fields are ignored.
- [ ] Typing never lags; the host app never crashes.
- [ ] Service survives the screen turning off and on, and app switching.

---

## 13. Failure modes and troubleshooting runbook

| Symptom | Likely cause | Fix |
|---|---|---|
| Pill never appears | Service disabled (Samsung may disable it after an update/crash) | Check the settings screen status; re-enable in Accessibility settings. |
| Cannot enable the service ("Restricted setting") | Android 13+ sideload restriction | App info → ⋮ → *Allow restricted settings*, then enable. |
| Works, then stops after a while | One UI battery optimization killed it | Add to *Never sleeping apps*; verify it's not in *Deep sleeping apps*. |
| Pill appears but context is ignored / wrong | App UI changed and the adapter misparses the tree | Dump the node tree (debug menu), update that app's `AppAdapter`. `GenericAdapter` is the fallback. |
| Accuracy on device much worse than in Python | Tokenizer or context formatting mismatch | Run the tokenizer unit test (§12.1) and the parity test (§12.3). Check `contract_version`. |
| "Model error" in settings | Missing/corrupt asset or unsupported `contract_version` | Verify the three files in `assets/model/`; rebuild the bundle with `06`/`07`. |
| First check is slow, later ones fast | No warm-up run | Confirm warm-up in `OnnxClassifier` init. |
| Pill flickers while typing | Hysteresis gap too small or debounce too short | Widen `t_on − t_off`; raise `display_delay_ms`. |
| Too many false positives | Thresholds too low / training data imbalance | Raise thresholds via `model_config.json`; add benign banter to training; check the "Not helpful" counters. |
| Misses obvious cases | Recall too low | Lower thresholds; add scenarios to synthetic data; consider MiniLM-L12. |
| Typing lag in host app | Work on the main thread | Ensure tree walks and inference use the background dispatchers (§5.11). |
| Pill covers the keyboard or the wrong spot | Stale input bounds | Refresh bounds on every text-change event; fall back to top-center. |

---

## 14. Roadmap and phase acceptance criteria

### Phase 1 — Android pipeline with a stand-in model
Build service, context cache, adapters, tokenizer, ONNX runner, decision engine, overlay, settings. Use an existing open-source toxicity classifier (BERT-family, exported to ONNX) or a quick fine-tune on public data, as long as it satisfies the §7 contract.

**Done when:** on the S25, the pill appears for overt toxic drafts in Discord within ~0.5 s, the inference benchmark is recorded, and the manual QA checklist (§12.5) passes except for the context-quality items.

**Status (2026-10-04): PC side complete; device testing pending.**
- ✅ App + core built; manifest has **no permissions at all** (verified in the merged manifest).
- ✅ 27/27 `:core` tests pass: wordpiece 113/113, encoder 32/32, model parity worst diff 5e-7, plus decision/wordlist/parser unit tests.
- ✅ Stand-in model trained and bundled (§6.1).
- ✅ Emulator (Pixel 8a, API 34, x86_64) via the test playground:
  - model load 1.3–4.9 s (cold), warm-up ~78 ms
  - first check after warm-up 6 ms
  - 200-run benchmark: 9 tokens p50 3.2 ms; **72 tokens (3 context messages) p50 12.3 ms, p95 13.9 ms**
  - "see you at practice tomorrow" → NONE (risk 1.6%); a direct insult → HIGH/harassment (95.5%); "damn that game was crazy" → MEDIUM (false positive, known stand-in issue)
- ✅ **S25 (SM-S931U, Android 16, arm64), 2026-10-06:** model load 239 ms, warm-up 27 ms; 200-run benchmark at **65 tokens (3 context messages): p50 3.53 ms, p95 3.56 ms, max 3.58 ms**. Well inside the §9 budget (≤ 20 ms typical).
- ⏳ On the S25: enable the service, use the debug tree dump to tune `DiscordAdapter`, run the §12.5 checklist.
- Fixed during Phase 1: warming up with only a tiny input left the first real check at ~170 ms. Warm-up now also runs a near-max-length input.

### Phase 2 — status log

**Round 1 data (2026-10-06):**
- Spec: `training/synthetic/SPEC.md`.
- 4,109 synthetic records generated in-session by 9 parallel agents (batches 01–09, 19 scenarios, 40% should-warn, 73% with context).
- 2,000 borderline civil_comments re-labeled blind under the SPEC (stage `02a_select_relabel.py`).
- 275-case golden set (`training/golden/golden_set.jsonl`), including a 10-case core family generalized from the motivating incident.
- Blind cross-check of 306 random synthetic records: **97.7% warn/no-warn agreement (Cohen's κ 0.95)**. Caveat: both labelers are Claude, so this shows consistency, not agreement with human judgment.
- Golden leakage check: 5 exact draft overlaps (short stock phrases) removed from training in `03b_build_phase2.py`.

**Round 1 models (2026-10-06):** `03b_build_phase2.py` → 59,925 train records (34k cleaned public + 5.2k focus ×5 = 43% focus). Masked BCE, 4 epochs, ~13 min each on the RTX 2060.

| Model | Golden recall (should-warn) | Golden false alarms | Synthetic test F1 | Public test recall / FPR |
|---|---|---|---|---|
| Stand-in (Phase 1) | 19% | 17% | — | — |
| **bge-small-en-v1.5 (`p2-bge-v1`)** | **82%** | 20% | 0.888 | 0.80 / 0.065 |
| all-MiniLM-L12-v2 (`p2-l12-v1`) | 78% | 20% | 0.841 | 0.63 / 0.032 |

- **bge-small selected** (D9).
- Probes: "I like slaves" now 93% (stand-in 45%); swearing, "slaving away at work" and history-class mentions are no longer flagged.
- **Diagnosed failure:** the model learned "sensitive topic in context ⇒ risky reply" (e.g. a supportive reply to a coming-out at 88%, "lol same" at 93%). Causes: training contexts were mostly 1 message (the golden set is mostly 2–3+), very few short drafts (75), and few kind replies in sensitive conversations.

**Decisions (2026-10-06/07):**
- **No calibration to a single friend group.** One group's reactions are a biased label source (local politics, in-jokes). The model follows the SPEC's neutral "typical friend group" standard, applied the same way across the political spectrum. Group-specific tolerance belongs to later per-user features (sensitivity preset, "Not helpful" feedback), not the model. Example: a "sir, a second ___ has hit the towers" meme is borderline (0.3–0.7) under the neutral standard, so a heads-up is acceptable even if a particular group laughed.
- **The golden set is now a dev set.** Its failures were inspected to design round 2, so it is no longer a blind exam. Before release, create a fresh held-out set the model has never influenced, ideally 50–100 messages written by the owner.

**Round 2 data (2026-10-07):** 3 targeted batches against the diagnosed shortcut, 1,297 records with 0 validation errors: `sensitive_context_kind_reply` / `sensitive_context_mocking_reply` (batch 10), `short_reaction` (batch 11), long 3–5 message multi-speaker threads across all scenarios (batch 12). Totals: 5,406 synthetic, 537 drafts of ≤ 3 words (was 75), 1,599 records with ≥ 3 context messages.

**Round 2 model `p2-bge-v2` (bge-small, now bundled in the app as `p2-bge-2026.10.0`):**

| | Stand-in | bge v1 | **bge v2** | Gate |
|---|---|---|---|---|
| Golden recall (should-warn) | 19% | 82% | **84.0%** | ≥ 85% |
| Golden false alarms | 17% | 20% | **11.5%** | ≤ 5% |
| Core incident family (10) | 6 | 7 | **9** | — |
| Probe sentences (14) | 6 | 11 | **13** | — |

- Fixed shortcut examples: supportive reply to a coming-out 0.58 → 0.04; "lol same" 0.93 → 0.02; "second ___ hit the towers" meme 0.85 → 0.25.
- Remaining false alarms: mostly affectionate insult-banter ("shut up you absolute walnut"), "slave wages" idiom, history-project planning, flirting.
- Remaining misses: implicit stereotypes ("is he one of the smart ones or the scam ones"), mean sarcasm, a threat with concrete detail ("bringing something to school…" at 0.05, a serious miss), body-focused sexual comments.
- Public test at the shipped threshold: recall 0.65, FPR 0.04. The risk threshold is tuned on the focus sources, which trades public recall for chat precision.

**Quantization decision (2026-10-07), golden set, same thresholds:**

| Variant | Size | Recall | False alarms | Decisions changed vs fp32 |
|---|---|---|---|---|
| fp32 | 134 MB | 84.0% | 11.5% | — |
| **MatMul-only per-channel (shipped)** | **70 MB** | **83.2%** | **11.5%** | **1/275** |
| + embeddings fp16 | 46.5 MB | 83.2% | 12.8% | 5 |
| MatMul+Gather per-channel | 34 MB | 80.7% | 12.2% | 9 |
| All ops per-tensor (Phase 1 recipe) | 34 MB | 77.3% | 9.6% | 11 |

Quantizing the 30k×384 word-embedding table causes most of the damage. `07_quantize.py` now uses MatMul-only per-channel (test risk F1 drop 0.06 pts, gate ≤ 1 ✅). Kotlin↔Python parity on the new bundle: 5e-7. Getting back to ~35 MB without accuracy loss is future work (quantization-aware training, or vocabulary pruning, which would need a new contract version).

**On-device (S25, 2026-10-07), v2 bundle:** load 354 ms, warm-up 30 ms, 4.5 ms per check (19 tokens). The phone matches the PC.

**Surface robustness finding (2026-10-07):** v2's risk for "I like slaves" after the "slave away" message was 73% with a comma in the context and **35% without it**. Fixes:
- `03b_build_phase2.py` now makes upsampling repeats 1..k-1 **noisy-surface variants** (punctuation dropped or spaced, lowercased, apostrophes removed, "!"/"?" dropped or doubled, one adjacent-letter typo; same label). `--no-augment` restores plain repeats.
- New stage `05c_robustness.py`: 3 noisy variants per golden record, reports the decision flip rate and recall/false alarms on the noisy set.

| | v2 | **v3 (`p2-bge-v3`, shipped as `p2-bge-2026.10.1`)** |
|---|---|---|
| Comma case (with / without comma) | 73% / 35% | **94% / 94%** |
| Decision flip rate under noise | 4.5% | **3.6%** |
| Public test recall / FPR (shipped thresholds) | 0.68 / 0.04 | 0.77 / 0.06 |
| Golden recall / false alarms (own thresholds) | 84.0% / 11.5% | 79.0% / 12.2% |
| Golden AUC clean / noisy | 0.917 / 0.904 | 0.912 / 0.903 |
| Golden recall at 10% / 5% false alarms | 78% / 55% | 73% / 61% |

v2 and v3 are statistically tied on the golden set (275 cases; the headline gap is mostly threshold placement). v3 shipped for robustness to real messy chat, better public recall, and better recall at the 5% false-alarm target. int8 recipe unchanged (risk F1 drop −0.15 pts). Parity 5e-7.

**Ceiling note:** with bge-small and the current data, golden AUC plateaus around 0.91. Reaching the gates (85% recall at ≤ 5% false alarms) likely needs the D10 step 2 teacher and/or more human-reviewed data.

**Next steps:**
1. ~~Install v2 on the S25 and benchmark.~~ Done (v3 installed).
2. Owner reviews debatable golden labels and writes a fresh held-out set.
3. Consider the D10 step 2 teacher (DeBERTa-v3-base) to push recall and false alarms toward the gates.
4. Tune `DiscordAdapter` on the device.

### Phase 2 — Dataset
Public data ingestion, synthetic generation, golden set authoring, unified dataset build.

**Done when:** train/val/test JSONL files exist with ≥ 5k context records, the golden set has ≥ 150 cases, and a ≥5% spot-check has been completed.

### Phase 3 — Train, evaluate, swap
Fine-tune MiniLM, tune thresholds, export, quantize, pass the parity test, drop into the app.

**Done when:** all §8.6 release gates pass and the context-dependent cases in the golden set are caught on device.

### Later (not committed)
- **Explanations and rewrites on tap** via an on-device LLM (Gemini Nano through AICore on supported phones like the S25). It runs only when the user taps the pill, so its latency is acceptable.
- **More app adapters** (WhatsApp, Samsung Messages, Instagram, Telegram).
- **Personal calibration:** adjust thresholds per user from "Not helpful" counters (still on-device).
- **Hybrid mode:** optional opt-in cloud second opinion for uncertain cases, behind a backend with auth and rate limits (never an API key in the app).
- **"Explain this thread":** help interpret how a group is reacting (likely needs the LLM tier).

---

## 15. Repository layout

```
Keyboard/
├── SOURCE_OF_TRUTH.md          ← this file
├── android/                    Android Studio project (Gradle root; open this folder)
│   ├── settings.gradle.kts, build.gradle.kts, gradle.properties
│   ├── core/                   pure-Kotlin module (see §5.2)
│   │   └── src/{main,test}/kotlin/dev/wakeitup/core/
│   └── app/
│       ├── build.gradle.kts
│       └── src/
│           ├── main/
│           │   ├── AndroidManifest.xml
│           │   ├── assets/model/        model.onnx, vocab.txt, model_config.json
│           │   ├── assets/wordlist.txt
│           │   ├── java/dev/wakeitup/  (see §5.2)
│           │   └── res/xml/accessibility_service_config.xml
│           └── test/resources/fixtures/ wordpiece / encoder / parity fixtures (from training)
└── training/
    ├── requirements.txt
    ├── configs/train.yaml
    ├── scripts/01_…07_*.py
    ├── golden/golden_set.jsonl
    ├── data/                   (gitignored) raw + processed datasets
    └── runs/                   (gitignored) checkpoints, reports, exports
```

---

## 16. Open questions

| # | Question | Default until decided |
|---|---|---|
| Q1 | Final package ID | Name decided: **Wake It Up**. Package `dev.wakeitup` is a placeholder; the applicationId is permanent once on Play, so confirm (e.g. a domain you own) before the first upload. |
| Q2 | ~~MiniLM-L6 vs L12~~ | **Resolved (D9):** train all-MiniLM-L12-v2 and bge-small-en-v1.5, ship the golden-set winner |
| Q3 | Show the pill only after a pause, or live while typing? | Pause (300 ms); live mode as a setting |
| Q4 | Which apps after Discord? | Decide after Phase 1 |
| Q5 | Exact ONNX Runtime version and execution provider | Decide by on-device benchmark |
| Q6 | Dataset licenses for commercial distribution | Must resolve before any public release |
| Q7 | Wordlist source and maintenance | Small curated list; review periodically |

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **Accessibility service** | Android component that can observe UI events and read on-screen content in other apps, with user permission. |
| **Overlay** | A floating window drawn above other apps. |
| **Draft** | The text currently typed but not yet sent. |
| **Context** | The last few messages visible in the conversation, including the reply target. |
| **Reply target** | The specific message the user is replying to, if the app shows it. |
| **Classifier** | A model that outputs scores for fixed categories (no free text). |
| **MiniLM** | A small, distilled BERT-style transformer, well suited to fast on-device inference. |
| **ONNX / ONNX Runtime** | A portable model format and the engine that runs it on Android. |
| **Quantization (int8)** | Storing weights as 8-bit integers to shrink the model and speed it up. |
| **WordPiece** | The tokenizer used by BERT-family models; splits words into sub-word pieces. |
| **Distillation** | Using a large model to produce labeled data that teaches a small model. |
| **Golden set** | A hand-curated evaluation set of realistic hard cases; never trained on. |
| **Hysteresis** | Using separate on/off thresholds so the UI doesn't flicker near the boundary. |
| **Debounce** | Waiting for input to settle before acting. |
