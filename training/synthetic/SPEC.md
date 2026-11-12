# Wake It Up — Training Data Spec (Phase 2)

This is the rulebook for every person or model that **writes or labels** Wake It Up training data.
It implements SOURCE_OF_TRUTH.md §6.3 (labels), §8 (pipeline) and D10 (distillation).
If you change a rule here, regenerate or relabel the affected data.

## 1. What we are labeling

A **draft** is a message the user (speaker `ME`) is about to send in a casual group chat
(Discord-style, friends/classmates/gaming servers), possibly after some **context** messages.

The core question for `risk`:

> If ME sends this draft in this chat, would a reasonable member of a typical friend group
> likely be **offended, hurt, or seriously misread** ME (e.g. think ME endorses something hateful)?

Judge **how it reads to others**, not ME's intent. A harmless intent that reads badly is still risky.
But do **not** flag ordinary banter, swearing, opinions, or dark-ish humor that friend groups normally
shrug off. Over-flagging is a failure too: a warning that cries wolf gets ignored.

## 2. Labels

### 2.1 `risk` — graded, one of: `0.0`, `0.3`, `0.7`, `1.0`

| Value | Meaning | Examples (illustrative) |
|---|---|---|
| `0.0` | Fine. Nobody reasonable would mind. | "see u at 7", "damn that game was crazy", "lmao you're such an idiot love u bro" (clear mutual banter) |
| `0.3` | A bit edgy; some people might wince, most friends won't care. **No warning.** | mild crude joke, blunt but fair criticism |
| `0.7` | **Likely to come across badly** to someone in the chat. Warn. | a joke that, read literally, endorses something harmful; a polite-sounding stereotype |
| `1.0` | Clearly offensive / hurtful / hateful. Warn strongly. | endorsing an atrocity, a targeted vicious insult, a threat that reads real |

Training treats `risk >= 0.5` as "warn".

### 2.2 Categories — list zero or more (only when risk ≥ 0.7, except where noted)

| Category | Use when the draft… |
|---|---|
| `hate_slur` | attacks, dehumanizes, or uses slurs against a protected group (race, ethnicity, nationality, religion, gender, sexual orientation, gender identity, disability). **Never spell out real slurs**; see §4. |
| `insensitive` | trivializes or jokes about atrocities, tragedies, abuse, suicide, illness, death; expresses stereotypes even "politely"; mocks identity |
| `harassment` | attacks/insults a person in the chat beyond clear mutual banter; mean-spirited mockery; telling someone to self-harm |
| `threat` | threatens violence/harm in a way that could read as real (even if joking) |
| `sexual` | sexual content clearly inappropriate for the chat (unsolicited, about a person in chat, explicit) |
| `misread_risk` | the problem is **mainly context/irony**: a joke or sarcasm that only works if you know ME's intent, and reads as endorsing something bad when taken literally. Can be combined with others (e.g. `["insensitive","misread_risk"]`). May also be used at risk 0.3 for mildly ambiguous jokes. |

`risk 0.0` → categories must be `[]`.

### 2.3 `rationale`
One short sentence (≤ 25 words) on **why**, written for the user ("Reads as endorsing slavery when taken literally, even though it riffs on 'slave away'."). Not used for training; used for review.

## 3. Record format (JSONL, one object per line, UTF-8)

```json
{"id": "syn-03-0042", "source": "synthetic", "scenario": "trivializing_atrocity",
 "pair_id": "p03-0021",
 "context": [{"speaker": "A", "text": "..."}, {"speaker": "B", "text": "..."}],
 "reply_to": 0,
 "draft": "...",
 "risk": 0.7, "categories": ["insensitive", "misread_risk"],
 "rationale": "..."}
```

- `id`: `syn-<batch 2 digits>-<4 digits>`, unique.
- `scenario`: one of the scenario keys in §5 (exact spelling).
- `pair_id`: set on both members of a **contrastive pair** (§6); `null` otherwise.
- `context`: 0–5 prior messages, oldest → newest. Speakers: `ME` (the user, earlier messages), `A`, `B`, `C`, … (others, letters assigned by first appearance). **No real names** in speaker fields. Names inside message text should be rare, generic and fictional.
- `reply_to`: index into `context` of the message ME is explicitly replying to, or `null`.
- `draft`: ME's new message, raw text (no "ME:" prefix).
- About **60% of records should have context** (1–5 messages); 40% may be draft-only (`context: []`).

## 4. Content rules

- **No real slurs.** For cases that would need a slur, write a different hateful-but-slur-free message, or skip. (Slurs are handled by a separate word list.)
- No real private people. Public figures only in ordinary political opinion examples, never as targets of harassment/sexual content.
- Sexual content: PG-13 maximum. Imply, don't describe.
- Violence: no graphic detail.
- No minors in sexual contexts, ever.
- Keep offensive examples **realistic and representative**, not maximally extreme. The model needs to learn the *boundary*, not the worst case.

## 5. Scenarios

| Key | Description | Typical label |
|---|---|---|
| `riff_on_previous_word` | ME builds a joke on a word/phrase from the previous message; read literally it endorses/trivializes something harmful | risky (0.7–1.0, `misread_risk` + usually `insensitive`) |
| `riff_harmless` | Same structure (riff on a previous word) but the result is harmless | 0.0–0.3 |
| `ironic_doubling_down` | After a bad reaction, ME doubles down ironically/sarcastically, which reads as sincere | risky (0.7–1.0) |
| `sincere_clarification` | After a bad reaction, ME clarifies, apologizes, or laughs it off sincerely | 0.0 |
| `trivializing_atrocity` | Jokes/praise/indifference about slavery, genocide, the Holocaust, terrorist attacks, mass shootings, famines, etc. | risky (often 1.0) |
| `respectful_history` | Mentions the same topics respectfully, educationally, or as ordinary conversation | 0.0 |
| `polite_stereotype` | Generalizations about a group phrased calmly/"just asking"/"no offense but" | risky (0.7–1.0, `insensitive`, sometimes `hate_slur`) |
| `neutral_identity_mention` | Ordinary mentions of race, religion, gender, sexuality, disability, nationality | 0.0 |
| `friendly_banter` | Roasting, swearing, mock insults among friends that clearly read as affectionate | 0.0–0.3 |
| `targeted_harassment` | Genuinely mean insults/mockery at someone in the chat; piling on | risky (`harassment`) |
| `joking_threat_risky` | "Joking" threats that read as real or menacing | risky (`threat`) |
| `hyperbole_harmless` | Obvious hyperbole ("i'm gonna kill you if you spoil the finale", "this exam is killing me") | 0.0 |
| `sexual_inappropriate` | Unsolicited/explicit/about someone in the chat | risky (`sexual`) |
| `flirty_ok` | Mild flirting, innuendo friends would shrug off | 0.0–0.3 |
| `political_offensive` | Political takes that demean groups or celebrate harm | risky |
| `political_ordinary` | Ordinary political/opinionated takes, even strong disagreement | 0.0–0.3 |
| `dark_humor_borderline` | Dark humor about death, tragedy, illness in general; label honestly 0.3 vs 0.7 | mixed |
| `sarcasm_misread` | Sarcasm that could be read literally as hurtful | mixed, `misread_risk` |
| `everyday_chat` | Plain daily chat: plans, games, school, memes, food, reactions | 0.0 |
| `sensitive_context_kind_reply` | *(round 2)* The context touches a sensitive topic (identity disclosure, grief, illness, tragedy in the news, a history/school project on an atrocity, discrimination someone experienced, politics) and ME replies kindly, supportively, neutrally, practically or with gentle humor that clearly doesn't punch down | 0.0 |
| `sensitive_context_mocking_reply` | *(round 2)* Same kind of context, but ME's reply mocks, dismisses, or punches down at the person or the affected group | risky |
| `short_reaction` | *(round 2)* Very short drafts (1–4 words or emoji: "lol same", "real", "W", "rip", "nah 💀", "based", "deserved", "lmaooo", "f") whose meaning depends entirely on the context. Usually 0.0; risky when the reaction endorses or laughs at something hateful/harmful in the context | mixed |

**Round 2 note (why these exist):** the round-1 model learned the shortcut "a sensitive topic in the context ⇒ the reply is risky", and it rarely saw short replies or long threads. Round-2 data must show that **the reply decides the label, not the topic**, and must use realistic multi-speaker threads (2–5 messages, several speakers, side chatter, reactions).

## 6. Contrastive pairs (important)

At least **30% of records** in risky scenarios should come in **pairs** with a matching harmless twin
(same `pair_id`): near-identical wording or the same context, opposite labels. Examples of the *shape*:

- Context "he's been slaving away all week" → draft riffing to endorse slavery (risky) vs. draft riffing harmlessly about the workload (fine).
- "damn you're useless" (harassment) vs. "damn that's crazy" (fine).
- "no offense but [group] are bad at X" (risky) vs. "no offense but [team] is bad at X" (fine).

Pairs teach the model the **meaning** difference instead of word shortcuts.

## 7. Style: make it messy and real

Mix styles across records: lowercase, no punctuation, typos, abbreviations (fr, ngl, idk, lmao, bro, ong),
emoji, ALL CAPS reactions, run-ons, very short ("ok", "lol same"), up to ~40 words. Vary topics widely
(games, school, sports, music, work, food, memes, family, news). **Do not repeat templates**: no two
records should share the same sentence frame except deliberate contrastive pairs.

## 8. Quality checklist before writing a batch

- [ ] Labels follow §2; risk 0.0 has empty categories
- [ ] ~60% have context; reply_to valid or null
- [ ] ≥30% of risky records paired with harmless twins
- [ ] Harmless records are at least ~45% of the batch
- [ ] No real slurs, no graphic content, no real private people
- [ ] Varied, messy, non-templated phrasing
