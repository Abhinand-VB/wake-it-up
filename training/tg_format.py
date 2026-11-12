"""WakeItUp input encoding (normative reference, SOURCE_OF_TRUTH.md §6.2 / §6.2.1).

The Kotlin `ml/InputEncoder.kt` must reproduce `encode()` exactly; parity is
checked with fixtures exported by `scripts/08_export_fixtures.py`.

wordpiece(text) reproduces Hugging Face's *slow* (pure-Python) BertTokenizer
with do_lower_case=True: BasicTokenizer (clean text, CJK padding, lowercase,
strip accents, punctuation split) followed by greedy longest-match WordPiece.
Literal special-token strings inside user text (e.g. "[SEP]") are NOT treated
as special tokens; they are tokenized like any other text.

Run `python tg_format.py` for the self-test.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# Contract constants (§7 model_config.json)
MAX_LEN = 128
CONTEXT_MAX_MESSAGES = 5
CONTEXT_SEPARATOR = " | "      # joins messages; becomes the single token "|"
REPLY_PREFIX = "reply_to "
EMPTY_CONTEXT_TOKEN = "none"
LABELS = ["risk", "hate_slur", "insensitive", "harassment", "threat", "sexual", "misread_risk"]

BASE_MODEL = "nreimers/MiniLM-L6-H384-uncased"
DEFAULT_VOCAB = Path(__file__).resolve().parent / "assets" / "vocab.txt"


@lru_cache(maxsize=4)
def get_tokenizer(vocab_path: str | None = None):
    """Slow (reference) BertTokenizer. Uses training/assets/vocab.txt if present,
    else the base model's vocab from the HF hub."""
    try:  # transformers >= 5: the pure-Python implementation lives here
        from transformers.models.bert.tokenization_bert_legacy import BertTokenizerLegacy as BertTokenizer
    except ImportError:  # transformers 4.x: BertTokenizer is the slow one
        from transformers import BertTokenizer

    vocab_path = vocab_path or os.environ.get("TG_VOCAB")
    if vocab_path is None and DEFAULT_VOCAB.exists():
        vocab_path = str(DEFAULT_VOCAB)
    if vocab_path:
        tok = BertTokenizer(vocab_file=vocab_path, do_lower_case=True,
                            tokenize_chinese_chars=True, strip_accents=None)
    else:
        tok = BertTokenizer.from_pretrained(BASE_MODEL, do_lower_case=True,
                                            tokenize_chinese_chars=True, strip_accents=None)
    assert not getattr(tok, "is_fast", False), "reference tokenizer must be the slow BertTokenizer"
    return tok


def wordpiece(text: str, tok=None) -> list[str]:
    """BERT uncased basic tokenization + WordPiece. Returns token strings."""
    tok = tok or get_tokenizer()
    out: list[str] = []
    # never_split=None: special-token-looking strings are split like normal text.
    for word in tok.basic_tokenizer.tokenize(text, never_split=None):
        out.extend(tok.wordpiece_tokenizer.tokenize(word))
    return out


def wordpiece_ids(text: str, tok=None) -> list[int]:
    tok = tok or get_tokenizer()
    return tok.convert_tokens_to_ids(wordpiece(text, tok))


def _msg_get(m, key: str, idx: int) -> str:
    if isinstance(m, Mapping):
        return m[key]
    return m[idx]


def format_messages(messages: Sequence, reply_index: int | None) -> list[str]:
    """Steps 1-3 of §6.2.1: reorder reply target, keep last 5, format strings."""
    items = [(_msg_get(m, "speaker", 0), _msg_get(m, "text", 1), False) for m in messages]
    # 1. move reply target to the end, marked as reply
    #    (an out-of-range reply_index is ignored, i.e. treated as null)
    if reply_index is not None and 0 <= reply_index < len(items):
        sp, tx, _ = items.pop(reply_index)
        items.append((sp, tx, True))
    # 2. keep only the last context_max_messages
    items = items[-CONTEXT_MAX_MESSAGES:]
    # 3. format
    return [(REPLY_PREFIX if is_reply else "") + f"{sp}: {tx}" for sp, tx, is_reply in items]


def encode_tokens(messages: Sequence, reply_index: int | None, draft: str,
                  max_len: int = MAX_LEN, tok=None) -> tuple[list[str], list[str]]:
    """Steps 1-6 of §6.2.1. Returns (context_tokens, draft_tokens) as strings."""
    tok = tok or get_tokenizer()
    formatted = format_messages(messages, reply_index)

    # 4. draft tokens, protected; keep LAST tokens if too long
    d = wordpiece(draft, tok)
    budget = max_len - 3
    if len(d) > budget - 1:
        d = d[len(d) - (budget - 1):]
    c_budget = budget - len(d)  # always >= 1

    # 5. context tokens; drop oldest while too long.
    msgs = [wordpiece(m, tok) for m in formatted]

    def cost(ms):
        return sum(len(m) for m in ms) + (len(ms) - 1)

    # The loop never drops the last remaining message; that one is trimmed to
    # its LAST c_budget tokens instead (a single long message never becomes "none").
    while len(msgs) > 1 and cost(msgs) > c_budget:
        msgs.pop(0)
    if len(msgs) == 1 and len(msgs[0]) > c_budget:
        msgs[0] = msgs[0][len(msgs[0]) - c_budget:]

    # 6. join with the single "|" token; "none" if empty
    c: list[str] = []
    for i, m in enumerate(msgs):
        if i:
            c.append("|")
        c.extend(m)
    if not c:
        c = wordpiece(EMPTY_CONTEXT_TOKEN, tok)
    return c, d


def encode(messages: Sequence, reply_index: int | None, draft: str,
           max_len: int = MAX_LEN, tok=None) -> dict:
    """Full §6.2.1 encoding. messages: list of {"speaker","text"} dicts or (speaker, text)
    tuples, oldest -> newest. Returns dict(input_ids, token_type_ids, attention_mask)
    as python int lists (batch size 1, no padding)."""
    tok = tok or get_tokenizer()
    c, d = encode_tokens(messages, reply_index, draft, max_len, tok)
    seg_a = [tok.cls_token] + c + [tok.sep_token]
    seg_b = d + [tok.sep_token]
    input_ids = tok.convert_tokens_to_ids(seg_a + seg_b)
    token_type_ids = [0] * len(seg_a) + [1] * len(seg_b)
    attention_mask = [1] * len(input_ids)
    assert len(input_ids) <= max_len
    return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}


# --------------------------------------------------------------------------- self-test
def _self_test() -> None:
    tok = get_tokenizer()
    cls, sep = tok.cls_token_id, tok.sep_token_id
    none_id = tok.convert_tokens_to_ids("none")
    bar_id = tok.convert_tokens_to_ids("|")
    assert bar_id != tok.unk_token_id, "'|' must be in vocab"
    assert wordpiece(" | ") == ["|"]

    # draft-only
    e = encode([], None, "hello there")
    assert e["input_ids"] == [cls, none_id, sep] + wordpiece_ids("hello there") + [sep]
    assert e["token_type_ids"] == [0, 0, 0, 1, 1, 1]

    # context join equals tokenizing the joined string (separator is a lone "|")
    msgs = [{"speaker": "A", "text": "anyone up for food later"},
            {"speaker": "B", "text": "cant, still working"}]
    c, _ = encode_tokens(msgs, 0, "only if someone else is paying lol")
    joined = " | ".join(["B: cant, still working", "reply_to A: anyone up for food later"])
    assert c == wordpiece(joined), (c, wordpiece(joined))

    # out-of-range reply_index is ignored (treated as null)
    assert encode(msgs, 7, "x") == encode(msgs, None, "x")
    assert encode(msgs, -1, "x") == encode(msgs, None, "x")

    # >5 messages keeps last 5
    many = [{"speaker": "A", "text": f"m{i}"} for i in range(8)]
    c, _ = encode_tokens(many, None, "ok")
    assert c == wordpiece(" | ".join(f"A: m{i}" for i in range(3, 8)))

    # long draft: keep last budget-1 tokens, context gets exactly 1 token
    long_draft = " ".join(f"w{i}" for i in range(300))
    e = encode(msgs, None, long_draft)
    assert len(e["input_ids"]) == MAX_LEN
    c, d = encode_tokens(msgs, None, long_draft)
    assert len(d) == MAX_LEN - 4 and len(c) == 1
    assert d[-1] == wordpiece(long_draft)[-1]

    # single huge message: keep its last c_budget tokens
    huge = [{"speaker": "A", "text": " ".join(["word"] * 400) + " end"}]
    c, d = encode_tokens(huge, None, "hi")
    assert len(c) == MAX_LEN - 3 - len(d) and c[-1] == "end"

    # oldest-drop
    m3 = [{"speaker": "A", "text": " ".join(["x"] * 90)},
          {"speaker": "B", "text": " ".join(["y"] * 40)},
          {"speaker": "C", "text": "short one"}]
    c, d = encode_tokens(m3, None, "draft text")
    assert c == wordpiece("B: " + " ".join(["y"] * 40) + " | C: short one")

    # empty draft
    e = encode([], None, "")
    assert e["input_ids"] == [cls, none_id, sep, sep] and e["token_type_ids"] == [0, 0, 0, 1]
    print("tg_format self-test OK")


if __name__ == "__main__":
    _self_test()
