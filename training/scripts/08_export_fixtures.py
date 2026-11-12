"""Stage 08: export JVM/instrumented test fixtures -> android/app/src/test/resources/fixtures/

  wordpiece_fixtures.json  slow BertTokenizer reference tokens/ids (§12.1)
  encoder_fixtures.json    tg_format.encode outputs (§6.2.1, §12.1)
  parity_fixtures.json     int8 model.onnx probabilities via onnxruntime (§12.3)

Uses the bundle installed in android/app/src/main/assets/model/ (run 07 first).
Also cross-checks BertTokenizerFast against the slow reference and prints mismatches
(the slow tokenizer is the reference either way).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

import tg_format  # noqa: E402
import tg_common  # noqa: E402
from tg_common import ANDROID_MODEL_DIR, MODEL_VERSION  # noqa: E402

WORDPIECE_TEXTS = [
    # basics / case
    "Hello World", "hello world", "HELLO WORLD", "HeLLo WoRLD", "ThIs Is MiXeD cAsE",
    "unaffable", "transformers are surprisingly good",
    # accents
    "café", "naïve", "résumé", "Crème Brûlée", "ÀÉÎÕÜ àéîõü", "Ångström", "São Paulo", "piñata",
    "café (combining acute)",
    # apostrophes / contractions
    "don't", "I'm", "you're", "can't won't shouldn't", "it's y'all's", "rock 'n' roll",
    # punctuation runs
    "wow!!!", "wait...", "really?!", "what?!?!", "no.....way", "(hi) [there] {you}", "a-b_c/d\\e",
    "$100 & 50% off #1 @home ~ ^ | ` = +", "[CLS] hello [SEP] [UNK] [PAD] [MASK]",
    # urls / mentions / tags
    "check https://example.com/path?q=1&x=y#frag", "www.example.org", "email me at someone@example.com",
    "@everyone look", "hey @john_doe123", "#blessed #NoFilter", "#1 fan",
    # numbers
    "3.14159", "1,000,000", "-42", "2nd 3rd 21st", "10:30pm", "1e10", "0.5x",
    # emojis
    "😂", "😂😂😂", "lol 😂 that's funny", "❤️", "👍🏽 ok", "🇺🇸 flag", "1️⃣ first", "👨‍👩‍👧 family", "this is 🔥🔥",
    # scripts
    "你好世界", "我爱你 and you", "日本語のテキスト", "こんにちは", "안녕하세요", "Привет, как дела?", "ЁЖИК",
    "مرحبا بالعالم", "שלום", "नमस्ते दुनिया", "สวัสดี", "Γειά σου Κόσμε",
    # whitespace / control / invisible
    "tab\there", "new\nline", "carriage\r\nreturn", "multiple   spaces", "  leading and trailing  ",
    "null\x00char", "bell\x07char", "escape\x1bseq", "zero​width", "zw‌non‍joiner",
    "soft­hyphen", "bom﻿here", "non breaking space", "line separator",
    "thin space", "ideographic　space", "rtl‏mark",
    # long / repeated / leet
    "a" * 101, "x" * 100, "supercalifragilisticexpialidocious" * 4, "para separator", "nelchar", "privateuse", "soooooo goooood", "nooooooo",
    "h4x0r l33t sp34k", "u r 2 g00d 4 me", "f*** this", "sh!t",
    # empty / whitespace-only
    "", " ", "   \t\n  ",
    # typographic
    "wait…", "…", "“quoted” text", "it’s fine", "‘single’", "em—dash", "en–dash",
    "«guillemets»", "¿qué?", "¡hola!", "• bullet", "™ © ®", "½ ¼", "ﬁne ligature", "ＦＵＬＬＷＩＤＴＨ",
]


def write_json(path: Path, obj) -> None:
    """UTF-8, LF line endings, non-ASCII kept literal (control chars are \\u-escaped by json)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write(chr(10))


def M(sp, tx):
    return {"speaker": sp, "text": tx}


def encoder_cases(tok):
    w = lambda n: " ".join(["word"] * n)  # noqa: E731  "word" is a single vocab token
    assert tg_format.wordpiece("word", tok) == ["word"]
    budget = tg_format.MAX_LEN - 3
    long_msg = "this is a very long message that keeps going " * 40
    cases = [
        ("no_context", [], None, "hello there"),
        ("no_context_punct", [], None, "lol that's WILD!!! 😂"),
        ("one_message", [M("A", "anyone up for food later")], None, "sure, where?"),
        ("two_messages", [M("A", "anyone up for food later"), M("B", "cant, still working")], None, "only if someone else is paying lol"),
        ("three_messages", [M("A", "hi"), M("B", "hey"), M("ME", "what's up")], None, "nm, you?"),
        ("spec_example_reply", [M("A", "anyone up for food later"), M("B", "cant, still working")], 0, "only if someone else is paying lol"),
        ("reply_to_last", [M("A", "first"), M("B", "second")], 1, "replying to second"),
        ("reply_to_middle", [M("A", "one"), M("B", "two"), M("C", "three"), M("A", "four")], 1, "re: two"),
        ("reply_index_out_of_range", [M("A", "one"), M("B", "two")], 5, "ignored reply index"),
        ("five_messages", [M("A", f"message {i}") for i in range(5)], None, "ok"),
        ("seven_messages_keep_last5", [M("ABCDEFG"[i], f"msg number {i}") for i in range(7)], None, "ok then"),
        ("seven_messages_reply_to_first", [M("ABCDEFG"[i], f"msg number {i}") for i in range(7)], 0, "replying to the first one"),
        ("seven_messages_reply_to_middle", [M("ABCDEFG"[i], f"msg number {i}") for i in range(7)], 3, "what about 3"),
        ("long_context_drop_oldest", [M("A", w(60)), M("B", w(40)), M("C", "short one")], None, "draft text here"),
        ("long_context_drop_two", [M("A", w(50)), M("B", w(50)), M("C", w(30)), M("A", "last")], None, "draft"),
        ("long_context_reply_kept", [M("A", "the replied message"), M("B", w(70)), M("C", w(60))], 0, "draft"),
        ("single_long_message_trim", [M("A", long_msg)], None, "hi"),
        ("single_long_message_trim_reply", [M("A", long_msg)], 0, "short reply"),
        ("drop_then_trim_last", [M("A", "older"), M("B", long_msg)], None, "hey"),
        ("draft_longer_than_budget", [M("A", "context gets one token")], None, w(200) + " end"),
        ("draft_longer_than_budget_no_context", [], None, " ".join(f"tok{i}" for i in range(150))),
        ("draft_exactly_budget_minus_1", [M("A", "hello there friend")], None, w(budget - 1)),
        ("draft_budget_tokens_truncated", [M("A", "hello there friend")], None, w(budget) + " last"),
        ("draft_budget_minus_2", [M("A", "hello there friend")], None, w(budget - 2)),
        ("empty_draft", [M("A", "you there?")], None, ""),
        ("empty_draft_no_context", [], None, ""),
        ("whitespace_draft", [], None, "   "),
        ("emoji_and_punct_messages", [M("A", "lmaooo 😂😂"), M("B", "wait... what?!"), M("C", "café @ 5pm? #plans")], 2, "I'm in!!! 🎉"),
        ("unicode_mixed", [M("A", "Привет"), M("B", "你好")], None, "naïve résumé…"),
        ("empty_message_text", [M("A", ""), M("B", "hello")], None, "hi"),
        ("pipe_in_text", [M("A", "a | b"), M("B", "c|d")], None, "e | f"),
        ("speaker_me", [M("ME", "i said this"), M("A", "and i replied")], None, "and now me again"),
    ]
    out = []
    for name, msgs, ri, draft in cases:
        e = tg_format.encode(msgs, ri, draft, tok=tok)
        out.append({"name": name, "messages": msgs, "reply_index": ri, "draft": draft,
                    "input_ids": e["input_ids"], "token_type_ids": e["token_type_ids"]})
    # sanity checks for the edge cases
    byname = {c["name"]: c for c in out}
    assert len(byname["draft_exactly_budget_minus_1"]["input_ids"]) == tg_format.MAX_LEN
    assert len(byname["draft_budget_tokens_truncated"]["input_ids"]) == tg_format.MAX_LEN
    assert len(byname["single_long_message_trim"]["input_ids"]) == tg_format.MAX_LEN
    none_id = tok.convert_tokens_to_ids("none")
    assert byname["single_long_message_trim"]["input_ids"][1] != none_id
    return out


PARITY = [
    # benign chat
    ([], None, "hey, are we still on for tonight?"),
    ([], None, "lol yeah that was hilarious"),
    ([], None, "thanks so much, really appreciate it"),
    ([], None, "running 10 min late, sorry!"),
    ([], None, "good morning everyone"),
    ([], None, "did anyone watch the game last night"),
    ([], None, "i love this song so much 😍"),
    ([], None, "can you send me the notes from class"),
    ([], None, "happy birthday!! 🎉🎉"),
    ([], None, "ok"),
    ([], None, "that movie was killer, the ending blew me away"),
    ([], None, "i'm going to kill this exam tomorrow"),
    # mild swearing / banter
    ([], None, "damn that's crazy"),
    ([], None, "holy shit that's amazing"),
    ([], None, "what the hell lol"),
    ([], None, "this traffic is so fucking annoying"),
    ([], None, "you lucky bastard, congrats"),
    # insults
    ([], None, "you're an idiot"),
    ([], None, "shut up, nobody asked you"),
    ([], None, "you are so stupid it's embarrassing"),
    ([], None, "what a pathetic loser"),
    ([], None, "go away, you're worthless"),
    # threats
    ([], None, "i will find you and hurt you"),
    ([], None, "say that again and i'll punch you"),
    ([], None, "you better watch your back"),
    ([], None, "i'm going to kill you"),
    # identity attacks (mild, no slurs)
    ([], None, "people from that country are all criminals"),
    ([], None, "women are too emotional to be leaders"),
    ([], None, "all religious people are idiots"),
    ([], None, "go back to where you came from"),
    # sexual
    ([], None, "send nudes"),
    # with context (encoder path)
    ([M("A", "anyone up for food later"), M("B", "cant, still working")], 0, "only if someone else is paying lol"),
    ([M("A", "i failed my driving test again"), M("B", "oh no")], 0, "you're hopeless lmao"),
    ([M("A", "my grandma passed away yesterday")], None, "sorry to hear that, sending love"),
    ([M("A", "x " * 200)], None, "you're an idiot"),
    ([], None, "word " * 200),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ANDROID_MODEL_DIR))
    ap.add_argument("--out-dir", default=str(tg_common.ANDROID_FIXTURES_DIR))
    args = ap.parse_args()
    OUT_DIR = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    vocab = model_dir / "vocab.txt"
    tok = tg_format.get_tokenizer(str(vocab))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- wordpiece fixtures + fast-vs-slow cross-check
    from transformers import BertTokenizerFast
    # transformers v5 ignores vocab_file= for the fast class; pass the vocab dict explicitly
    fast = BertTokenizerFast(vocab=dict(tok.vocab), do_lower_case=True, tokenize_chinese_chars=True, strip_accents=None)
    assert len(fast.get_vocab()) == len(tok.vocab)
    cases, mism = [], []
    for t in WORDPIECE_TEXTS:
        toks = tg_format.wordpiece(t, tok)
        ids = tok.convert_tokens_to_ids(toks)
        cases.append({"text": t, "tokens": toks, "ids": ids})
        fids = fast(t, add_special_tokens=False)["input_ids"]
        if fids != ids:
            mism.append({"text": t, "slow": toks, "fast": fast.convert_ids_to_tokens(fids)})
    assert len(cases) >= 60
    write_json(OUT_DIR / "wordpiece_fixtures.json",
               {"vocab": "assets/model/vocab.txt", "do_lower_case": True, "cases": cases})
    print(f"[08] wordpiece: {len(cases)} cases; fast!=slow on {len(mism)}")
    for m in mism:
        print("     MISMATCH", json.dumps(m, ensure_ascii=False))

    # ---- encoder fixtures
    enc = encoder_cases(tok)
    assert len(enc) >= 25
    write_json(OUT_DIR / "encoder_fixtures.json", {"max_len": tg_format.MAX_LEN, "cases": enc})
    print(f"[08] encoder: {len(enc)} cases")

    # ---- parity fixtures (int8 model.onnx as shipped)
    import onnxruntime as ort
    cfg = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    assert cfg["model_version"] == MODEL_VERSION
    sess = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    pcases = []
    for msgs, ri, draft in PARITY:
        e = tg_format.encode(msgs, ri, draft, tok=tok)
        feeds = {k: np.array([e[k]], dtype=np.int64) for k in ("input_ids", "attention_mask", "token_type_ids")}
        logits = sess.run(["logits"], feeds)[0][0].astype(np.float64)
        probs = (1.0 / (1.0 + np.exp(-logits))).tolist()
        pcases.append({"messages": msgs, "reply_index": ri, "draft": draft,
                       "input_ids": e["input_ids"], "probs": [round(p, 6) for p in probs]})
        print(f"     risk={probs[0]:.3f} {' '.join(f'{p:.2f}' for p in probs[1:])} | {draft[:50]!r}")
    assert len(pcases) >= 30
    write_json(OUT_DIR / "parity_fixtures.json",
               {"model_version": cfg["model_version"], "labels": cfg["labels"], "tolerance": 1e-3, "cases": pcases})
    print(f"[08] parity: {len(pcases)} cases -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
