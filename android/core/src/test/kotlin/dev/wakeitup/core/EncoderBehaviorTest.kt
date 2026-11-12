package dev.wakeitup.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Truncation and formatting rules of §6.2.1, checked with a tiny synthetic vocab (no fixtures needed). */
class EncoderBehaviorTest {
    private val vocab = listOf("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "|", ":", "a", "b", "me", "none", "reply", "_", "to", "x", "y", "z")
    private val tokenizer = WordPieceTokenizer(vocab)
    private fun id(t: String) = vocab.indexOf(t).toLong()

    private fun tokens(e: EncodedInput) = e.inputIds.map { vocab[it.toInt()] }

    @Test
    fun emptyContextBecomesNone() {
        val e = InputEncoder(tokenizer).encode(emptyList(), null, "x y")
        assertEquals(listOf("[CLS]", "none", "[SEP]", "x", "y", "[SEP]"), tokens(e))
        assertEquals(listOf(0L, 0L, 0L, 1L, 1L, 1L), e.tokenTypeIds.toList())
        assertTrue(e.attentionMask.all { it == 1L })
    }

    @Test
    fun replyTargetMovesToEndWithPrefix() {
        val messages = listOf(ChatMessage("A", "x"), ChatMessage("B", "y"), ChatMessage("A", "z"))
        val e = InputEncoder(tokenizer).encode(messages, 0, "x")
        assertEquals(
            listOf("[CLS]", "b", ":", "y", "|", "a", ":", "z", "|", "reply", "_", "to", "a", ":", "x", "[SEP]", "x", "[SEP]"),
            tokens(e),
        )
    }

    @Test
    fun keepsOnlyLastFiveMessages() {
        val messages = List(7) { ChatMessage("A", "x") }
        val e = InputEncoder(tokenizer).encode(messages, null, "y")
        assertEquals(5, tokens(e).count { it == "a" })
    }

    @Test
    fun dropsOldestMessagesFirstButNeverTheLastOne() {
        // maxLen 12 → budget 9; draft "y" (1) → context budget 8.
        // Messages are "A: x" = 3 tokens; three of them = 3+3+3+2 separators = 11 > 8 → drop oldest → 3+3+1 = 7.
        val config = EncoderConfig(maxLen = 12)
        val messages = listOf(ChatMessage("A", "x"), ChatMessage("B", "y"), ChatMessage("A", "z"))
        val e = InputEncoder(tokenizer, config).encode(messages, null, "y")
        assertEquals(listOf("[CLS]", "b", ":", "y", "|", "a", ":", "z", "[SEP]", "y", "[SEP]"), tokens(e))
    }

    @Test
    fun singleLongMessageKeepsItsLastTokens() {
        // maxLen 10 → budget 7; draft 1 token → context budget 6; message "A: x x x x x x x" = 9 tokens.
        val config = EncoderConfig(maxLen = 10)
        val e = InputEncoder(tokenizer, config).encode(listOf(ChatMessage("A", "x x x x x x z")), null, "y")
        assertEquals(listOf("[CLS]", "x", "x", "x", "x", "x", "z", "[SEP]", "y", "[SEP]"), tokens(e))
        assertEquals(10, e.length)
    }

    @Test
    fun longDraftKeepsItsLastTokensAndLeavesOneForContext() {
        val config = EncoderConfig(maxLen = 8) // budget 5 → draft max 4
        val e = InputEncoder(tokenizer, config).encode(listOf(ChatMessage("A", "x")), null, "x x x y z")
        assertEquals(listOf("[CLS]", "x", "[SEP]", "x", "x", "y", "z", "[SEP]"), tokens(e))
        assertEquals(id("[CLS]"), e.inputIds.first())
    }

    @Test
    fun outOfRangeReplyIndexIsIgnored() {
        val messages = listOf(ChatMessage("A", "x"))
        val a = InputEncoder(tokenizer).encode(messages, 5, "y")
        val b = InputEncoder(tokenizer).encode(messages, null, "y")
        assertEquals(a.inputIds.toList(), b.inputIds.toList())
    }
}
