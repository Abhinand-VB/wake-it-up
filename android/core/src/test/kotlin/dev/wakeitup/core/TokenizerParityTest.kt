package dev.wakeitup.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Kotlin WordPiece + encoder must match the Python reference exactly (SOURCE_OF_TRUTH.md §12.1).
 * A mismatch here silently degrades accuracy on device, so every case must pass.
 */
class TokenizerParityTest {

    @Test
    fun wordpieceMatchesPythonReference() {
        val tokenizer = WordPieceTokenizer(Fixtures.vocab())
        val cases = Fixtures.fixture("wordpiece_fixtures.json").getJSONArray("cases")
        val failures = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val text = case.getString("text")
            val expected = List(case.getJSONArray("tokens").length()) { case.getJSONArray("tokens").getString(it) }
            val actual = tokenizer.tokenize(text)
            if (actual != expected) failures += "text=${text.take(80)}\n  expected=$expected\n  actual  =$actual"
            if (case.has("ids")) {
                val expectedIds = Fixtures.ints(case.getJSONArray("ids"))
                val actualIds = tokenizer.tokenizeToIds(text).toList()
                if (actualIds != expectedIds) failures += "ids mismatch for text=${text.take(80)}"
            }
        }
        println("wordpiece: ${cases.length() - failures.size}/${cases.length()} cases match")
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun encoderMatchesPythonReference() {
        val tokenizer = WordPieceTokenizer(Fixtures.vocab())
        val fixture = Fixtures.fixture("encoder_fixtures.json")
        val config = ModelConfig.parse(Fixtures.configJson()).encoder.copy(maxLen = fixture.optInt("max_len", 128))
        val encoder = InputEncoder(tokenizer, config)
        val cases = fixture.getJSONArray("cases")
        val failures = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val name = case.optString("name", "case $i")
            val encoded = encoder.encode(
                Fixtures.messages(case.optJSONArray("messages")),
                Fixtures.replyIndex(case),
                case.getString("draft"),
            )
            val expectedIds = Fixtures.ints(case.getJSONArray("input_ids"))
            val expectedTypes = Fixtures.ints(case.getJSONArray("token_type_ids"))
            if (encoded.inputIds.map { it.toInt() } != expectedIds) {
                failures += "$name: input_ids differ\n  expected=$expectedIds\n  actual  =${encoded.inputIds.toList()}"
            } else if (encoded.tokenTypeIds.map { it.toInt() } != expectedTypes) {
                failures += "$name: token_type_ids differ"
            }
            if (case.has("attention_mask")) {
                assertEquals("$name attention_mask", Fixtures.ints(case.getJSONArray("attention_mask")), encoded.attentionMask.map { it.toInt() })
            }
        }
        println("encoder: ${cases.length() - failures.size}/${cases.length()} cases match")
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }
}
