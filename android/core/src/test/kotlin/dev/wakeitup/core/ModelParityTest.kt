package dev.wakeitup.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * End-to-end parity: Kotlin encoder + ONNX Runtime must reproduce the probabilities the
 * Python pipeline computed with the same model.onnx (SOURCE_OF_TRUTH.md §12.3), on the PC.
 */
class ModelParityTest {

    @Test
    fun probabilitiesMatchPythonReference() {
        val fixture = Fixtures.fixture("parity_fixtures.json")
        ToneEngine.load(Fixtures.vocab(), Fixtures.configJson(), Fixtures.modelBytes()).use { engine ->
            assertEquals("fixture/model version", engine.config.modelVersion, fixture.getString("model_version"))
            val cases = fixture.getJSONArray("cases")
            var worst = 0.0
            val failures = ArrayList<String>()
            for (i in 0 until cases.length()) {
                val case = cases.getJSONObject(i)
                val expected = case.getJSONArray("probs")
                val result = engine.check(Fixtures.messages(case.optJSONArray("messages")), Fixtures.replyIndex(case), case.getString("draft"))
                for (j in 0 until expected.length()) {
                    val diff = abs(result.probs[j] - expected.getDouble(j))
                    worst = maxOf(worst, diff)
                    if (diff > TOLERANCE) failures += "case $i label ${engine.config.labels[j]}: expected ${expected.getDouble(j)}, got ${result.probs[j]}"
                }
            }
            println("parity: ${cases.length()} cases, worst abs diff %.2e".format(worst))
            assertTrue(failures.joinToString("\n"), failures.isEmpty())
        }
    }

    /** Informational: JVM latency on this PC. Phone numbers come from the in-app benchmark. */
    @Test
    fun pcLatency() {
        ToneEngine.load(Fixtures.vocab(), Fixtures.configJson(), Fixtures.modelBytes()).use { engine ->
            val messages = listOf(
                ChatMessage("A", "anyone up for food later"),
                ChatMessage("B", "cant, still working on the project tonight"),
                ChatMessage("A", "we could order in and keep you company"),
            )
            val draft = "only if someone else is paying lol, my wallet is not ready for this week"
            repeat(20) { engine.check(messages, 0, draft) }
            val times = DoubleArray(200) { engine.check(messages, 0, draft).inferenceMs }.sorted()
            val tokens = engine.check(messages, 0, draft).tokenCount
            println("PC inference (%d tokens): p50 %.2f ms, p95 %.2f ms".format(tokens, times[100], times[189]))
        }
    }

    private companion object {
        const val TOLERANCE = 1e-3
    }
}
