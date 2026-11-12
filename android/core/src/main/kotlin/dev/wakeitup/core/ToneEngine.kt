package dev.wakeitup.core

class CheckResult(
    /** Per-label probabilities, in [ModelConfig.labels] order. */
    val probs: FloatArray,
    val encodeMs: Double,
    val inferenceMs: Double,
    val tokenCount: Int,
)

/** Tokenizer + encoder + classifier for one loaded model bundle. Use from a single thread. */
class ToneEngine(
    val config: ModelConfig,
    private val encoder: InputEncoder,
    private val classifier: Classifier,
) : AutoCloseable {

    fun check(messages: List<ChatMessage>, replyIndex: Int?, draft: String): CheckResult {
        val t0 = System.nanoTime()
        val input = encoder.encode(messages, replyIndex, draft)
        val t1 = System.nanoTime()
        val probs = classifier.predict(input)
        val t2 = System.nanoTime()
        check(probs.size == config.labels.size) {
            "model returned ${probs.size} outputs, config lists ${config.labels.size} labels"
        }
        return CheckResult(probs, (t1 - t0) / 1e6, (t2 - t1) / 1e6, input.length)
    }

    /**
     * Throwaway runs so the first real check isn't slow (SOURCE_OF_TRUTH.md §5.7).
     * Uses a short and a near-max-length input: ONNX Runtime allocates buffers per input
     * shape, and warming up with only a tiny input left the first real check at ~170 ms.
     * Returns total ms.
     */
    fun warmUp(): Double {
        val t0 = System.nanoTime()
        check(listOf(ChatMessage("A", "warm up")), null, "warm up")
        val filler = List(5) { ChatMessage("A", "warm up message number $it with some extra words to fill space") }
        repeat(2) { check(filler, null, "a longer warm up draft with a reasonable number of words in it") }
        return (System.nanoTime() - t0) / 1e6
    }

    override fun close() {
        (classifier as? AutoCloseable)?.close()
    }

    companion object {
        fun load(vocabLines: List<String>, configJson: String, modelBytes: ByteArray, threads: Int = 2): ToneEngine {
            val config = ModelConfig.parse(configJson)
            val tokenizer = WordPieceTokenizer(vocabLines, doLowerCase = config.doLowerCase)
            return ToneEngine(config, InputEncoder(tokenizer, config.encoder), OnnxClassifier(modelBytes, threads))
        }
    }
}
