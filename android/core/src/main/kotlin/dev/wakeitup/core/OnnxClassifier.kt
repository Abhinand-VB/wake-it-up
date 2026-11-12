package dev.wakeitup.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer
import kotlin.math.exp

fun interface Classifier {
    /** Returns one sigmoid probability per label, in model_config.json label order. */
    fun predict(input: EncodedInput): FloatArray
}

/**
 * Runs the exported classifier with ONNX Runtime (SOURCE_OF_TRUTH.md §5.7).
 * Compiled against the desktop ONNX Runtime API; on Android the identical API is
 * provided by onnxruntime-android. Not thread-safe: call from one thread.
 */
class OnnxClassifier(modelBytes: ByteArray, intraOpThreads: Int = 2) : Classifier, AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputNames: Set<String>

    init {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(intraOpThreads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelBytes, options)
        inputNames = session.inputNames
        require(INPUT_IDS in inputNames) { "model has no '$INPUT_IDS' input (has $inputNames)" }
    }

    override fun predict(input: EncodedInput): FloatArray {
        val shape = longArrayOf(1, input.length.toLong())
        val tensors = LinkedHashMap<String, OnnxTensor>()
        try {
            tensors[INPUT_IDS] = OnnxTensor.createTensor(env, LongBuffer.wrap(input.inputIds), shape)
            // Exporters sometimes prune unused inputs; only feed what the graph declares.
            if (ATTENTION_MASK in inputNames) {
                tensors[ATTENTION_MASK] = OnnxTensor.createTensor(env, LongBuffer.wrap(input.attentionMask), shape)
            }
            if (TOKEN_TYPE_IDS in inputNames) {
                tensors[TOKEN_TYPE_IDS] = OnnxTensor.createTensor(env, LongBuffer.wrap(input.tokenTypeIds), shape)
            }
            session.run(tensors).use { result ->
                @Suppress("UNCHECKED_CAST")
                val logits = (result.get(0).value as Array<FloatArray>)[0]
                return FloatArray(logits.size) { sigmoid(logits[it]) }
            }
        } finally {
            tensors.values.forEach { it.close() }
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val INPUT_IDS = "input_ids"
        const val ATTENTION_MASK = "attention_mask"
        const val TOKEN_TYPE_IDS = "token_type_ids"

        private fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()
    }
}
