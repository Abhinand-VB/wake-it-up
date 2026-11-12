package dev.wakeitup.ui

import android.app.Activity
import android.graphics.Typeface
import android.os.Bundle
import android.text.InputType
import android.widget.EditText
import android.widget.TextView
import dev.wakeitup.BuildConfig
import dev.wakeitup.WakeItUpApp
import dev.wakeitup.core.EncodedInput
import dev.wakeitup.core.OnnxClassifier
import dev.wakeitup.core.ChatMessage
import dev.wakeitup.core.DecisionEngine
import dev.wakeitup.core.Explanations
import dev.wakeitup.core.RiskLevel
import java.util.concurrent.Executors

/**
 * Manual playground + on-device benchmark (SOURCE_OF_TRUTH.md §12.4). Runs the exact
 * same engine as the service. Works on an emulator, so it doubles as a PC-side smoke test.
 */
class TestActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var contextInput: EditText
    private lateinit var draftInput: EditText
    private lateinit var output: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(page {
            heading("Test playground")
            body("Recent messages, one per line, like \"A: message\". Start a line with > to mark the message being replied to. Leave empty for draft-only.")
            contextInput = EditText(context).apply {
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
                minLines = 3
                hint = "A: anyone up for food later\nB: cant, still working"
            }.also { addView(it) }
            body("Your draft")
            draftInput = EditText(context).apply {
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
                hint = "Type a message to check"
            }.also { addView(it) }
            button("Check") { runCheck() }
            button("Benchmark (200 runs)") { runBenchmark() }
            if (BuildConfig.DEBUG) {
                button("Benchmark candidate models (debug)") { runCandidateBenchmark() }
            }
            output = body("").apply { typeface = Typeface.MONOSPACE }
        })
    }

    private fun parseContext(): Pair<List<ChatMessage>, Int?> {
        val messages = ArrayList<ChatMessage>()
        var replyIndex: Int? = null
        for (raw in contextInput.text.lines()) {
            var line = raw.trim()
            if (line.isEmpty()) continue
            val isReply = line.startsWith(">")
            if (isReply) line = line.removePrefix(">").trim()
            val speaker = line.substringBefore(':', "").trim().uppercase().ifEmpty { "A" }
            val text = if (':' in line) line.substringAfter(':').trim() else line
            if (isReply) replyIndex = messages.size
            messages += ChatMessage(speaker, text)
        }
        return messages to replyIndex
    }

    private fun runCheck() {
        val (messages, replyIndex) = parseContext()
        val draft = draftInput.text.toString()
        output.text = "Checking…"
        worker.execute {
            val app = WakeItUpApp.from(this)
            val engine = app.engineHolder.getOrLoad()
            val text = if (engine == null) {
                "Model failed to load: ${app.engineHolder.status}"
            } else if (app.engineHolder.wordlist().matches(draft)) {
                "Wordlist hit → HIGH (hate_slur)"
            } else {
                val result = engine.check(messages, replyIndex, draft)
                val verdict = DecisionEngine(engine.config.labels, engine.config.thresholds, app.settings.sensitivity)
                    .evaluate(result.probs, currentlyShowing = false)
                buildString {
                    append("Verdict: ").append(verdict.level)
                    if (verdict.level != RiskLevel.NONE) {
                        append(" · ").append(Explanations.title(verdict.category))
                    }
                    append("\n\n")
                    engine.config.labels.forEachIndexed { i, label ->
                        append("%-13s %5.1f%%  (thr %.0f%%)\n".format(label, result.probs[i] * 100, engine.config.thresholds[label]!! * 100))
                    }
                    append("\ntokens %d · encode %.2f ms · inference %.2f ms".format(result.tokenCount, result.encodeMs, result.inferenceMs))
                }
            }
            runOnUiThread { output.text = text }
        }
    }

    private fun runBenchmark() {
        val (messages, replyIndex) = parseContext()
        val draft = draftInput.text.toString().ifBlank { "this is a reasonably typical chat message to time the model with" }
        output.text = "Benchmarking…"
        worker.execute {
            val engine = WakeItUpApp.from(this).engineHolder.getOrLoad()
            val text = if (engine == null) {
                "Model failed to load"
            } else {
                repeat(10) { engine.check(messages, replyIndex, draft) }
                val times = DoubleArray(200) { engine.check(messages, replyIndex, draft).inferenceMs }.sorted()
                val tokens = engine.check(messages, replyIndex, draft).tokenCount
                "Inference over 200 runs (%d tokens)\np50 %.2f ms\np95 %.2f ms\nmax %.2f ms".format(
                    tokens, times[100], times[189], times.last(),
                )
            }
            runOnUiThread { output.text = text }
        }
    }

    /**
     * Debug only: times every *.onnx pushed to <external files>/bench (speed only; inputs are
     * synthetic token ids). Used to pick a Phase 2 base model (SOURCE_OF_TRUTH.md §6.1).
     * Push with: adb push model.onnx /sdcard/Android/data/dev.wakeitup/files/bench/
     */
    private fun runCandidateBenchmark() {
        val dir = getExternalFilesDir("bench")
        val models = dir?.listFiles { f -> f.name.endsWith(".onnx") }?.sortedBy { it.length() }.orEmpty()
        if (models.isEmpty()) {
            output.text = "No models in ${dir?.absolutePath}"
            return
        }
        output.text = "Benchmarking ${models.size} models…"
        worker.execute {
            val lengths = intArrayOf(32, 64, 128)
            val sb = StringBuilder("median ms (2 threads: 32/64/128 tok | 1 thread: 128 tok)\n")
            for (file in models) {
                try {
                    val bytes = file.readBytes()
                    val row = StringBuilder("%-18s %4d MB ".format(file.name.removeSuffix(".int8.onnx"), file.length() / 1_000_000))
                    for (threads in intArrayOf(2, 1)) {
                        OnnxClassifier(bytes, threads).use { clf ->
                            for (len in if (threads == 2) lengths else intArrayOf(128)) {
                                val ids = LongArray(len) { 1000L + (it * 7919L) % 28000L }
                                val input = EncodedInput(ids, LongArray(len), LongArray(len) { 1L })
                                repeat(5) { clf.predict(input) }
                                val times = DoubleArray(40) {
                                    val t0 = System.nanoTime()
                                    clf.predict(input)
                                    (System.nanoTime() - t0) / 1e6
                                }.sorted()
                                row.append(if (threads == 1) " | " else " ").append("%.1f".format(times[20]))
                            }
                        }
                    }
                    sb.append(row).append('\n')
                    android.util.Log.i("WakeItUp", "bench $row")
                } catch (t: Throwable) {
                    sb.append("${file.name}: ${t.javaClass.simpleName}: ${t.message}\n")
                }
                val partial = sb.toString()
                runOnUiThread { output.text = partial }
            }
        }
    }

    override fun onDestroy() {
        worker.shutdown()
        super.onDestroy()
    }
}
