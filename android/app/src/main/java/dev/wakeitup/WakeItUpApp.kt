package dev.wakeitup

import android.app.Application
import android.content.Context
import dev.wakeitup.core.ToneEngine
import dev.wakeitup.core.Wordlist
import dev.wakeitup.settings.Settings
import dev.wakeitup.util.SafeLog

class WakeItUpApp : Application() {
    lateinit var settings: Settings
        private set
    lateinit var engineHolder: EngineHolder
        private set

    override fun onCreate() {
        super.onCreate()
        settings = Settings(this)
        engineHolder = EngineHolder(this)
    }

    companion object {
        fun from(context: Context): WakeItUpApp = context.applicationContext as WakeItUpApp
    }
}

/**
 * Loads the model bundle from assets once per process (SOURCE_OF_TRUTH.md §5.7, §7).
 * Loading is blocking: call [getOrLoad] from a background thread.
 */
class EngineHolder(private val context: Context) {
    sealed interface Status {
        data object NotLoaded : Status
        data object Loading : Status
        data class Ready(val modelVersion: String, val warmUpMs: Double, val loadMs: Double) : Status
        data class Failed(val reason: String) : Status
    }

    @Volatile
    var status: Status = Status.NotLoaded
        private set

    private var engine: ToneEngine? = null
    private var wordlist: Wordlist? = null

    @Synchronized
    fun getOrLoad(): ToneEngine? {
        engine?.let { return it }
        if (status is Status.Failed) return null
        status = Status.Loading
        return try {
            val t0 = System.nanoTime()
            val assets = context.assets
            val configJson = assets.open("$MODEL_DIR/model_config.json").bufferedReader().use { it.readText() }
            val vocab = assets.open("$MODEL_DIR/vocab.txt").bufferedReader().use { it.readLines() }
            val modelBytes = assets.open("$MODEL_DIR/model.onnx").use { it.readBytes() }
            val loaded = ToneEngine.load(vocab, configJson, modelBytes)
            val loadMs = (System.nanoTime() - t0) / 1e6
            val warmUpMs = loaded.warmUp()
            engine = loaded
            status = Status.Ready(loaded.config.modelVersion, warmUpMs, loadMs)
            SafeLog.info("model ${loaded.config.modelVersion} loaded in ${loadMs.toInt()} ms, warm-up ${warmUpMs.toInt()} ms")
            loaded
        } catch (t: Throwable) {
            SafeLog.error("model load", t)
            status = Status.Failed("${t.javaClass.simpleName}: ${t.message ?: "no details"}")
            null
        }
    }

    @Synchronized
    fun wordlist(): Wordlist {
        wordlist?.let { return it }
        val loaded = try {
            Wordlist.parse(context.assets.open("wordlist.txt").bufferedReader().use { it.readLines() })
        } catch (t: Throwable) {
            SafeLog.error("wordlist load", t)
            Wordlist.EMPTY
        }
        wordlist = loaded
        return loaded
    }

    /** Allows a retry after a failed load (e.g. from the settings screen). */
    @Synchronized
    fun resetFailure() {
        if (status is Status.Failed) status = Status.NotLoaded
    }

    private companion object {
        const val MODEL_DIR = "model"
    }
}
