package dev.wakeitup.core

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume
import java.io.File

/** Locates the model bundle and training-exported fixtures (paths set in core/build.gradle.kts). */
object Fixtures {
    val modelDir = File(System.getProperty("wakeitup.modelDir") ?: "../app/src/main/assets/model")
    val fixturesDir = File(System.getProperty("wakeitup.fixturesDir") ?: "../app/src/test/resources/fixtures")

    fun requireFile(file: File): File {
        Assume.assumeTrue("missing ${file.path} — run the training pipeline first", file.isFile)
        return file
    }

    fun vocab(): List<String> = requireFile(File(modelDir, "vocab.txt")).readLines(Charsets.UTF_8)
    fun configJson(): String = requireFile(File(modelDir, "model_config.json")).readText(Charsets.UTF_8)
    fun modelBytes(): ByteArray = requireFile(File(modelDir, "model.onnx")).readBytes()
    fun fixture(name: String): JSONObject = JSONObject(requireFile(File(fixturesDir, name)).readText(Charsets.UTF_8))

    fun messages(array: JSONArray?): List<ChatMessage> {
        if (array == null) return emptyList()
        return List(array.length()) {
            val m = array.getJSONObject(it)
            ChatMessage(m.getString("speaker"), m.getString("text"))
        }
    }

    fun replyIndex(case: JSONObject): Int? =
        if (!case.has("reply_index") || case.isNull("reply_index")) null else case.getInt("reply_index")

    fun ints(array: JSONArray): List<Int> = List(array.length()) { array.getInt(it) }
}
