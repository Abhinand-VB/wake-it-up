package dev.wakeitup.core

import org.json.JSONObject

class UnsupportedModelException(message: String) : Exception(message)

/** Parsed `model_config.json` — the app side of the model contract (SOURCE_OF_TRUTH.md §7). */
data class ModelConfig(
    val contractVersion: Int,
    val modelVersion: String,
    val baseModel: String,
    val doLowerCase: Boolean,
    val encoder: EncoderConfig,
    val labels: List<String>,
    val thresholds: Map<String, Float>,
) {
    companion object {
        const val SUPPORTED_CONTRACT_VERSION = 1
        const val RISK_LABEL = "risk"

        fun parse(json: String): ModelConfig {
            val o = JSONObject(json)
            val contract = o.getInt("contract_version")
            if (contract != SUPPORTED_CONTRACT_VERSION) {
                throw UnsupportedModelException(
                    "model contract_version $contract, app supports $SUPPORTED_CONTRACT_VERSION",
                )
            }
            val labelsJson = o.getJSONArray("labels")
            val labels = List(labelsJson.length()) { labelsJson.getString(it) }
            if (RISK_LABEL !in labels) throw UnsupportedModelException("labels must include '$RISK_LABEL'")

            val thresholdsJson = o.getJSONObject("thresholds")
            val thresholds = labels.associateWith { label ->
                if (thresholdsJson.has(label)) thresholdsJson.getDouble(label).toFloat() else 0.5f
            }
            return ModelConfig(
                contractVersion = contract,
                modelVersion = o.getString("model_version"),
                baseModel = o.optString("base_model", "unknown"),
                doLowerCase = o.optBoolean("do_lower_case", true),
                encoder = EncoderConfig(
                    maxLen = o.optInt("max_len", 128),
                    contextMaxMessages = o.optInt("context_max_messages", 5),
                    contextSeparator = o.optString("context_separator", " | "),
                    replyPrefix = o.optString("reply_prefix", "reply_to "),
                    emptyContextToken = o.optString("empty_context_token", "none"),
                ),
                labels = labels,
                thresholds = thresholds,
            )
        }
    }
}
