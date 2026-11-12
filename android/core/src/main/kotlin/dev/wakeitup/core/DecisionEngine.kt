package dev.wakeitup.core

enum class Sensitivity(val thresholdShift: Float) {
    RELAXED(0.10f),
    NORMAL(0f),
    STRICT(-0.10f),
}

enum class RiskLevel { NONE, MEDIUM, HIGH }

data class Verdict(
    val level: RiskLevel,
    /** Most likely category label (e.g. "insensitive"), or null for a general warning. */
    val category: String?,
    val risk: Float,
) {
    companion object {
        val SAFE = Verdict(RiskLevel.NONE, null, 0f)
        val WORDLIST_HIT = Verdict(RiskLevel.HIGH, "hate_slur", 1f)
    }
}

/**
 * Turns per-label probabilities into a warning decision (SOURCE_OF_TRUTH.md §5.8):
 * thresholds from model_config.json, shifted by the sensitivity preset, with hysteresis
 * so the pill doesn't flicker while the user types.
 */
class DecisionEngine(
    private val labels: List<String>,
    private val thresholds: Map<String, Float>,
    var sensitivity: Sensitivity = Sensitivity.NORMAL,
    private val hysteresis: Float = 0.10f,
    private val highMargin: Float = 0.20f,
) {
    private val riskIndex = labels.indexOf(ModelConfig.RISK_LABEL).also {
        require(it >= 0) { "labels must include '${ModelConfig.RISK_LABEL}'" }
    }

    fun threshold(label: String): Float =
        ((thresholds[label] ?: 0.5f) + sensitivity.thresholdShift).coerceIn(0.05f, 0.99f)

    /** @param currentlyShowing whether a warning is on screen now (enables the lower "off" threshold). */
    fun evaluate(probs: FloatArray, currentlyShowing: Boolean): Verdict {
        require(probs.size == labels.size) { "expected ${labels.size} probs, got ${probs.size}" }
        val slack = if (currentlyShowing) hysteresis else 0f
        val risk = probs[riskIndex]
        val riskThreshold = threshold(ModelConfig.RISK_LABEL)

        var bestCategory: String? = null
        var bestMargin = Float.NEGATIVE_INFINITY
        for (i in labels.indices) {
            if (i == riskIndex) continue
            val margin = probs[i] - (threshold(labels[i]) - slack)
            if (margin >= 0f && margin > bestMargin) {
                bestCategory = labels[i]
                bestMargin = margin
            }
        }

        val flagged = risk >= riskThreshold - slack || bestCategory != null
        if (!flagged) return Verdict(RiskLevel.NONE, null, risk)

        val high = risk >= (riskThreshold + highMargin).coerceAtMost(0.99f) ||
            bestCategory == "hate_slur" || bestCategory == "threat"
        return Verdict(if (high) RiskLevel.HIGH else RiskLevel.MEDIUM, bestCategory, risk)
    }
}
