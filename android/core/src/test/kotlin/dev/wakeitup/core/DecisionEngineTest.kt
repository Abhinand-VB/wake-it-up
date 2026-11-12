package dev.wakeitup.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class DecisionEngineTest {
    private val labels = listOf("risk", "hate_slur", "insensitive", "harassment", "threat", "sexual", "misread_risk")
    private val thresholds = mapOf(
        "risk" to 0.60f, "hate_slur" to 0.50f, "insensitive" to 0.55f, "harassment" to 0.60f,
        "threat" to 0.55f, "sexual" to 0.65f, "misread_risk" to 0.99f,
    )

    private fun probs(risk: Float, vararg others: Pair<String, Float>): FloatArray {
        val p = FloatArray(labels.size)
        p[0] = risk
        for ((label, value) in others) p[labels.indexOf(label)] = value
        return p
    }

    @Test
    fun belowAllThresholdsIsSafe() {
        val v = DecisionEngine(labels, thresholds).evaluate(probs(0.3f), currentlyShowing = false)
        assertEquals(RiskLevel.NONE, v.level)
    }

    @Test
    fun riskAboveThresholdIsMediumWithoutCategory() {
        val v = DecisionEngine(labels, thresholds).evaluate(probs(0.65f), currentlyShowing = false)
        assertEquals(RiskLevel.MEDIUM, v.level)
        assertNull(v.category)
    }

    @Test
    fun picksCategoryWithLargestMargin() {
        val v = DecisionEngine(labels, thresholds)
            .evaluate(probs(0.7f, "insensitive" to 0.6f, "harassment" to 0.9f), currentlyShowing = false)
        assertEquals("harassment", v.category)
    }

    @Test
    fun veryHighRiskOrSlurIsHigh() {
        val engine = DecisionEngine(labels, thresholds)
        assertEquals(RiskLevel.HIGH, engine.evaluate(probs(0.85f), false).level)
        assertEquals(RiskLevel.HIGH, engine.evaluate(probs(0.62f, "hate_slur" to 0.7f), false).level)
    }

    @Test
    fun hysteresisKeepsWarningUntilClearlyBelow() {
        val engine = DecisionEngine(labels, thresholds)
        // 0.55 is below t_on (0.60) but above t_off (0.50).
        assertEquals(RiskLevel.NONE, engine.evaluate(probs(0.55f), currentlyShowing = false).level)
        assertEquals(RiskLevel.MEDIUM, engine.evaluate(probs(0.55f), currentlyShowing = true).level)
        assertEquals(RiskLevel.NONE, engine.evaluate(probs(0.45f), currentlyShowing = true).level)
    }

    @Test
    fun sensitivityShiftsThresholds() {
        val p = probs(0.55f)
        assertEquals(RiskLevel.MEDIUM, DecisionEngine(labels, thresholds, Sensitivity.STRICT).evaluate(p, false).level)
        assertEquals(RiskLevel.NONE, DecisionEngine(labels, thresholds, Sensitivity.NORMAL).evaluate(p, false).level)
        assertEquals(RiskLevel.NONE, DecisionEngine(labels, thresholds, Sensitivity.RELAXED).evaluate(probs(0.65f), false).level)
    }

    @Test
    fun untrainedLabelWithHighThresholdNeverTriggersAlone() {
        val v = DecisionEngine(labels, thresholds).evaluate(probs(0.2f, "misread_risk" to 0.9f), false)
        assertEquals(RiskLevel.NONE, v.level)
    }
}
