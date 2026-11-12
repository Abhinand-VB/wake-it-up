package dev.wakeitup.core

/** Category-based explanation templates shown in the detail card (SOURCE_OF_TRUTH.md §5.8). */
object Explanations {
    private val titles = mapOf(
        "hate_slur" to "Targets a group",
        "insensitive" to "Could read as insensitive",
        "harassment" to "May read as an attack",
        "threat" to "Could read as a threat",
        "sexual" to "Sexual content",
        "misread_risk" to "Easy to misread",
    )

    private val bodies = mapOf(
        "hate_slur" to "Contains a slur or language that targets a group. This is likely to hurt people even as a joke.",
        "insensitive" to "Could read as making light of a serious or painful topic, especially out of context.",
        "harassment" to "May come across as an attack on someone in the chat rather than banter.",
        "threat" to "Could be read as a threat, even if you're joking.",
        "sexual" to "Sexual content that may not fit this conversation.",
        "misread_risk" to "The joke depends on context. Someone skimming may take it literally.",
    )

    private const val GENERAL_TITLE = "Might land badly"
    private const val GENERAL_BODY =
        "Some people in the chat may read this differently than you mean it. Worth a second look before sending."

    fun title(category: String?): String = titles[category] ?: GENERAL_TITLE
    fun body(category: String?): String = bodies[category] ?: GENERAL_BODY
}
