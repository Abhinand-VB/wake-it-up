package dev.wakeitup.core

/** One chat message. [speaker] is already anonymized: "ME", "A", "B", ... (SOURCE_OF_TRUTH.md §5.4). */
data class ChatMessage(val speaker: String, val text: String)

/** The recent conversation around the user's draft, oldest → newest. */
data class ConversationSnapshot(
    val messages: List<ChatMessage>,
    /** Index into [messages] of the message being replied to, if the app shows one. */
    val replyIndex: Int?,
    val capturedAtMs: Long,
) {
    companion object {
        val EMPTY = ConversationSnapshot(emptyList(), null, 0L)
    }
}
