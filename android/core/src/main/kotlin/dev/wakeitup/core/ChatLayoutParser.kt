package dev.wakeitup.core

/** A piece of visible text on screen, flattened from the accessibility node tree. */
data class ScreenText(
    val text: String,
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
    val editable: Boolean = false,
)

/**
 * Turns visible screen text into an anonymized [ConversationSnapshot] (SOURCE_OF_TRUTH.md §5.4).
 *
 * Layout heuristic shared by most chat apps: an author label sits on the same row as a
 * timestamp, and the lines below it (until the next author) are that author's messages.
 * Apps without author labels still work; their lines get a single unknown speaker.
 *
 * Pure logic with no Android types, so it is unit-tested on the JVM with synthetic layouts.
 */
class ChatLayoutParser(
    /** Matches a "replying to <name>" banner near the input; group 1 = name. Null = no reply detection. */
    private val replyBanner: Regex? = null,
    /** Extra app-specific UI strings to ignore (compared case-insensitively, trimmed). */
    private val chrome: Set<String> = emptySet(),
    private val maxMessages: Int = 8,
    private val rowTolerancePx: Int = 24,
) {
    private val chromeLower = (DEFAULT_CHROME + chrome).map { it.lowercase() }.toSet()

    /**
     * @param items all visible text on screen (any order)
     * @param inputTop top edge of the message input field in screen px, or null if unknown
     * @param selfName the user's display name in this app, mapped to "ME"; null if unknown
     */
    fun parse(items: List<ScreenText>, inputTop: Int?, selfName: String?, nowMs: Long = 0L): ConversationSnapshot {
        val visible = items
            .filter { !it.editable && it.text.isNotBlank() }
            .filter { inputTop == null || it.top < inputTop }
            .sortedWith(compareBy<ScreenText> { it.top }.thenBy { it.left })

        // Reply banner ("Replying to X") — remember the name, drop the line.
        var replyName: String? = null
        val lines = ArrayList<ScreenText>(visible.size)
        for (item in visible) {
            val match = replyBanner?.matchEntire(item.text.trim())
            if (match != null) {
                replyName = match.groupValues[1].trim()
            } else if (!isChrome(item.text)) {
                lines += item
            }
        }

        // Author labels: the nearest earlier item on the same row as a timestamp.
        val isTimestamp = BooleanArray(lines.size) { isTimestamp(lines[it].text) }
        val isAuthor = BooleanArray(lines.size)
        for (i in lines.indices) {
            if (!isTimestamp[i]) continue
            for (j in i - 1 downTo 0) {
                if (kotlin.math.abs(lines[j].top - lines[i].top) > rowTolerancePx) break
                if (!isTimestamp[j] && lines[j].left < lines[i].left) {
                    isAuthor[j] = true
                    break
                }
            }
        }

        // Build (author, text) messages.
        val raw = ArrayList<Pair<String, String>>()
        var currentAuthor = UNKNOWN_AUTHOR
        for (i in lines.indices) {
            when {
                isTimestamp[i] -> Unit
                isAuthor[i] -> currentAuthor = lines[i].text.trim()
                else -> raw += currentAuthor to lines[i].text.trim()
            }
        }
        val recent = raw.takeLast(maxMessages)

        // Anonymize: self → ME, everyone else → A, B, C... by first appearance.
        val names = HashMap<String, String>()
        var next = 0
        fun alias(name: String): String {
            if (selfName != null && selfName.isNotBlank() && name.equals(selfName.trim(), ignoreCase = true)) return SELF
            return names.getOrPut(name.lowercase()) { speakerLetter(next++) }
        }
        val messages = recent.map { (author, text) -> ChatMessage(alias(author), text) }

        val replyIndex = replyName?.let { name ->
            recent.indices.lastOrNull { recent[it].first.equals(name, ignoreCase = true) }
        }
        return ConversationSnapshot(messages, replyIndex, nowMs)
    }

    private fun isChrome(text: String): Boolean {
        val t = text.trim()
        if (t.isEmpty()) return true
        if (t.lowercase() in chromeLower) return true
        if (REACTION_COUNT.matches(t)) return true
        return false
    }

    companion object {
        const val SELF = "ME"
        private const val UNKNOWN_AUTHOR = "\u0000unknown"

        private val DEFAULT_CHROME = setOf("(edited)", "edited", "new", "new messages", "unread", "jump to present")
        private val REACTION_COUNT = Regex("""^\d{1,4}$""")

        private val TIMESTAMPS = listOf(
            Regex("""^(today|yesterday)(\s+at)?\s+\d{1,2}:\d{2}(\s?[ap]\.?m\.?)?$""", RegexOption.IGNORE_CASE),
            Regex("""^\d{1,2}:\d{2}(\s?[ap]\.?m\.?)?$""", RegexOption.IGNORE_CASE),
            Regex("""^\d{1,2}/\d{1,2}/\d{2,4}(,?\s+\d{1,2}:\d{2}(\s?[ap]\.?m\.?)?)?$""", RegexOption.IGNORE_CASE),
            Regex(
                """^(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\.?""" +
                    """(\s+\d{1,2}:\d{2}(\s?[ap]\.?m\.?)?)?$""",
                RegexOption.IGNORE_CASE,
            ),
        )

        fun isTimestamp(text: String): Boolean {
            val t = text.trim()
            return TIMESTAMPS.any { it.matches(t) }
        }

        /** A, B, ..., Z, then A2, B2, ... */
        fun speakerLetter(index: Int): String {
            val letter = ('A' + index % 26).toString()
            return if (index < 26) letter else letter + (index / 26 + 1)
        }
    }
}
