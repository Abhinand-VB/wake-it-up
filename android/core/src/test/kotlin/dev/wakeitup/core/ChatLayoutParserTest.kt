package dev.wakeitup.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** Synthetic screen layouts standing in for a real chat app (SOURCE_OF_TRUTH.md §5.4). */
class ChatLayoutParserTest {
    private val discordLike = ChatLayoutParser(replyBanner = Regex("""^replying to\s+@?(.+)$""", RegexOption.IGNORE_CASE))

    private fun t(text: String, left: Int, top: Int, editable: Boolean = false) =
        ScreenText(text, left, top, left + 300, top + 40, editable)

    private val screen = listOf(
        t("Sam", 120, 100), t("Today at 9:41 PM", 260, 104),
        t("anyone up for food later", 120, 150),
        t("Riley", 120, 220), t("Today at 9:42 PM", 260, 222),
        t("cant, still working", 120, 270),
        t("2", 130, 320), // reaction count
        t("Alex", 120, 380), t("Today at 9:43 PM", 260, 381),
        t("same lol", 120, 430),
        t("Replying to Sam", 40, 1500),
        t("Message #general", 40, 1600, editable = true),
    )

    @Test
    fun groupsLinesUnderAuthorsAndAnonymizes() {
        val snap = discordLike.parse(screen, inputTop = 1600, selfName = "Alex")
        assertEquals(
            listOf(
                ChatMessage("A", "anyone up for food later"),
                ChatMessage("B", "cant, still working"),
                ChatMessage("ME", "same lol"),
            ),
            snap.messages,
        )
    }

    @Test
    fun detectsReplyTargetByAuthor() {
        val snap = discordLike.parse(screen, inputTop = 1600, selfName = null)
        assertEquals(0, snap.replyIndex)
    }

    @Test
    fun ignoresTextBelowTheInputAndEditableFields() {
        val snap = discordLike.parse(screen + t("keyboard suggestion", 40, 1700), inputTop = 1600, selfName = null)
        assertEquals(3, snap.messages.size)
    }

    @Test
    fun appsWithoutAuthorsGetOneUnknownSpeaker() {
        val plain = listOf(t("hey", 40, 100), t("what's up", 40, 160))
        val snap = ChatLayoutParser().parse(plain, inputTop = null, selfName = null)
        assertEquals(listOf("A", "A"), snap.messages.map { it.speaker })
        assertNull(snap.replyIndex)
    }

    @Test
    fun recognizesCommonTimestamps() {
        listOf("Today at 9:41 PM", "Yesterday at 11:19 PM", "12:03 AM", "9:41", "10/04/2026", "10/4/26, 9:41 PM", "Monday 9:41 PM")
            .forEach { assertEquals(it, true, ChatLayoutParser.isTimestamp(it)) }
        listOf("see you at 9", "today was fun", "sunshine", "monday plans?").forEach { assertEquals(it, false, ChatLayoutParser.isTimestamp(it)) }
    }
}
