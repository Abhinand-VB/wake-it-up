package dev.wakeitup.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Uses a neutral placeholder term; the real list ships separately (SOURCE_OF_TRUTH.md §5.5). */
class WordlistAndFilterTest {
    private val list = Wordlist.parse(listOf("# comment", "", "badword", "two words are skipped"))

    @Test
    fun parsesSingleWordEntriesOnly() {
        assertEquals(1, list.size)
    }

    @Test
    fun matchesObfuscatedWholeWords() {
        assertTrue(list.matches("you BADWORD"))
        assertTrue(list.matches("b4dw0rd!"))
        assertTrue(list.matches("baaaadwooord"))
        assertTrue(list.matches("bádword"))
    }

    @Test
    fun doesNotMatchSubstrings() {
        assertFalse(list.matches("badwordsmith is a name"))
        assertFalse(list.matches("nothing here"))
    }

    @Test
    fun quickFilterSkipsShortAndDuplicateDrafts() {
        val filter = QuickFilter()
        assertFalse(filter.isWorthChecking(" k "))
        assertTrue(filter.isWorthChecking("ok"))
        val key = filter.key("hello there", 1)
        assertFalse(filter.isDuplicate(key))
        filter.markChecked(key)
        assertTrue(filter.isDuplicate(filter.key("hello there ", 1)))
        assertFalse(filter.isDuplicate(filter.key("hello there", 2)))
        filter.reset()
        assertFalse(filter.isDuplicate(key))
    }
}
