package dev.wakeitup.core

import java.text.Normalizer
import java.util.Locale

/**
 * Instant hard-flag terms (SOURCE_OF_TRUTH.md §5.5). Matches whole words only, after
 * normalization (lowercase, accents stripped, common leetspeak mapped, repeated letters
 * collapsed), so "s0ooo" style obfuscation is caught without substring false positives.
 * v1 supports single-word entries only.
 */
class Wordlist(entries: Collection<String>) {
    private val terms: Set<String> = entries.map { normalizeWord(it) }.filter { it.length >= 2 }.toSet()

    val size: Int get() = terms.size

    fun matches(text: String): Boolean {
        if (terms.isEmpty()) return false
        return WORD_SPLIT.split(text).any { it.isNotEmpty() && normalizeWord(it) in terms }
    }

    companion object {
        val EMPTY = Wordlist(emptyList())

        private val WORD_SPLIT = Regex("""[\s.,!?;:"'()\[\]{}<>/\\|~`^*_+=\-]+""")
        private val COMBINING = Regex("""\p{Mn}+""")
        private val LEET = mapOf('0' to 'o', '1' to 'i', '3' to 'e', '4' to 'a', '5' to 's', '7' to 't', '@' to 'a', '$' to 's')

        /** Parses a wordlist file: one term per line, `#` comments and blank lines ignored. */
        fun parse(lines: List<String>): Wordlist =
            Wordlist(lines.map { it.substringBefore('#').trim() }.filter { it.isNotEmpty() && ' ' !in it })

        fun normalizeWord(word: String): String {
            val stripped = COMBINING.replace(Normalizer.normalize(word.lowercase(Locale.ROOT), Normalizer.Form.NFD), "")
            val sb = StringBuilder(stripped.length)
            for (c in stripped) {
                val mapped = LEET[c] ?: c
                if (!mapped.isLetter()) continue
                if (sb.isNotEmpty() && sb[sb.length - 1] == mapped) continue
                sb.append(mapped)
            }
            return sb.toString()
        }
    }
}
