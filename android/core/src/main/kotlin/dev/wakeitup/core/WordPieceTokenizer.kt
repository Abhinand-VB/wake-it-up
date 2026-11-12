package dev.wakeitup.core

import java.text.Normalizer
import java.util.Locale

/**
 * BERT uncased WordPiece tokenizer. Must reproduce Hugging Face's slow `BertTokenizer`
 * (the reference used in training) token-for-token — see SOURCE_OF_TRUTH.md §6.2.1.
 * Parity is verified against fixtures exported by the training pipeline.
 */
class WordPieceTokenizer(
    vocabLines: List<String>,
    private val doLowerCase: Boolean = true,
    private val maxInputCharsPerWord: Int = 100,
) {
    private val vocab: Map<String, Int> = HashMap<String, Int>(vocabLines.size * 2).also { map ->
        // HF load_vocab: index = line number, later duplicates overwrite earlier ones.
        vocabLines.forEachIndexed { index, line -> map[line.trimEnd('\n', '\r')] = index }
    }

    val unkId: Int = vocab[UNK] ?: error("vocab has no $UNK")
    val clsId: Int = vocab[CLS] ?: error("vocab has no $CLS")
    val sepId: Int = vocab[SEP] ?: error("vocab has no $SEP")

    fun id(token: String): Int = vocab[token] ?: unkId

    /**
     * Special-token strings typed by the user ("[SEP]", "[CLS]", ...) are treated as plain
     * text, exactly like the legacy slow reference tokenizer. This also means user text can
     * never inject control tokens into the model input.
     */
    fun tokenize(text: String): List<String> {
        val out = ArrayList<String>()
        for (word in basicTokenize(text)) out += wordPiece(word)
        return out
    }

    fun tokenizeToIds(text: String): IntArray {
        val tokens = tokenize(text)
        return IntArray(tokens.size) { id(tokens[it]) }
    }

    private fun basicTokenize(input: String): List<String> {
        // Same order as the reference: lowercase, clean, pad CJK, NFC, whitespace split,
        // lowercase + strip accents per token, split punctuation.
        val lowered = if (doLowerCase) input.lowercase(Locale.ROOT) else input
        val cleaned = cleanText(lowered)
        val spaced = padCjk(cleaned)
        val normalized = Normalizer.normalize(spaced, Normalizer.Form.NFC)
        val out = ArrayList<String>()
        for (token in pythonWhitespaceSplit(normalized)) {
            var t = token
            if (doLowerCase) {
                t = stripAccents(t.lowercase(Locale.ROOT))
            }
            splitOnPunctuation(t, out)
        }
        return out
    }

    private fun wordPiece(word: String): List<String> {
        val cps = word.codePoints().toArray()
        if (cps.size > maxInputCharsPerWord) return listOf(UNK)
        val pieces = ArrayList<String>()
        var start = 0
        while (start < cps.size) {
            var end = cps.size
            var current: String? = null
            while (start < end) {
                var candidate = String(cps, start, end - start)
                if (start > 0) candidate = "##$candidate"
                if (vocab.containsKey(candidate)) {
                    current = candidate
                    break
                }
                end--
            }
            if (current == null) return listOf(UNK)
            pieces += current
            start = end
        }
        return pieces
    }

    companion object {
        const val UNK = "[UNK]"
        const val SEP = "[SEP]"
        const val CLS = "[CLS]"

        private fun cleanText(text: String): String {
            val sb = StringBuilder(text.length)
            var i = 0
            while (i < text.length) {
                val cp = text.codePointAt(i)
                i += Character.charCount(cp)
                if (cp == 0 || cp == 0xFFFD || isControl(cp)) continue
                if (isWhitespace(cp)) sb.append(' ') else sb.appendCodePoint(cp)
            }
            return sb.toString()
        }

        /** HF `_is_whitespace`: space, \t, \n, \r, or category Zs. */
        private fun isWhitespace(cp: Int): Boolean =
            cp == ' '.code || cp == '\t'.code || cp == '\n'.code || cp == '\r'.code ||
                Character.getType(cp) == Character.SPACE_SEPARATOR.toInt()

        /**
         * HF `_is_control`: category C* except \t \n \r.
         * Deliberate deviation: UNASSIGNED (Cn) is NOT treated as control, because the JVM's
         * Unicode tables can be older than Python's, and newer emoji would otherwise be dropped
         * here but kept (as [UNK]) by the reference tokenizer.
         */
        private fun isControl(cp: Int): Boolean {
            if (cp == '\t'.code || cp == '\n'.code || cp == '\r'.code) return false
            return when (Character.getType(cp).toByte()) {
                Character.CONTROL, Character.FORMAT, Character.PRIVATE_USE, Character.SURROGATE -> true
                else -> false
            }
        }

        private fun isPunctuation(cp: Int): Boolean {
            if (cp in 33..47 || cp in 58..64 || cp in 91..96 || cp in 123..126) return true
            return when (Character.getType(cp).toByte()) {
                Character.CONNECTOR_PUNCTUATION, Character.DASH_PUNCTUATION,
                Character.START_PUNCTUATION, Character.END_PUNCTUATION,
                Character.INITIAL_QUOTE_PUNCTUATION, Character.FINAL_QUOTE_PUNCTUATION,
                Character.OTHER_PUNCTUATION -> true
                else -> false
            }
        }

        private fun isCjk(cp: Int): Boolean =
            cp in 0x4E00..0x9FFF || cp in 0x3400..0x4DBF || cp in 0x20000..0x2A6DF ||
                cp in 0x2A700..0x2B73F || cp in 0x2B740..0x2B81F || cp in 0x2B820..0x2CEAF ||
                cp in 0xF900..0xFAFF || cp in 0x2F800..0x2FA1F

        private fun padCjk(text: String): String {
            val sb = StringBuilder(text.length)
            var i = 0
            while (i < text.length) {
                val cp = text.codePointAt(i)
                i += Character.charCount(cp)
                if (isCjk(cp)) sb.append(' ').appendCodePoint(cp).append(' ') else sb.appendCodePoint(cp)
            }
            return sb.toString()
        }

        private fun stripAccents(text: String): String {
            val nfd = Normalizer.normalize(text, Normalizer.Form.NFD)
            val sb = StringBuilder(nfd.length)
            var i = 0
            while (i < nfd.length) {
                val cp = nfd.codePointAt(i)
                i += Character.charCount(cp)
                if (Character.getType(cp) != Character.NON_SPACING_MARK.toInt()) sb.appendCodePoint(cp)
            }
            return sb.toString()
        }

        private fun splitOnPunctuation(text: String, out: MutableList<String>) {
            val current = StringBuilder()
            var i = 0
            while (i < text.length) {
                val cp = text.codePointAt(i)
                i += Character.charCount(cp)
                if (isPunctuation(cp)) {
                    if (current.isNotEmpty()) {
                        out += current.toString()
                        current.setLength(0)
                    }
                    out += String(Character.toChars(cp))
                } else {
                    current.appendCodePoint(cp)
                }
            }
            if (current.isNotEmpty()) out += current.toString()
        }

        /** Python `str.split()` with no arguments: splits on any `str.isspace()` character. */
        private fun pythonWhitespaceSplit(text: String): List<String> {
            val out = ArrayList<String>()
            val current = StringBuilder()
            var i = 0
            while (i < text.length) {
                val cp = text.codePointAt(i)
                i += Character.charCount(cp)
                if (isPythonSpace(cp)) {
                    if (current.isNotEmpty()) {
                        out += current.toString()
                        current.setLength(0)
                    }
                } else {
                    current.appendCodePoint(cp)
                }
            }
            if (current.isNotEmpty()) out += current.toString()
            return out
        }

        private fun isPythonSpace(cp: Int): Boolean = when (cp) {
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0,
            0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 -> true
            in 0x2000..0x200A -> true
            else -> false
        }
    }
}
