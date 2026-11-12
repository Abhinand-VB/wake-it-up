package dev.wakeitup.core

/**
 * Cheap pre-checks before running the model (SOURCE_OF_TRUTH.md §5.5).
 * A key is only marked as checked once a check actually completes, so a cancelled
 * check for the same draft will run again.
 */
class QuickFilter(private val minChars: Int = 2) {
    private var lastCheckedKey: String? = null

    fun isWorthChecking(draft: String): Boolean = draft.trim().length >= minChars

    fun key(draft: String, contextVersion: Long): String = "$contextVersion\u0000${draft.trim()}"

    fun isDuplicate(key: String): Boolean = key == lastCheckedKey

    fun markChecked(key: String) {
        lastCheckedKey = key
    }

    fun reset() {
        lastCheckedKey = null
    }
}
