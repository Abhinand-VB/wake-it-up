package dev.wakeitup.util

import android.util.Log
import dev.wakeitup.BuildConfig

/**
 * Logging that never writes message content in release builds (SOURCE_OF_TRUTH.md §10).
 * Use [info]/[error] for timings, codes and counts; [content] only for debug-only dumps.
 */
object SafeLog {
    private const val TAG = "WakeItUp"

    fun info(message: String) {
        Log.i(TAG, message)
    }

    fun error(where: String, t: Throwable) {
        // Exception messages can contain text from the screen, so only the type is logged.
        Log.e(TAG, "$where: ${t.javaClass.simpleName}")
        if (BuildConfig.DEBUG) Log.e(TAG, where, t)
    }

    /** Debug builds only. The lambda is never evaluated in release. */
    inline fun content(message: () -> String) {
        if (BuildConfig.DEBUG) Log.d("WakeItUp", message())
    }
}
