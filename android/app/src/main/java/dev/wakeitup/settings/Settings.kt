package dev.wakeitup.settings

import android.content.Context
import android.content.SharedPreferences
import dev.wakeitup.core.Sensitivity

data class KnownApp(val packageName: String, val label: String, val hasDedicatedAdapter: Boolean)

/** User settings, stored in private SharedPreferences. Never stores message text. */
class Settings(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("wakeitup", Context.MODE_PRIVATE)

    var enabledPackages: Set<String>
        get() = prefs.getStringSet(KEY_PACKAGES, null)?.toSet() ?: DEFAULT_PACKAGES
        set(value) = prefs.edit().putStringSet(KEY_PACKAGES, value).apply()

    var sensitivity: Sensitivity
        get() = runCatching { Sensitivity.valueOf(prefs.getString(KEY_SENSITIVITY, null) ?: "") }
            .getOrDefault(Sensitivity.NORMAL)
        set(value) = prefs.edit().putString(KEY_SENSITIVITY, value.name).apply()

    /** The user's display name in chat apps, used to label their own messages as "ME". */
    var selfName: String
        get() = prefs.getString(KEY_SELF_NAME, "") ?: ""
        set(value) = prefs.edit().putString(KEY_SELF_NAME, value.trim()).apply()

    var displayDelayMs: Long
        get() = prefs.getLong(KEY_DELAY, DEFAULT_DELAY_MS)
        set(value) = prefs.edit().putLong(KEY_DELAY, value).apply()

    fun notHelpfulCount(category: String): Int = prefs.getInt(KEY_FEEDBACK_PREFIX + category, 0)

    fun recordNotHelpful(category: String) {
        prefs.edit().putInt(KEY_FEEDBACK_PREFIX + category, notHelpfulCount(category) + 1).apply()
    }

    fun register(listener: SharedPreferences.OnSharedPreferenceChangeListener) =
        prefs.registerOnSharedPreferenceChangeListener(listener)

    fun unregister(listener: SharedPreferences.OnSharedPreferenceChangeListener) =
        prefs.unregisterOnSharedPreferenceChangeListener(listener)

    companion object {
        const val KEY_PACKAGES = "enabled_packages"
        const val KEY_SENSITIVITY = "sensitivity"
        const val KEY_SELF_NAME = "self_name"
        const val KEY_DELAY = "display_delay_ms"
        private const val KEY_FEEDBACK_PREFIX = "not_helpful_"
        const val DEFAULT_DELAY_MS = 300L
        /** Category key used for feedback on general (uncategorized) warnings. */
        const val GENERAL_CATEGORY = "general"

        val DEFAULT_PACKAGES = setOf("com.discord")

        val KNOWN_APPS = listOf(
            KnownApp("com.discord", "Discord", hasDedicatedAdapter = true),
            KnownApp("com.whatsapp", "WhatsApp", hasDedicatedAdapter = false),
            KnownApp("com.samsung.android.messaging", "Samsung Messages", hasDedicatedAdapter = false),
            KnownApp("com.google.android.apps.messaging", "Google Messages", hasDedicatedAdapter = false),
            KnownApp("com.instagram.android", "Instagram", hasDedicatedAdapter = false),
            KnownApp("org.telegram.messenger", "Telegram", hasDedicatedAdapter = false),
            KnownApp("com.snapchat.android", "Snapchat", hasDedicatedAdapter = false),
        )
    }
}
