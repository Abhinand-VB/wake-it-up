package dev.wakeitup.ui

import android.accessibilityservice.AccessibilityServiceInfo
import android.app.Activity
import android.app.AlertDialog
import android.content.ActivityNotFoundException
import android.content.ComponentName
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings as AndroidSettings
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.view.accessibility.AccessibilityManager
import android.widget.CheckBox
import android.widget.EditText
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import dev.wakeitup.BuildConfig
import dev.wakeitup.EngineHolder
import dev.wakeitup.WakeItUpApp
import dev.wakeitup.core.Sensitivity
import dev.wakeitup.service.WakeItUpAccessibilityService
import dev.wakeitup.settings.Settings

/** Settings → Accessibility → this service's own page (public since API 33; string works on older too). */
private const val ACTION_ACCESSIBILITY_DETAILS_SETTINGS = "android.settings.ACCESSIBILITY_DETAILS_SETTINGS"

/** Setup flow and settings (SOURCE_OF_TRUTH.md §5.10). */
class SettingsActivity : Activity() {
    private lateinit var settings: Settings
    private lateinit var status: TextView
    private lateinit var feedback: TextView
    /** Show the setup prompt at most once per screen instance, so "Later" isn't nagged. */
    private var setupPromptShown = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        settings = WakeItUpApp.from(this).settings

        setContentView(page {
            heading("Wake It Up")
            body(
                "Warns you privately when a message you're typing might come across badly, " +
                    "before you send it.\n\n" +
                    "What it reads: the message you're typing and the recent messages on screen, " +
                    "only in the apps you turn on below. Password fields are never read.\n\n" +
                    "Where it goes: nowhere. Everything is checked on this phone. The app has no " +
                    "internet permission and stores no messages.",
            )

            section("Status")
            status = body("")

            section("1. Turn on the service")
            body(
                "Android doesn't allow a permission popup for this kind of access; you have to " +
                    "switch it on yourself. If the switch is greyed out or says \"Restricted setting\", " +
                    "open App info → ⋮ (top right) → Allow restricted settings, then try again.",
            )
            button("Turn on Wake It Up") { openServiceToggle() }
            button("Open app info") { openAppInfo() }

            section("2. Keep it running (Samsung)")
            body("Settings → Battery → Background usage limits → Never sleeping apps → add Wake It Up.")
            button("Open battery settings") {
                startActivity(Intent(AndroidSettings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
            }

            section("Apps to watch")
            val enabled = settings.enabledPackages.toMutableSet()
            for (app in Settings.KNOWN_APPS) {
                addView(CheckBox(context).apply {
                    text = if (app.hasDedicatedAdapter) app.label else "${app.label} (experimental)"
                    isChecked = app.packageName in enabled
                    setOnCheckedChangeListener { _, checked ->
                        if (checked) enabled += app.packageName else enabled -= app.packageName
                        settings.enabledPackages = enabled.toSet()
                    }
                })
            }

            section("Sensitivity")
            addView(RadioGroup(context).apply {
                val current = settings.sensitivity
                for (s in Sensitivity.entries) {
                    addView(RadioButton(context).apply {
                        id = s.ordinal + 1
                        text = when (s) {
                            Sensitivity.RELAXED -> "Relaxed: fewer warnings"
                            Sensitivity.NORMAL -> "Normal"
                            Sensitivity.STRICT -> "Strict: more warnings"
                        }
                        isChecked = s == current
                    })
                }
                setOnCheckedChangeListener { _, checkedId ->
                    Sensitivity.entries.getOrNull(checkedId - 1)?.let { settings.sensitivity = it }
                }
            })

            section("Your display name (optional)")
            body("Helps tell your own messages apart from others'. Stays on this phone.")
            addView(EditText(context).apply {
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
                setText(settings.selfName)
                hint = "e.g. your Discord display name"
                addTextChangedListener(object : TextWatcher {
                    override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
                    override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
                    override fun afterTextChanged(s: Editable?) {
                        settings.selfName = s?.toString() ?: ""
                    }
                })
            })

            section("Try it")
            button("Open test playground") { startActivity(Intent(context, TestActivity::class.java)) }

            section("\"Not helpful\" counts")
            feedback = body("")

            if (BuildConfig.DEBUG) {
                section("Debug")
                button("Log screen structure on next refresh") {
                    WakeItUpAccessibilityService.dumpNextTree = true
                    status.text = status.text.toString() + "\nTree dump armed: switch to the chat app (see logcat, tag WakeItUp)."
                }
                button("Retry model load") {
                    WakeItUpApp.from(context).engineHolder.resetFailure()
                    Thread { WakeItUpApp.from(context).engineHolder.getOrLoad(); runOnUiThread { refresh() } }.start()
                }
            }
        })
    }

    override fun onResume() {
        super.onResume()
        refresh()
        if (!isServiceEnabled() && !setupPromptShown) {
            setupPromptShown = true
            AlertDialog.Builder(this)
                .setTitle("Turn on Wake It Up")
                .setMessage(
                    "Wake It Up needs accessibility access to see what you type in the apps you choose. " +
                        "Android doesn't allow a permission popup for this, so the next screen opens " +
                        "Wake It Up's switch: turn it on there.\n\n" +
                        "If the switch is greyed out, come back and tap \"Open app info\" → ⋮ → " +
                        "Allow restricted settings first.",
                )
                .setPositiveButton("Open switch") { _, _ -> openServiceToggle() }
                .setNeutralButton("Open app info") { _, _ -> openAppInfo() }
                .setNegativeButton("Later", null)
                .show()
        }
        // Load the model in the background so the status line can show it.
        Thread {
            WakeItUpApp.from(this).engineHolder.getOrLoad()
            runOnUiThread { refresh() }
        }.start()
    }

    private fun refresh() {
        val serviceOn = isServiceEnabled()
        val model = when (val s = WakeItUpApp.from(this).engineHolder.status) {
            EngineHolder.Status.NotLoaded -> "not loaded"
            EngineHolder.Status.Loading -> "loading…"
            is EngineHolder.Status.Ready -> "${s.modelVersion} (load ${s.loadMs.toInt()} ms, warm-up ${s.warmUpMs.toInt()} ms)"
            is EngineHolder.Status.Failed -> "ERROR: ${s.reason}"
        }
        val last = WakeItUpAccessibilityService.lastInferenceMs?.let { "%.1f ms".format(it) } ?: "—"
        status.text = buildString {
            append(if (serviceOn) "✅ Service on" else "❌ Service off: turn it on in step 1")
            if (serviceOn && !WakeItUpAccessibilityService.running) append(" (not connected yet)")
            append("\nModel: ").append(model)
            append("\nLast check: ").append(last)
            append("\nApp version: ").append(BuildConfig.VERSION_NAME)
        }
        val categories = listOf("general", "hate_slur", "insensitive", "harassment", "threat", "sexual", "misread_risk")
        feedback.text = categories.joinToString("\n") { "$it: ${settings.notHelpfulCount(it)}" }
    }

    /** Opens WakeItUp's own accessibility switch; falls back to the accessibility list. */
    private fun openServiceToggle() {
        val component = ComponentName(this, WakeItUpAccessibilityService::class.java).flattenToString()
        val details = Intent(ACTION_ACCESSIBILITY_DETAILS_SETTINGS).putExtra(Intent.EXTRA_COMPONENT_NAME, component)
        try {
            startActivity(details)
        } catch (e: ActivityNotFoundException) {
            startActivity(Intent(AndroidSettings.ACTION_ACCESSIBILITY_SETTINGS))
        } catch (e: SecurityException) {
            startActivity(Intent(AndroidSettings.ACTION_ACCESSIBILITY_SETTINGS))
        }
    }

    private fun openAppInfo() {
        startActivity(Intent(AndroidSettings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$packageName")))
    }

    private fun isServiceEnabled(): Boolean {
        val am = getSystemService(AccessibilityManager::class.java) ?: return false
        return am.getEnabledAccessibilityServiceList(AccessibilityServiceInfo.FEEDBACK_ALL_MASK)
            .any { it.resolveInfo?.serviceInfo?.packageName == packageName }
    }
}
