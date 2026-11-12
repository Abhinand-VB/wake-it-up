package dev.wakeitup.service

import android.accessibilityservice.AccessibilityService
import android.content.SharedPreferences
import android.graphics.Rect
import android.os.SystemClock
import android.view.accessibility.AccessibilityEvent
import dev.wakeitup.BuildConfig
import dev.wakeitup.EngineHolder
import dev.wakeitup.WakeItUpApp
import dev.wakeitup.context.ContextCache
import dev.wakeitup.context.adapters.Adapters
import dev.wakeitup.context.adapters.NodeWalker
import dev.wakeitup.core.DecisionEngine
import dev.wakeitup.core.QuickFilter
import dev.wakeitup.core.RiskLevel
import dev.wakeitup.core.Verdict
import dev.wakeitup.overlay.OverlayController
import dev.wakeitup.settings.Settings
import dev.wakeitup.util.SafeLog
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Entry point for the on-device pipeline (SOURCE_OF_TRUTH.md §4, §5.3):
 * text change → quick filter → (cached context + draft) → classifier → decision → overlay.
 *
 * Threading (§5.11): accessibility callbacks and overlay on main; node-tree walks on a
 * single context thread; tokenize + inference on a single inference thread. Every entry
 * point catches everything: failures stay silent and never affect the host app (G6).
 */
class WakeItUpAccessibilityService : AccessibilityService() {

    private lateinit var settings: Settings
    private lateinit var engineHolder: EngineHolder
    private lateinit var overlay: OverlayController

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val contextExecutor: ExecutorService = Executors.newSingleThreadExecutor { Thread(it, "wiu-context") }
    private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor { Thread(it, "wiu-inference") }
    private val contextDispatcher = contextExecutor.asCoroutineDispatcher()
    private val inferenceDispatcher = inferenceExecutor.asCoroutineDispatcher()

    private val contextCache = ContextCache()
    private val quickFilter = QuickFilter()
    private var decision: DecisionEngine? = null

    // Main-thread state.
    private var checkJob: Job? = null
    private var refreshJob: Job? = null
    private var refreshDirty = false
    private var lastRefreshAt = 0L
    private var watchdogJob: Job? = null
    private val inputBounds = Rect()
    @Volatile private var inputTop: Int? = null
    private var activePackage: String? = null
    private var currentDraft = ""
    private var dismissedUntilCleared = false
    private var showing: Verdict? = null

    private val prefsListener = SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
        when (key) {
            Settings.KEY_PACKAGES -> applyPackageFilter()
            Settings.KEY_SENSITIVITY -> decision?.sensitivity = settings.sensitivity
        }
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        val app = WakeItUpApp.from(this)
        settings = app.settings
        engineHolder = app.engineHolder
        overlay = OverlayController(this) { verdict, helpful -> onWarningDismissed(verdict, helpful) }
        settings.register(prefsListener)
        applyPackageFilter()
        running = true
        // Load + warm up the model now so the first check is fast.
        scope.launch(inferenceDispatcher) {
            engineHolder.getOrLoad()
            engineHolder.wordlist()
        }
        SafeLog.info("service connected")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent) {
        try {
            val pkg = event.packageName?.toString() ?: return
            if (pkg == packageName || pkg !in settings.enabledPackages) return
            when (event.eventType) {
                AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED -> onTextChanged(event, pkg)
                AccessibilityEvent.TYPE_VIEW_FOCUSED -> if (isEditableClass(event.className)) scheduleContextRefresh(pkg, immediate = true)
                AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED ->
                    // Skip changes caused by typing in the input itself.
                    if (!isEditableClass(event.className)) scheduleContextRefresh(pkg)
                AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED -> {
                    hideWarning()
                    contextCache.clear()
                    scheduleContextRefresh(pkg, immediate = true)
                }
            }
        } catch (t: Throwable) {
            SafeLog.error("event", t)
        }
    }

    private fun onTextChanged(event: AccessibilityEvent, pkg: String) {
        if (event.isPassword) return
        val source = event.source
        val draft: String
        try {
            if (source != null) {
                if (source.isPassword) return
                source.getBoundsInScreen(inputBounds)
                inputTop = inputBounds.top
            }
            val showingHint = source?.isShowingHintText == true
            draft = if (showingHint) "" else event.text?.joinToString("")?.toString() ?: ""
        } finally {
            @Suppress("DEPRECATION")
            if (android.os.Build.VERSION.SDK_INT < android.os.Build.VERSION_CODES.TIRAMISU) source?.recycle()
        }

        if (pkg != activePackage) {
            activePackage = pkg
            contextCache.clear()
            scheduleContextRefresh(pkg, immediate = true)
        }
        currentDraft = draft

        if (draft.isBlank()) {
            onDraftCleared()
            return
        }
        if (dismissedUntilCleared) return

        checkJob?.cancel()
        if (!quickFilter.isWorthChecking(draft)) return
        val startedAt = SystemClock.uptimeMillis()
        checkJob = scope.launch { runCheck(draft, startedAt) }
    }

    private suspend fun runCheck(draft: String, startedAt: Long) {
        val snapshot = contextCache.snapshot
        val key = quickFilter.key(draft, contextCache.version)
        if (quickFilter.isDuplicate(key)) return
        val wasShowing = showing != null

        val verdict = withContext(inferenceDispatcher) {
            try {
                if (engineHolder.wordlist().matches(draft)) return@withContext Verdict.WORDLIST_HIT
                val engine = engineHolder.getOrLoad() ?: return@withContext null
                val engineDecision = decision ?: DecisionEngine(
                    engine.config.labels, engine.config.thresholds, settings.sensitivity,
                ).also { decision = it }
                val result = engine.check(snapshot.messages, snapshot.replyIndex, draft)
                lastInferenceMs = result.inferenceMs
                SafeLog.content {
                    "check: ctx=${snapshot.messages.size} reply=${snapshot.replyIndex} tokens=${result.tokenCount} " +
                        "infer=${"%.1f".format(result.inferenceMs)}ms probs=${result.probs.joinToString { "%.2f".format(it) }}"
                }
                engineDecision.evaluate(result.probs, wasShowing)
            } catch (t: Throwable) {
                SafeLog.error("check", t)
                null
            }
        } ?: return

        quickFilter.markChecked(key)
        if (verdict.level == RiskLevel.NONE) {
            hideWarning()
            return
        }
        // Display debounce: only show once typing has paused (§5.8). Hiding is never delayed.
        val wait = settings.displayDelayMs - (SystemClock.uptimeMillis() - startedAt)
        if (wait > 0) delay(wait)
        if (currentDraft != draft || dismissedUntilCleared) return
        showing = verdict
        overlay.show(verdict, inputBounds)
        startWatchdog()
    }

    private fun onDraftCleared() {
        checkJob?.cancel()
        hideWarning()
        dismissedUntilCleared = false
        quickFilter.reset()
    }

    private fun onWarningDismissed(verdict: Verdict, helpful: Boolean) {
        showing = null
        watchdogJob?.cancel()
        dismissedUntilCleared = true
        if (!helpful) settings.recordNotHelpful(verdict.category ?: Settings.GENERAL_CATEGORY)
    }

    private fun hideWarning() {
        showing = null
        watchdogJob?.cancel()
        if (::overlay.isInitialized) overlay.hide()
    }

    /**
     * Events from apps that aren't enabled are filtered out, so we can't see the user
     * switching away. While a warning is visible, poll the active window and hide the
     * warning when it no longer belongs to an enabled app.
     */
    private fun startWatchdog() {
        if (watchdogJob?.isActive == true) return
        watchdogJob = scope.launch {
            while (isActive && overlay.isShowing) {
                delay(WATCHDOG_INTERVAL_MS)
                val activePkg = try {
                    withContext(contextDispatcher) { rootInActiveWindow?.packageName?.toString() }
                } catch (t: Throwable) {
                    null
                }
                if (activePkg == null || activePkg !in settings.enabledPackages) {
                    hideWarning()
                    break
                }
            }
        }
    }

    /** Throttled, coalescing refresh of the conversation context (§5.4). */
    private fun scheduleContextRefresh(pkg: String, immediate: Boolean = false) {
        if (refreshJob?.isActive == true) {
            refreshDirty = true
            return
        }
        val sinceLast = SystemClock.uptimeMillis() - lastRefreshAt
        val wait = if (immediate) 0L else (REFRESH_THROTTLE_MS - sinceLast).coerceAtLeast(0L)
        refreshJob = scope.launch {
            if (wait > 0) delay(wait)
            refreshDirty = false
            withContext(contextDispatcher) { refreshContext(pkg) }
            lastRefreshAt = SystemClock.uptimeMillis()
            if (refreshDirty) {
                refreshDirty = false
                scheduleContextRefresh(pkg)
            }
        }
    }

    private fun refreshContext(pkg: String) {
        try {
            val root = rootInActiveWindow ?: return
            if (root.packageName?.toString() != pkg) return
            if (BuildConfig.DEBUG && dumpNextTree) {
                dumpNextTree = false
                SafeLog.content { "tree dump ($pkg):\n" + NodeWalker.dump(root) }
            }
            val selfName = settings.selfName.ifBlank { null }
            val snapshot = Adapters.forPackage(pkg).extract(root, inputTop, selfName, SystemClock.uptimeMillis())
            if (snapshot != null) contextCache.update(snapshot)
            SafeLog.content { "context: ${contextCache.snapshot}" }
        } catch (t: Throwable) {
            SafeLog.error("context refresh", t)
        }
    }

    private fun applyPackageFilter() {
        try {
            val info = serviceInfo ?: return
            val enabled = settings.enabledPackages
            // An empty/null list would mean "all apps" — never allow that.
            info.packageNames = if (enabled.isEmpty()) arrayOf(packageName) else enabled.toTypedArray()
            serviceInfo = info
            hideWarning()
        } catch (t: Throwable) {
            SafeLog.error("package filter", t)
        }
    }

    override fun onInterrupt() {
        hideWarning()
    }

    override fun onDestroy() {
        running = false
        try {
            settings.unregister(prefsListener)
            hideWarning()
        } catch (t: Throwable) {
            SafeLog.error("destroy", t)
        }
        scope.cancel()
        contextExecutor.shutdown()
        inferenceExecutor.shutdown()
        super.onDestroy()
    }

    private fun isEditableClass(className: CharSequence?): Boolean {
        val name = className?.toString() ?: return false
        return name.endsWith("EditText") || name.endsWith("TextInput") || name.contains("ReactEditText")
    }

    companion object {
        private const val REFRESH_THROTTLE_MS = 250L
        private const val WATCHDOG_INTERVAL_MS = 750L

        /** True while the service is connected; shown on the settings screen. */
        @Volatile
        var running: Boolean = false
            private set

        @Volatile
        var lastInferenceMs: Double? = null
            private set

        /** Debug builds: log the next context refresh's node tree (for tuning adapters). */
        @Volatile
        var dumpNextTree: Boolean = false
    }
}
