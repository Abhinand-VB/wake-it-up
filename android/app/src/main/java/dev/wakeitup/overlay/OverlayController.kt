package dev.wakeitup.overlay

import android.accessibilityservice.AccessibilityService
import android.annotation.SuppressLint
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.TextView
import dev.wakeitup.core.Explanations
import dev.wakeitup.core.RiskLevel
import dev.wakeitup.core.Verdict
import dev.wakeitup.util.SafeLog

/**
 * The floating warning (SOURCE_OF_TRUTH.md §5.9). A non-focusable accessibility overlay:
 * it never takes focus or input from the host app. Collapsed = pill, tapped = detail card.
 * All methods must be called on the main thread.
 */
class OverlayController(
    private val service: AccessibilityService,
    /** "Got it" (helpful = true) or "Not helpful" (false); either way the warning is dismissed. */
    private val onDismiss: (verdict: Verdict, helpful: Boolean) -> Unit,
) {
    private val windowManager = service.getSystemService(WindowManager::class.java)
    private var view: View? = null
    private var expanded = false
    private var verdict: Verdict? = null
    private val anchor = Rect()

    val isShowing: Boolean get() = view != null

    fun show(newVerdict: Verdict, inputBounds: Rect) {
        verdict = newVerdict
        anchor.set(inputBounds)
        if (view != null && expanded) {
            // Keep the card open, but refresh its content.
            replaceView(buildCard(newVerdict))
            return
        }
        replaceView(buildPill(newVerdict))
    }

    fun hide() {
        removeView()
        expanded = false
        verdict = null
    }

    private fun expand() {
        val v = verdict ?: return
        expanded = true
        replaceView(buildCard(v))
    }

    private fun collapse() {
        val v = verdict ?: return
        expanded = false
        replaceView(buildPill(v))
    }

    private fun replaceView(newView: View) {
        removeView()
        try {
            windowManager.addView(newView, layoutParams(fullWidth = expanded))
            view = newView
        } catch (t: Throwable) {
            SafeLog.error("overlay add", t)
            view = null
        }
    }

    private fun removeView() {
        val current = view ?: return
        view = null
        try {
            windowManager.removeViewImmediate(current)
        } catch (t: Throwable) {
            SafeLog.error("overlay remove", t)
        }
    }

    private fun layoutParams(fullWidth: Boolean): WindowManager.LayoutParams {
        val screen = screenSize()
        val margin = dp(12)
        val params = WindowManager.LayoutParams(
            if (fullWidth) screen.width() - 2 * margin else WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS or
                WindowManager.LayoutParams.FLAG_WATCH_OUTSIDE_TOUCH,
            PixelFormat.TRANSLUCENT,
        )
        if (anchor.isEmpty) {
            // Input position unknown: top-centre fallback.
            params.gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
            params.y = dp(48)
        } else {
            // Bottom-anchored just above the input field, so the view grows upward.
            params.gravity = Gravity.BOTTOM or Gravity.START
            params.x = if (fullWidth) margin else (anchor.left + dp(8)).coerceIn(margin, screen.width() - dp(200))
            params.y = (screen.height() - anchor.top + dp(8)).coerceAtLeast(dp(8))
        }
        return params
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun buildPill(v: Verdict): View {
        val high = v.level == RiskLevel.HIGH
        return TextView(service).apply {
            text = if (high) "⚠️ Likely to offend" else "⚠️ Might land badly"
            setTextColor(if (high) Color.WHITE else Color.parseColor("#1F1300"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
            typeface = Typeface.DEFAULT_BOLD
            setPadding(dp(14), dp(8), dp(14), dp(8))
            background = rounded(if (high) "#D93036" else "#FFB020", dp(20).toFloat())
            elevation = dp(6).toFloat()
            contentDescription = "Wake It Up warning. Double tap for details."
            setOnClickListener { expand() }
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun buildCard(v: Verdict): View {
        val card = LinearLayout(service).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(14), dp(16), dp(10))
            background = rounded("#F21F2328", dp(16).toFloat())
            elevation = dp(8).toFloat()
            setOnTouchListener { _, event ->
                if (event.action == MotionEvent.ACTION_OUTSIDE) collapse()
                false
            }
        }
        card.addView(label("⚠️ " + Explanations.title(v.category), 16f, Color.WHITE, bold = true))
        card.addView(label(Explanations.body(v.category), 14f, Color.parseColor("#E6E8EB")).apply {
            setPadding(0, dp(6), 0, dp(4))
        })
        card.addView(label("Risk ${(v.risk * 100).toInt()}% · checked on your phone", 12f, Color.parseColor("#9BA1A6")))

        val buttons = LinearLayout(service).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
            setPadding(0, dp(6), 0, 0)
        }
        buttons.addView(button("Not helpful") { hide(); onDismiss(v, false) })
        buttons.addView(button("Got it") { hide(); onDismiss(v, true) })
        card.addView(buttons)
        return card
    }

    private fun label(text: String, sp: Float, color: Int, bold: Boolean = false) = TextView(service).apply {
        this.text = text
        setTextColor(color)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sp)
        if (bold) typeface = Typeface.DEFAULT_BOLD
    }

    private fun button(text: String, onClick: () -> Unit) = TextView(service).apply {
        this.text = text
        setTextColor(Color.parseColor("#8AB4FF"))
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
        typeface = Typeface.DEFAULT_BOLD
        setPadding(dp(12), dp(10), dp(12), dp(10))
        isClickable = true
        setOnClickListener { onClick() }
    }

    private fun rounded(color: String, radius: Float) = GradientDrawable().apply {
        setColor(Color.parseColor(color))
        cornerRadius = radius
    }

    private fun screenSize(): Rect =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            windowManager.maximumWindowMetrics.bounds
        } else {
            @Suppress("DEPRECATION")
            val metrics = android.util.DisplayMetrics().also { windowManager.defaultDisplay.getRealMetrics(it) }
            Rect(0, 0, metrics.widthPixels, metrics.heightPixels)
        }

    private fun dp(value: Int): Int = (value * service.resources.displayMetrics.density).toInt()
}
