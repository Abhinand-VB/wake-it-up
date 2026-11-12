package dev.wakeitup.ui

import android.content.Context
import android.graphics.Typeface
import android.util.TypedValue
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/** Tiny helpers for the programmatic (XML-free) settings/test screens. */
internal fun Context.dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

internal fun Context.page(build: LinearLayout.() -> Unit): ScrollView {
    val column = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(16), dp(24), dp(16), dp(32))
        build()
    }
    return ScrollView(this).apply {
        fitsSystemWindows = true
        addView(column)
    }
}

internal fun LinearLayout.heading(text: String) = addView(TextView(context).apply {
    this.text = text
    setTextSize(TypedValue.COMPLEX_UNIT_SP, 22f)
    typeface = Typeface.DEFAULT_BOLD
})

internal fun LinearLayout.section(text: String) = addView(TextView(context).apply {
    this.text = text
    setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
    typeface = Typeface.DEFAULT_BOLD
    setPadding(0, context.dp(20), 0, context.dp(4))
})

internal fun LinearLayout.body(text: String): TextView = TextView(context).apply {
    this.text = text
    setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
    setPadding(0, context.dp(4), 0, context.dp(4))
}.also { addView(it) }

internal fun LinearLayout.button(text: String, onClick: (View) -> Unit): Button = Button(context).apply {
    this.text = text
    isAllCaps = false
    setOnClickListener(onClick)
}.also { addView(it) }
