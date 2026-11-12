package dev.wakeitup.context.adapters

import android.graphics.Rect
import android.os.Build
import android.view.accessibility.AccessibilityNodeInfo
import dev.wakeitup.core.ChatLayoutParser
import dev.wakeitup.core.ConversationSnapshot
import dev.wakeitup.core.ScreenText

/**
 * Per-app conversation extraction (SOURCE_OF_TRUTH.md §5.4). A UI change in one app
 * should only ever require changing that app's adapter. Must return null rather than
 * throw when the screen is unrecognizable; the pipeline then runs draft-only.
 */
interface AppAdapter {
    fun extract(root: AccessibilityNodeInfo, inputTop: Int?, selfName: String?, nowMs: Long): ConversationSnapshot?
}

object Adapters {
    fun forPackage(packageName: String): AppAdapter = when (packageName) {
        "com.discord" -> DiscordAdapter
        else -> GenericAdapter
    }
}

/** Fallback for apps without a dedicated adapter: visible text above the input, by position. */
object GenericAdapter : AppAdapter {
    private val parser = ChatLayoutParser()

    override fun extract(root: AccessibilityNodeInfo, inputTop: Int?, selfName: String?, nowMs: Long): ConversationSnapshot? =
        parser.parse(NodeWalker.collect(root), inputTop, selfName, nowMs)
}

/**
 * Discord. Heuristics are untested against a real device yet: use the debug tree dump
 * (Settings → Debug) on the S25 and refine [chrome] / the reply banner as needed.
 */
object DiscordAdapter : AppAdapter {
    private val parser = ChatLayoutParser(
        replyBanner = Regex("""^replying to\s+@?(.+)$""", RegexOption.IGNORE_CASE),
        chrome = setOf("jump to present", "mark as read", "new", "bot", "app", "server tag"),
    )

    override fun extract(root: AccessibilityNodeInfo, inputTop: Int?, selfName: String?, nowMs: Long): ConversationSnapshot? =
        parser.parse(NodeWalker.collect(root), inputTop, selfName, nowMs)
}

/** Flattens the visible part of a node tree into [ScreenText] items. */
object NodeWalker {
    private const val MAX_NODES = 800

    fun collect(root: AccessibilityNodeInfo): List<ScreenText> {
        val out = ArrayList<ScreenText>()
        val bounds = Rect()
        var visited = 0
        val stack = ArrayDeque<AccessibilityNodeInfo>()
        stack.addLast(root)
        while (stack.isNotEmpty() && visited < MAX_NODES) {
            val node = stack.removeLast()
            visited++
            if (node.isVisibleToUser && !node.isPassword) {
                val text = node.text?.toString()?.takeIf { it.isNotBlank() }
                    // A container's content description often repeats its children; only use it on leaves.
                    ?: node.contentDescription?.toString()?.takeIf { it.isNotBlank() && node.childCount == 0 }
                if (text != null) {
                    node.getBoundsInScreen(bounds)
                    out += ScreenText(text, bounds.left, bounds.top, bounds.right, bounds.bottom, node.isEditable)
                }
                // Push children in reverse so they pop in document order.
                for (i in node.childCount - 1 downTo 0) node.getChild(i)?.let { stack.addLast(it) }
            }
            if (node !== root) recycle(node)
        }
        stack.forEach { if (it !== root) recycle(it) }
        return out
    }

    /** Debug-only structural dump for tuning adapters on a real device. */
    fun dump(root: AccessibilityNodeInfo, maxNodes: Int = 400): String {
        val sb = StringBuilder()
        val bounds = Rect()
        var count = 0
        fun walk(node: AccessibilityNodeInfo, depth: Int) {
            if (count++ >= maxNodes) return
            node.getBoundsInScreen(bounds)
            sb.append("  ".repeat(depth))
                .append(node.className?.toString()?.substringAfterLast('.') ?: "?")
                .append(" id=").append(node.viewIdResourceName?.substringAfterLast('/') ?: "-")
                .append(" ").append(bounds.toShortString())
                .append(if (node.isEditable) " EDITABLE" else "")
                .append(" text=").append(node.text?.toString()?.take(60)?.replace('\n', ' ') ?: "-")
                .append(" desc=").append(node.contentDescription?.toString()?.take(60)?.replace('\n', ' ') ?: "-")
                .append('\n')
            for (i in 0 until node.childCount) {
                val child = node.getChild(i) ?: continue
                walk(child, depth + 1)
                recycle(child)
            }
        }
        walk(root, 0)
        return sb.toString()
    }

    @Suppress("DEPRECATION")
    private fun recycle(node: AccessibilityNodeInfo) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) node.recycle()
    }
}
