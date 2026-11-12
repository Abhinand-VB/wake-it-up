package dev.wakeitup.context

import dev.wakeitup.core.ConversationSnapshot

/**
 * Holds the most recent conversation snapshot so drafts can be checked without
 * walking the node tree on every keystroke (SOURCE_OF_TRUTH.md §5.4).
 * [version] changes whenever the snapshot content changes; it is part of the
 * QuickFilter key so a new incoming message re-triggers a check of the same draft.
 */
class ContextCache {
    @Volatile
    var snapshot: ConversationSnapshot = ConversationSnapshot.EMPTY
        private set

    @Volatile
    var version: Long = 0L
        private set

    fun update(newSnapshot: ConversationSnapshot) {
        val changed = newSnapshot.messages != snapshot.messages || newSnapshot.replyIndex != snapshot.replyIndex
        snapshot = newSnapshot
        if (changed) version++
    }

    fun clear() {
        if (snapshot.messages.isNotEmpty()) version++
        snapshot = ConversationSnapshot.EMPTY
    }
}
