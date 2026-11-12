package dev.wakeitup.core

/** Model inputs for a single example (batch size 1, no padding). */
class EncodedInput(
    val inputIds: LongArray,
    val tokenTypeIds: LongArray,
    val attentionMask: LongArray,
) {
    val length: Int get() = inputIds.size
}

data class EncoderConfig(
    val maxLen: Int = 128,
    val contextMaxMessages: Int = 5,
    val contextSeparator: String = " | ",
    val replyPrefix: String = "reply_to ",
    val emptyContextToken: String = "none",
)

/**
 * Builds the (context, draft) pair input. This is the normative algorithm from
 * SOURCE_OF_TRUTH.md §6.2.1 and must stay identical to `training/tg_format.py`.
 */
class InputEncoder(
    private val tokenizer: WordPieceTokenizer,
    private val config: EncoderConfig = EncoderConfig(),
) {
    private val separatorIds: IntArray = tokenizer.tokenizeToIds(config.contextSeparator)
    private val emptyContextIds: IntArray = tokenizer.tokenizeToIds(config.emptyContextToken)

    fun encode(messages: List<ChatMessage>, replyIndex: Int?, draft: String): EncodedInput {
        // 1–3. Move the reply target to the end, keep the last N, format.
        val validReply = replyIndex?.takeIf { it in messages.indices }
        val ordered = ArrayList<String>(messages.size)
        messages.forEachIndexed { i, m -> if (i != validReply) ordered += "${m.speaker}: ${m.text}" }
        if (validReply != null) {
            val m = messages[validReply]
            ordered += "${config.replyPrefix}${m.speaker}: ${m.text}"
        }
        val formatted = ordered.takeLast(config.contextMaxMessages)

        // 4. Draft is protected; keep its last tokens if it alone is too long.
        val budget = config.maxLen - 3
        var draftIds = tokenizer.tokenizeToIds(draft)
        if (draftIds.size > budget - 1) {
            draftIds = draftIds.copyOfRange(draftIds.size - (budget - 1), draftIds.size)
        }
        val contextBudget = budget - draftIds.size

        // 5. Drop oldest messages until the context fits; never drop the last one.
        val msgs = ArrayDeque(formatted.map { tokenizer.tokenizeToIds(it) })
        fun total(): Int = msgs.sumOf { it.size } + (msgs.size - 1).coerceAtLeast(0) * separatorIds.size
        while (msgs.size > 1 && total() > contextBudget) msgs.removeFirst()
        if (msgs.size == 1 && msgs[0].size > contextBudget) {
            val only = msgs[0]
            msgs[0] = only.copyOfRange(only.size - contextBudget, only.size)
        }

        // 6. Join with the separator token(s); empty context becomes "none".
        val contextIds = ArrayList<Int>()
        msgs.forEachIndexed { i, ids ->
            if (i > 0) separatorIds.forEach { contextIds += it }
            ids.forEach { contextIds += it }
        }
        if (contextIds.isEmpty()) emptyContextIds.forEach { contextIds += it }

        // 7. [CLS] context [SEP] draft [SEP]
        val total = 1 + contextIds.size + 1 + draftIds.size + 1
        val inputIds = LongArray(total)
        val typeIds = LongArray(total)
        var p = 0
        inputIds[p++] = tokenizer.clsId.toLong()
        for (id in contextIds) inputIds[p++] = id.toLong()
        inputIds[p++] = tokenizer.sepId.toLong()
        val segmentBStart = p
        for (id in draftIds) inputIds[p++] = id.toLong()
        inputIds[p++] = tokenizer.sepId.toLong()
        for (i in segmentBStart until total) typeIds[i] = 1L
        return EncodedInput(inputIds, typeIds, LongArray(total) { 1L })
    }
}
