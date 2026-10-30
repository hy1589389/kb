"""提示词构造：把检索结果编号后塞进上下文，并守住长度上限。

## 上下文预算怎么算

两级上限：

    max_chunk_chars     单块正文最多留多少字（默认 1500）
    max_context_chars   所有块正文之和最多多少字（默认 6000）

先按块截断，再按排名累加；累加会超预算时**停止**，后面的块整块丢弃，
并在消息里写明「共 N 块，只提供前 M 块」。整块丢弃而不是截半块，是因为
半个块容易把「结论句」切掉，模型拿到残句反而更容易编。

预算只统计正文，不统计每块那几行「来源 / 标题」头（每条约 60 字）。
差这一点不影响上限的量级，换来实现简单、和 `Citation.text` 的截断口径
完全一致。

## 为什么编号就是引用编号

提示词里的 `[1] [2]` 和答案里的 `[1] [2]` 是同一套编号，所以
`citations[i-1]` 天然对应 `[i]`。`Answerer` 用 `select_hits()` 拿到「实际
进了提示词的那几块」，引用列表只从这些块生成 —— 否则模型引用了 `[4]`
而第 4 块其实因为超预算没给它，校验会把一个正确的引用判成越界。
"""

from __future__ import annotations

from dataclasses import replace

from ..retrieve.types import SearchHit

__all__ = [
    "CITE_FORMAT",
    "INSUFFICIENT",
    "SYSTEM_PROMPT",
    "build_messages",
    "merge_same_location",
    "prepare_blocks",
    "select_hits",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_MAX_CHUNK_CHARS",
]

DEFAULT_MAX_CONTEXT_CHARS = 6000
DEFAULT_MAX_CHUNK_CHARS = 1500


def merge_same_location(hits: list[SearchHit]) -> list[SearchHit]:
    """把指向同一处原文的多个块合并成一条。

    为什么需要：Phase 1 把一个超长 section 切成了多个 part，但每个 part 都继承了
    **整节**的 line_start/line_end（切块时拿不到逐段行号）。于是详情页会出现
    「引用 4 条，其中 3 条都指向 xxx.md:222-297」这种看着像 bug 的输出。

    关键是这些 part **内容确实不同**（同一节的不同段落），所以不能简单丢弃，
    否则会丢资料。这里按 (rel_path, line_start, line_end) 归并，正文用空行拼起来，
    位置取第一块的名次。合并后一个位置只产生一条引用，同时省下重复的 breadcrumb 开销。

    代价是合并后的块可能超过 max_chunk_chars，交给 _clip 截断即可。
    """
    if not hits:
        return []
    out: list[SearchHit] = []
    index: dict[tuple[str, int, int], int] = {}
    for hit in hits:
        key = (hit.rel_path, hit.line_start, hit.line_end)
        pos = index.get(key)
        if pos is None:
            index[key] = len(out)
            out.append(hit)
            continue
        prev = out[pos]
        merged = prev.text + "\n\n" + hit.text if hit.text else prev.text
        out[pos] = replace(prev, text=merged)
    return out

# 模型判断「资料不足」时唯一允许输出的标记。判定用 `in` 而不是 `==`，
# 因为模型偶尔会在标记前后加空白或句号。
INSUFFICIENT = "INSUFFICIENT"

CITE_FORMAT = (
    "引用格式：在每条关键论断的句末标注来源编号，写作 [n]，"
    "n 是资料前的方括号编号（从 1 开始）；一句话可以标多个，如 [1][3]。"
    "不要标资料编号以外的数字。"
)

SYSTEM_PROMPT = (
    "你是本地知识库的问答助手。必须遵守以下规则：\n"
    "1. 只能依据用户提供的资料回答，不得使用资料之外的知识，也不要根据常识补充。\n"
    "2. 每条关键论断后面都要标注引用编号，格式为 [n]，n 对应资料编号；"
    "可以在一句话末尾标多个，如 [1][3]。\n"
    f"3. 如果资料不足以回答问题，只输出一个词：{INSUFFICIENT}。"
    "不要输出解释、道歉或其他任何内容。\n"
    "4. 不要编造资料里没有的事实、数字、代码或文件名。\n"
    "5. 用中文回答，直接给结论，不要复述问题，不要输出「根据资料」这类开场白。\n"
    f"{CITE_FORMAT}"
)

_TRUNCATE_MARK = "…（本块已截断）"


def _clip(text: str, max_chars: int) -> str:
    """把单块正文截到 `max_chars` 以内，截过就加标记。

    按字符切而不是按 token：这里的目标是防止单块吃掉整个上下文，
    不是精确控制 token 数；中文 1500 字约 1500 token，量级够用。
    """
    text = (text or "").strip()
    if not max_chars or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars)].rstrip() + _TRUNCATE_MARK


def prepare_blocks(hits: list[SearchHit], *,
                   max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
                   max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
                   ) -> list[tuple[SearchHit, str]]:
    """按排名选出能进上下文的块，返回 `[(hit, 截断后的正文), ...]`。

    `max_context_chars <= 0` 视为不给上下文（返回空列表）。
    """
    budget = int(max_context_chars or 0)
    if budget <= 0:
        return []
    # 先归并同一原文位置的块：一个位置只占一条引用，也省下重复的 breadcrumb。
    hits = merge_same_location(hits)
    blocks: list[tuple[SearchHit, str]] = []
    used = 0
    for hit in hits:
        body = _clip(hit.text, int(max_chunk_chars or 0))
        if blocks and used + len(body) > budget:
            break
        if not blocks and len(body) > budget:
            # 第一条就超预算：硬截到预算内，否则一块都进不去，
            # 模型只会看到「没有资料」，白白浪费一次检索。
            body = body[:budget].rstrip() + _TRUNCATE_MARK
        blocks.append((hit, body))
        used += len(body)
    return blocks


def select_hits(hits: list[SearchHit], *,
                max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
                max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
                ) -> list[SearchHit]:
    """`prepare_blocks` 的薄包装：只要块、不要正文。

    引用列表必须由这批块生成，编号才能和提示词里的 `[n]` 对上。
    """
    return [hit for hit, _ in prepare_blocks(
        hits,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
    )]


def build_messages(question: str, hits: list[SearchHit], *,
                   max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
                   max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
                   ) -> list[dict]:
    """构造 `[system, user]` 两条消息。

    user 消息的骨架：

        注意：共检索到 N 块资料，受上下文长度限制只提供前 M 块。
        以下是从本地知识库检索到的资料，按相关度从高到低编号：
        [1] 来源：repo/path:12-34
            标题：一级 > 二级
            内容：
        ...
        问题：...
    """
    blocks = prepare_blocks(
        hits,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
    )
    parts: list[str] = []

    if not blocks:
        parts.append("（没有检索到任何资料）")
    else:
        if len(blocks) < len(hits):
            parts.append(
                f"注意：共检索到 {len(hits)} 块资料，"
                f"受上下文长度限制，资料已截断到前 {len(blocks)} 条。"
            )
        parts.append("以下是从本地知识库检索到的资料，按相关度从高到低编号：")
        for ref, (hit, body) in enumerate(blocks, start=1):
            parts.append(f"[{ref}] 来源：{hit.location()}")
            parts.append(f"    标题：{hit.breadcrumb or '(无标题)'}")
            parts.append("    内容：")
            parts.append(body)
            parts.append("")

    parts.append(f"问题：{(question or '').strip()}")
    parts.append(
        f"请依据以上资料回答。关键论断后用 [n] 标注来源；"
        f"资料不足时只输出 {INSUFFICIENT}。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]
