"""Markdown 结构感知切块器 —— 本项目的第一个技术亮点。

为什么不按固定字符数切？
  技术文档 / 八股仓库的 Markdown 有天然结构：标题层级就是语义边界，
  代码块就是不可分割单元。固定 512 字符硬切会同时犯两个错：
    1. 把一个函数/一段配置劈成两半，检索命中的是半截代码；
    2. 把「TCP 三次握手」的标题和它的正文分到不同块，检索失去语境。

本模块的策略：
  · 按 ATX 标题（# ~ ######）切分为 Section，记录完整标题路径；
  · 切块绝不跨越围栏代码块；
  · 过小的相邻 Section 合并（同父节点），过大的 Section 按段落边界二次切分；
  · 每个 chunk 记录 line_start / line_end，可直接回链到原文行。

这些设计让你后面能做「混合检索 + 重排 + 引用溯源」，而不是只有一个黑盒。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .clean import FENCE_RE, clean_text, collapse_blanks, mark_code_fences

_RE_ATX = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
# 中文句读优先，其次英文句读，最后逗号级；用于切开「没有空行的超长段落」
_RE_SENTENCE = re.compile(r"(?<=[。！？；!?;])|(?<=[，、,])")


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    """一个标题及其名下的正文。"""

    path: tuple[str, ...]  # 标题路径，如 ("Java 基础", "集合", "HashMap")
    level: int
    text: str
    line_start: int  # 1-based，含
    line_end: int  # 1-based，含


@dataclass
class Chunk:
    """切块结果，也就是最终会被嵌入与检索的最小单元。"""

    text: str
    path: tuple[str, ...]
    line_start: int
    line_end: int
    index: int = 0
    part: int = 0  # 同一 Section 被拆开时的序号，0 表示未拆
    n_parts: int = 1
    merged_paths: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.path)

    @property
    def embed_text(self) -> str:
        """真正送进嵌入模型的内容：面包屑 + 正文。

        把标题路径拼进来，能显著改善「问题里只有关键词、正文里没有」的召回。
        """
        head = self.breadcrumb
        return f"{head}\n\n{self.text}".strip() if head else self.text.strip()


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def parse_sections(md_text: str) -> tuple[str, list[Section]]:
    """把 Markdown 解析成 (文档标题, Section 列表)。"""
    # 关键：lines 与 flags 必须严格一一对应，绝对不能在这里改动行数。
    # 空行压缩放在 flush() 里对「拼好的字符串」做，不影响行号。
    lines, flags = clean_text(md_text)

    title = ""
    sections: list[Section] = []

    stack: list[tuple[int, str]] = []  # [(level, heading)]
    buf: list[str] = []
    buf_start = 0  # 0-based，正文首行索引
    cur_path: tuple[str, ...] = ()
    cur_level = 0

    def flush(end_exclusive: int) -> None:
        nonlocal buf
        text = collapse_blanks("\n".join(buf)).strip()
        if text:
            sections.append(
                Section(
                    path=cur_path,
                    level=cur_level,
                    text=text,
                    line_start=buf_start + 1,
                    line_end=end_exclusive,
                )
            )
        buf = []

    for i, (line, in_code) in enumerate(zip(lines, flags)):
        if not in_code:
            m = _RE_ATX.match(line)
            if m:
                flush(i)
                level = len(m.group(1))
                heading = m.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                if level == 1 and not title:
                    title = heading
                stack.append((level, heading))
                cur_path = tuple(h for _, h in stack)
                cur_level = level
                buf_start = i + 1  # 正文从标题下一行开始
                continue
        buf.append(line)

    flush(len(lines))
    return title, sections


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #
def _split_long_line(line: str, max_chars: int) -> list[str]:
    """切开「单行就超长」的文本（没有空行的大段落、压缩过的 JSON 等）。

    先按句读边界切，再贪心装箱；仍有超长片段就退化为字符级硬切。
    这类文本在真实语料里很常见，早期版本按行切会直接切不动，产出 3000+ 字的巨块。
    """
    pieces = [p for p in _RE_SENTENCE.split(line) if p]
    out: list[str] = []
    cur: list[str] = []
    size = 0

    for p in pieces:
        while len(p) > max_chars:  # 单句仍超长 → 字符级硬切
            if cur:
                out.append("".join(cur))
                cur, size = [], 0
            out.append(p[:max_chars])
            p = p[max_chars:]
        if cur and size + len(p) > max_chars:
            out.append("".join(cur))
            cur, size = [], 0
        cur.append(p)
        size += len(p)
    if cur:
        out.append("".join(cur))
    return out


def _hard_split(text: str, max_chars: int) -> list[str]:
    """按行边界切分超长段落，并且**绝不把围栏代码块切成两半**。

    做法：先标记每行是否在代码块内，切分点只能落在代码块之外。
    代价是含超长代码块的 chunk 会超出 max_chars —— 这是刻意的取舍：
    一个完整的、可复制运行的代码块，比一个刚好卡在字数上限的半截代码有用得多。
    """
    lines = text.split("\n")
    flags = mark_code_fences(lines)

    out: list[str] = []
    cur: list[str] = []
    size = 0

    for ln, in_code in zip(lines, flags):
        if not in_code and len(ln) > max_chars:
            # 超长单行：先收尾，再按句读切开
            if cur:
                out.append("\n".join(cur))
                cur, size = [], 0
            out.extend(_split_long_line(ln, max_chars))
            continue
        # 只在代码块之外允许切分
        if cur and not in_code and size + len(ln) + 1 > max_chars:
            out.append("\n".join(cur))
            cur, size = [], 0
        cur.append(ln)
        size += len(ln) + 1

    if cur:
        out.append("\n".join(cur))
    return out


def _is_closing_fence(line: str, open_char: str, open_len: int) -> bool:
    m = FENCE_RE.match(line)
    if not m:
        return False
    marker, rest = m.group(1), m.group(2).strip()
    return marker[0] == open_char and len(marker) >= open_len and rest == ""


def _chop_code_block(block: list[str], marker: str, info: str, max_chars: int) -> list[str]:
    """把一个超大代码块拆成若干合法的小代码块（每个都补全围栏）。

    两层策略：
      1. 优先在空行处断开 —— 代码块内部的空行通常就是函数/段落边界，语义损失最小；
      2. 若某一段内部根本没有空行（压缩过的 JSON、长配置 dump），退化为按行硬切。

    只用第一层是不够的：实测有 7KB 的配置示例内部一个空行都没有，
    结果整块原样保留，把尺寸分布的长尾又拉了回来。
    """
    body = block[1:-1]

    coarse: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for ln in body:
        if cur and not ln.strip() and size > max_chars:
            coarse.append(cur)
            cur, size = [], 0
            continue
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        coarse.append(cur)

    pieces: list[list[str]] = []
    for piece in coarse:
        if sum(len(x) + 1 for x in piece) <= max_chars:
            pieces.append(piece)
            continue
        buf: list[str] = []
        size = 0
        for ln in piece:
            if len(ln) > max_chars:
                # 第三层：单行本身就超长。真实案例是 self-llm 里推理结果的 repr dump，
                # 一行 7172 字符、整块只有 3 行 —— 按行切完全切不动。
                if buf:
                    pieces.append(buf)
                    buf, size = [], 0
                pieces.extend([seg] for seg in _split_long_line(ln, max_chars))
                continue
            if buf and size + len(ln) + 1 > max_chars:
                pieces.append(buf)
                buf, size = [], 0
            buf.append(ln)
            size += len(ln) + 1
        if buf:
            pieces.append(buf)

    out: list[str] = []
    for k, piece in enumerate(pieces):
        if k:
            out.append("")  # 子块之间留空行，让后续按段落切分能找到边界
        out.append(marker + info)
        out.extend(piece)
        out.append(marker)
    return out


def _split_oversized_code_blocks(text: str, max_chars: int) -> str:
    """把超过 max_chars 的代码块，在空行处拆成多个代码块。

    这是「绝不切断代码块」原则的唯一例外，也是被真实数据打脸后的修正：
    self-llm 的 Gemma3 教程里有 56KB 的 base64 图片示例，硬留着的话
    它既超出嵌入模型上下文（会被静默截断），又永远不可能被检索命中。
    超过 2 倍 max_chars 的代码块，拆开比留着有用。
    """
    lines = text.split("\n")
    n = len(lines)
    out: list[str] = []
    i = 0

    while i < n:
        m = FENCE_RE.match(lines[i])
        if m is None:
            out.append(lines[i])
            i += 1
            continue

        marker, info = m.group(1), m.group(2)
        j = i + 1
        while j < n and not _is_closing_fence(lines[j], marker[0], len(marker)):
            j += 1

        if j >= n:
            # 围栏没闭合（脏数据），当普通行处理，避免死循环
            out.append(lines[i])
            i += 1
            continue

        block = lines[i: j + 1]
        if sum(len(x) + 1 for x in block) <= max_chars:
            out.extend(block)
        else:
            out.extend(_chop_code_block(block, marker, info, max_chars))
        i = j + 1

    return "\n".join(out)


def _opening_fence_flags(lines: list[str]) -> list[bool]:
    """标记每行是否为「开启围栏」。

    不能靠 `mark_code_fences` 的 in_code 推出来：闭合围栏那一行本身也被标记为
    in_code=True，于是紧随其后的开启围栏会被误判成「仍在代码块内的普通行」。
    """
    out: list[bool] = []
    in_code = False
    fence_char = ""
    fence_len = 0

    for line in lines:
        m = FENCE_RE.match(line)
        if m and not in_code:
            in_code, fence_char, fence_len = True, m.group(1)[0], len(m.group(1))
            out.append(True)
            continue
        if m and in_code:
            marker, rest = m.group(1), m.group(2).strip()
            if marker[0] == fence_char and len(marker) >= fence_len and rest == "":
                in_code, fence_char, fence_len = False, "", 0
        out.append(False)
    return out


def _split_paragraphs(text: str) -> list[str]:
    """按空行切段，**但代码块内部的空行不算边界**。

    两处关键细节，都是被真实数据打出来的：
      1. 代码块内部的空行不是段落边界 —— 早期版本用正则 split 空行，
         把一个内部含空行的代码块拦腰截断，前半段少了闭合围栏、
         后半段多了孤儿围栏，Markdown 直接坏掉；
      2. **开启围栏本身也是段落边界** —— leetcode-master 里有三个代码块
         紧挨着、中间一个空行都没有的写法（闭合围栏的下一行就是开启围栏），
         只认空行的话这三块会被当成一整段，而 _hard_split 又不能切代码块，
         结果 5000+ 字的巨块原样留了下来。
    """
    lines = text.split("\n")
    flags = mark_code_fences(lines)
    opens = _opening_fence_flags(lines)

    paragraphs: list[str] = []
    cur: list[str] = []

    for ln, in_code, is_open in zip(lines, flags, opens):
        if not in_code and not ln.strip():
            if cur:
                paragraphs.append("\n".join(cur))
                cur = []
            continue

        if is_open and cur:
            paragraphs.append("\n".join(cur))
            cur = []

        cur.append(ln)

    if cur:
        paragraphs.append("\n".join(cur))
    return paragraphs


def _split_long(
    text: str,
    max_chars: int,
    overlap: int,
    code_block_max_chars: int,
) -> list[str]:
    """超长 Section 按段落边界切分，并在块首补上一段上文（overlap）。"""
    text = _split_oversized_code_blocks(text, code_block_max_chars)

    paragraphs = _split_paragraphs(text)
    parts: list[str] = []
    cur: list[str] = []
    size = 0

    for p in paragraphs:
        if len(p) > max_chars:
            if cur:
                parts.append("\n\n".join(cur))
                cur, size = [], 0
            parts.extend(_hard_split(p, max_chars))
            continue
        if cur and size + len(p) + 2 > max_chars:
            parts.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(p)
        size += len(p) + 2
    if cur:
        parts.append("\n\n".join(cur))

    # overlap 前缀是「上一块的尾巴」，如果尾巴里含围栏，复制过去会让这一块的
    # 围栏不成对、Markdown 直接坏掉。涉及代码块时宁可不加 overlap。
    if overlap > 0 and len(parts) > 1:
        with_overlap = [parts[0]]
        for i in range(1, len(parts)):
            tail = parts[i - 1][-overlap:]
            if any(f in tail or f in parts[i] for f in ("```", "~~~")):
                with_overlap.append(parts[i])
            else:
                with_overlap.append(tail + "\n\n" + parts[i])
        parts = with_overlap
    return parts


def _common_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return tuple(out)


def _merge_small(chunks: list[Chunk], min_chars: int, max_chars: int) -> list[Chunk]:
    """把「过小的相邻块」合并成一个。

    合并条件（必须同时满足，否则宁可不合 —— 把不相关内容搅在一起，
    嵌入向量会被稀释，检索反而更差）：
      · 两块都小于 min_chars（不能把一个大章节吸进小碎片里）；
      · 两块同属一个父标题，或后者是前者的子节点；
      · 合并后不超过 max_chars。

    合并后 breadcrumb 取两者的「最深公共前缀」，而不是沿用前一块的路径 ——
    否则 chunk 里明明有三节内容，却标着第一节的标题，引用溯源会误导人。
    """
    if not chunks:
        return []

    # ---- 第一轮：两个都小的相邻兄弟块合并 ----
    merged: list[Chunk] = [chunks[0]]
    for nxt in chunks[1:]:
        prev = merged[-1]
        same_parent = prev.path[:-1] == nxt.path[:-1]
        nested = nxt.path[: len(prev.path)] == prev.path
        combined = len(prev.text) + len(nxt.text) + 2

        if (
            len(prev.text) < min_chars
            and len(nxt.text) < min_chars
            and (same_parent or nested)
            and combined <= max_chars
        ):
            prev.path = _common_prefix(prev.path, nxt.path)
            prev.text = prev.text + "\n\n" + nxt.text
            prev.line_end = nxt.line_end
            prev.merged_paths.append(nxt.path)
        else:
            merged.append(nxt)

    # ---- 第二轮：尾巴吸收 ----
    # 仍然过小的碎片（例如「为什么是 8」这种一句话小节），挂到前一块上。
    # 宁可让前一块稍微杂一点，也不能静默丢内容 —— 丢了就再也检索不到了。
    absorbed: list[Chunk] = [merged[0]] if merged else []
    for nxt in merged[1:]:
        prev = absorbed[-1]
        same_parent = prev.path[:-1] == nxt.path[:-1]
        nested = nxt.path[: len(prev.path)] == prev.path
        if (
            len(nxt.text) < min_chars
            and (same_parent or nested)
            and len(prev.text) + len(nxt.text) + 2 <= max_chars
        ):
            prev.path = _common_prefix(prev.path, nxt.path)
            prev.text = prev.text + "\n\n" + nxt.text
            prev.line_end = nxt.line_end
            prev.merged_paths.append(nxt.path)
        else:
            absorbed.append(nxt)
    return absorbed


def chunk_document(
    md_text: str,
    *,
    max_chars: int = 1200,
    min_chars: int = 120,
    drop_below: int = 40,
    overlap: int = 120,
    code_block_max_chars: int = 2400,
) -> tuple[str, list[Chunk]]:
    """把一个 Markdown 文档切成 Chunk 列表，返回 (文档标题, chunks)。"""
    title, sections = parse_sections(md_text)

    raw: list[Chunk] = []
    for sec in sections:
        if len(sec.text) <= max_chars:
            raw.append(
                Chunk(
                    text=sec.text,
                    path=sec.path,
                    line_start=sec.line_start,
                    line_end=sec.line_end,
                )
            )
            continue
        parts = _split_long(sec.text, max_chars, overlap, code_block_max_chars)
        for j, part in enumerate(parts):
            raw.append(
                Chunk(
                    text=part,
                    path=sec.path,
                    line_start=sec.line_start,
                    line_end=sec.line_end,
                    part=j,
                    n_parts=len(parts),
                )
            )

    chunks = _merge_small(raw, min_chars, max_chars)
    chunks = [c for c in chunks if len(c.text) >= drop_below]

    for i, c in enumerate(chunks):
        c.index = i
    return title, chunks


def make_doc_id(repo: str, rel_path: str) -> str:
    return hashlib.sha1(f"{repo}/{rel_path}".encode("utf-8")).hexdigest()[:16]


def make_chunk_id(doc_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text[:256].encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}-{index:04d}-{digest}"
