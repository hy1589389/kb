"""引用后置校验：提取 `[n]`、切句、统计未引用句子。

放在生成层而不是塞进 `pipeline.py`，是因为这三件事都是纯函数，评测阶段
（Phase 3）和 Web 层都能直接复用；也让 `pipeline.answer()` 的流程保持在
「检索 -> 提示词 -> 模型 -> 校验」四条主线内，不掺字符串处理细节。

## 为什么值得单独校验

模型偶尔会引用不存在的编号（`[7]` 但只给了 5 块资料），或者整段答案一个
`[n]` 都不标。前者是幻觉的硬信号，后者是「答案没落地到资料上」的软信号。
两件事都不该让答案直接通过而不留痕，所以 `Answer` 里存了 `invalid_refs`
和 `uncited_sentences`，评测脚本可以拿它们当指标。

## 切句规则（故意做得简单）

按 `。！？\n` 切，丢掉空串和过短的残句（`是的。`、`注意：` 这种）。不做
分号/冒号切分，也不处理引号内的句号 —— 中文技术答案里这两类切错只会让
句子粒度更碎，不会让 `uncited_sentences` 失真到不可用。markdown 结构行
（标题、表格、代码围栏）不计入句子，否则一个表格页会把指标拉满。
"""

from __future__ import annotations

import re

__all__ = [
    "extract_refs",
    "invalid_refs",
    "split_sentences",
    "count_uncited",
    "GAP_PREFIX",
    "MIN_SENTENCE_CHARS",
]

# `[n]` 里的 n 最多 4 位：答案正文不会有 9999 条引用，限制位数可以避免
# 把 markdown 链接语法 `[文字](url)` 里的方括号误判成引用 —— 那种方括号
# 里不是纯数字，正则天然不匹配，位数限制只是再加一道。
_REF_RE = re.compile(r"\[(\d{1,4})\]")

# 去掉引用标记后，字符数低于这个值的残句不计入统计。
MIN_SENTENCE_CHARS = 4

# 「资料未覆盖」这类句子不是事实主张，是**声明缺口**：拒答复核之后模型会写
# 「资料未覆盖：<子问题>」，这句本来就该没有 [n]，也不该被逐句忠实度判定当成
# 编造。所以切句和引用统计都把它排除掉。
GAP_PREFIX = "资料未覆盖"

_SENT_END = "。！？\n"

# 这些行是文档结构，不是论断，不参与「有没有引用」的统计。
_STRUCTURAL_PREFIXES = ("```", "|", "#", ">", "---", "===")


def extract_refs(text: str) -> list[int]:
    """按出现顺序返回正文里的所有引用编号（保留重复、不排序）。"""
    return [int(m.group(1)) for m in _REF_RE.finditer(text or "")]


def invalid_refs(text: str, n_citations: int) -> list[int]:
    """返回正文里引用了、但不在 1..n_citations 范围内的编号（升序去重）。"""
    bad = {n for n in extract_refs(text) if n < 1 or n > int(n_citations)}
    return sorted(bad)


def _is_meaningful(sentence: str) -> bool:
    """这句话值不值得计入「有没有引用」的统计。"""
    raw = sentence.strip()
    if not raw:
        return False
    if raw.startswith(_STRUCTURAL_PREFIXES):
        return False
    if raw.startswith(GAP_PREFIX):
        # 「资料未覆盖：X」是缺口声明，不是需要引用的论断
        return False
    body = _REF_RE.sub("", raw)
    # 只留下字母 / 数字 / 汉字，标点和空白不算长度
    body = re.sub(r"[\s\W_]+", "", body, flags=re.UNICODE)
    return len(body) >= MIN_SENTENCE_CHARS


def split_sentences(text: str) -> list[str]:
    """按 `。！？\\n` 切句，保留句末标点，过滤空串和过短残句。"""
    out: list[str] = []
    buf: list[str] = []
    for ch in text or "":
        if ch in _SENT_END:
            buf.append(ch)
            sentence = "".join(buf)
            if _is_meaningful(sentence):
                out.append(sentence.strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf)
    if _is_meaningful(tail):
        out.append(tail.strip())
    return out


def count_uncited(text: str) -> tuple[int, int]:
    """返回 `(没有引用标记的句子数, 参与统计的句子总数)`。"""
    sentences = split_sentences(text)
    uncited = sum(1 for s in sentences if not _REF_RE.search(s))
    return uncited, len(sentences)
