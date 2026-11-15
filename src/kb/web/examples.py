"""首页的示例问题。

这些题不是随手编的：从 `eval/questions.jsonl` 的 40 条里挑「实测答得好」的 ——
支持率 100%、至少 2 条引用、没被拒答（依据 `results/run_answer_final.json`），
每个题型各一条，这样点下去不会得到一段「资料不足」。

放后端而不是写死在前端，理由有两条：
  · 换语料 / 换评测集时要跟着换，前端 96 KB 的单文件不适合塞这种会变的内容；
  · 能被测试盯住（字段、模式、去重、数量上限）。

模式分两种：`ask` 走问答（答案带引用），`search` 只检索（秒回，适合只想看看
能搜到什么的人）。
"""

from __future__ import annotations

from typing import Any

__all__ = ["EXAMPLES", "MAX_EXAMPLES", "default_examples", "normalize_examples"]

# 首页最多显示几条：三到四条够了，再多就把「输入框」挤下去了。
MAX_EXAMPLES = 4

# mode: search = 只检索；ask = 检索 + 模型作答（带引用）
EXAMPLES: tuple[dict[str, str], ...] = (
    {
        "q": "为什么扩容要重新计算哈希？扩容那次为什么是 O(n) 的代价，"
             "不是说哈希表插入是 O(1) 吗？",
        "mode": "ask",
        "tag": "HashMap",
    },
    {
        "q": "CS-Notes 里 24. 反转链表那篇，用迭代（头插法）反转单链表的 Java 代码怎么写？",
        "mode": "ask",
        "tag": "代码",
    },
    {
        "q": "JVM 里 G1 和 CMS 收集器在回收算法和停顿控制上有什么区别？",
        "mode": "ask",
        "tag": "JVM",
    },
    {
        "q": "HashMap 什么时候转红黑树，什么时候退回链表？",
        "mode": "search",
        "tag": "检索",
    },
)

_VALID_MODES = ("search", "ask")


def normalize_examples(items: Any) -> list[dict[str, str]]:
    """过滤掉不合规的示例：没有 q、模式不认识、重复问题、超过上限。

    不做异常而是静默丢弃 —— 示例只是首页装饰，配错一条不该让服务起不来。
    但顺序保留，方便按「最想展示的排前面」来排。
    """
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items or ():
        if not isinstance(item, dict):
            continue
        question = str(item.get("q") or "").strip()
        mode = str(item.get("mode") or "ask").strip().lower()
        if not question or mode not in _VALID_MODES or question in seen:
            continue
        seen.add(question)
        out.append({"q": question, "mode": mode, "tag": str(item.get("tag") or "").strip()})
        if len(out) >= MAX_EXAMPLES:
            break
    return out


def default_examples() -> list[dict[str, str]]:
    """默认示例（拷贝，调用方可以随便改）。"""
    return normalize_examples(EXAMPLES)
