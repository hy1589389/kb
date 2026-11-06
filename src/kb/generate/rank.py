"""LLM 名单式精排：把检索结果列给模型，让它挑出真正相关的并排序。

为什么不用交叉编码器：`bge-reranker-base` 在同一份 40 条评测集上把 R@5 从 85.0%
压到 80.0%、MRR 从 0.667 掉到 0.580，对中文技术文本区分力不足；对「token怎么控制」
这类口语查询还会把「全栈学习路线」顶到第一。

让生成模型直接读标题和片段做判断，效果好得多。同一句查询实测（2026-11-06）：

    原顺序：Token预算 / 采样解码 / 适合谁看 / Fencing Token / 投机解码 / 全栈路线 /
            LLM运行机制 / 接口幂等Token / 面试准备要点 / Token预算公式
    模型回答：1,10,7  →  Token预算 / Token预算公式 / LLM运行机制

噪音（工具书的「适合谁看」、鉴权用的 Fencing Token、鉴权 Token 机制）全部剔掉。
代价是一次 chat 调用：约 1,200 token、1~2 秒。

**它不做兜底**：模型回答「0」就是「没有相关的」，调用方如实返回空列表；解析失败
或调用出错返回 None，调用方退回原顺序 —— 精排是加分项，不能变成单点。
"""

from __future__ import annotations

import re

__all__ = ["LLMReranker", "RERANK_PROMPT"]

RERANK_PROMPT = (
    "用户在技术知识库里检索，下面是从知识库召回的结果。\n"
    "判断每一条是否真的与问题相关：只保留能直接回答、或提供关键背景的结果；"
    "只是词面重合、主题不同（比如问「大模型的 Token 怎么控制」，却召回「鉴权 Token」"
    "或「工具书目录」）的都不算相关。\n"
    "把保留的结果按相关度从高到低排序，只输出编号，用逗号分隔，例如：2,5,1。"
    "不要输出理由。一条都不相关就输出 0。"
)

_ID_RE = re.compile(r"\d+")


class LLMReranker:
    """用一次 LLM 调用对候选结果做名单式重排。失败返回 None。"""

    def __init__(self, llm, *, snippet_chars: int = 150, max_candidates: int = 15) -> None:
        self.llm = llm
        self.snippet_chars = snippet_chars
        self.max_candidates = max_candidates

    @property
    def available(self) -> bool:
        return self.llm is not None

    # ------------------------------------------------------------ 对外

    def rerank(self, question: str, hits: list) -> list | None:
        """返回重排后的命中列表（可能变短）；无法判断时返回 None。"""
        text = str(question or "").strip()
        candidates = list(hits)[: self.max_candidates]
        if self.llm is None or not text or not candidates:
            return None

        try:
            result = self.llm.complete(
                [{"role": "system", "content": RERANK_PROMPT},
                 {"role": "user", "content": self._render(text, candidates)}],
                temperature=0.0,
                max_tokens=64,
            )
        except Exception:  # noqa: BLE001 —— LLMError / 网络 / 超时都退回原顺序
            return None

        answer = result[0] if isinstance(result, tuple) else result
        order = self._parse(answer, len(candidates))
        if order is None:
            return None
        return [candidates[index] for index in order]

    # ------------------------------------------------------------ 内部

    def _render(self, question: str, hits: list) -> str:
        lines = [f"问题：{question}", "", "检索结果："]
        for index, hit in enumerate(hits, start=1):
            data = hit if isinstance(hit, dict) else getattr(hit, "to_dict", lambda: {})()
            breadcrumb = str(data.get("breadcrumb") or "(无标题)")
            snippet = str(data.get("text") or "").replace("\n", " ")
            lines.append(f"[{index}] {breadcrumb}")
            lines.append(f"    {snippet[:self.snippet_chars]}")
        return "\n".join(lines)

    @staticmethod
    def _parse(answer: object, total: int) -> list[int] | None:
        """把 "2,5,1" 解析成 0 基下标；"0" 表示没有相关结果，非法输入返回 None。"""
        text = str(answer or "").strip()
        if not text:
            return None
        numbers = [int(item) for item in _ID_RE.findall(text)]
        if not numbers:
            return None
        if numbers == [0]:
            return []
        order: list[int] = []
        for number in numbers:
            index = number - 1
            if 0 <= index < total and index not in order:
                order.append(index)
        return order or None
