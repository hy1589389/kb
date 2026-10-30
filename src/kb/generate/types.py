"""生成层的数据结构：`Citation` 和 `Answer`。

`Citation` 是检索块在答案里的投影 —— 只保留回链需要的元数据，正文可以
截断（`full_length` 记原长度，`truncated` 标是否截过）。`Answer` 把
「问题 -> 检索 -> 模型 -> 后置校验」这一整轮的所有结果装在一起：

    text            答案正文，含 [n] 标记
    citations       与正文 [n] 一一对应的引用（ref 从 1 开始）
    refused         是否拒答；拒答时 text 是一句人话，不是模型原文
    hits            原始检索结果，`--show-hits` 和调试用
    invalid_refs    正文里引用了但 citations 里不存在的编号
    uncited_sentences / sentence_count   没有引用标记的句子数，供 Phase 3 评测使用

`render()` 是给人看的；`to_dict()` 是给 `--json` 和 Web 层用的。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..retrieve.types import SearchHit

__all__ = ["Citation", "Answer", "REFUSE_TEXT"]

# 拒答时正文里的那句话。两层拒答（检索层 / 模型层）共用它，
# 保证「拒答」在输出上只有一种样子。
REFUSE_TEXT = "知识库中没有找到足够的信息来回答这个问题。"

# 正文截断标记。单独定义是为了测试能断言「截过」而不是断言完整尾巴。
TRUNCATE_MARK = "…（本块已截断）"


@dataclass
class Citation:
    """一条引用。

    `text` 与送进提示词的正文一致（都按 `max_chunk_chars` 截断），
    `full_length` 是块的真实字符数。截断而不是全量保存的原因：一段答案
    的引用可能有 5~10 条，全量正文会让 `--json` 输出膨胀到几百 KB。
    """

    ref: int
    chunk_id: str
    repo: str
    rel_path: str
    breadcrumb: str
    line_start: int
    line_end: int
    text: str
    score: float
    full_length: int = 0
    truncated: bool = False

    @classmethod
    def from_hit(cls, hit: SearchHit, ref: int,
                 max_chars: int | None = None) -> "Citation":
        """由检索结果构造引用；`max_chars` 为 None 或 0 表示不截断。"""
        text = hit.text or ""
        full = len(text)
        truncated = bool(max_chars and full > int(max_chars))
        if truncated:
            text = text[: int(max_chars)].rstrip() + TRUNCATE_MARK
        return cls(
            ref=int(ref),
            chunk_id=hit.chunk_id,
            repo=hit.repo,
            rel_path=hit.rel_path,
            breadcrumb=hit.breadcrumb,
            line_start=int(hit.line_start),
            line_end=int(hit.line_end),
            text=text,
            score=float(hit.score),
            full_length=full,
            truncated=truncated,
        )

    def location(self) -> str:
        """`repo/rel_path:12-34`，与 `SearchHit.location()` 同一格式。"""
        path = f"{self.repo}/{self.rel_path}" if self.repo else self.rel_path
        if self.line_start:
            return f"{path}:{self.line_start}-{self.line_end}"
        return path

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Answer:
    """一轮问答的完整结果。

    字段顺序按「先看结论、再看依据、最后看过程」排。后四个字段带默认值，
    这样只关心 `text` 的调用方（比如评测脚本）可以少填几个参数。
    """

    question: str
    text: str
    citations: list[Citation]
    refused: bool
    refuse_reason: str
    hits: list[SearchHit]
    model: str
    usage: dict
    latency_ms: int
    # ---- 后置校验结果 ----
    invalid_refs: list[int] = field(default_factory=list)
    uncited_sentences: int = 0
    sentence_count: int = 0
    # 实际送进提示词的块数（受 max_context_chars 限制，<= len(hits)）
    context_chunks: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "refused": self.refused,
            "refuse_reason": self.refuse_reason,
            "hits": [h.to_dict() for h in self.hits],
            "model": self.model,
            "usage": dict(self.usage or {}),
            "latency_ms": int(self.latency_ms),
            "invalid_refs": list(self.invalid_refs),
            "uncited_sentences": int(self.uncited_sentences),
            "sentence_count": int(self.sentence_count),
            "context_chunks": int(self.context_chunks),
        }

    def render(self) -> str:
        """人类可读文本：答案 + 引用列表（含 breadcrumb 与 文件:行号）。"""
        lines: list[str] = []
        if self.refused:
            lines.append(f"[拒答] {self.refuse_reason}")
        lines.append(self.text or "（空答案）")
        lines.append("")

        if self.citations:
            lines.append(f"引用（{len(self.citations)} 条）：")
            for c in self.citations:
                lines.append(f"[{c.ref}] {c.location()}  {c.breadcrumb or '(无标题)'}")
        else:
            lines.append("引用：无")

        if self.invalid_refs:
            bad = " ".join(f"[{n}]" for n in self.invalid_refs)
            lines.append(
                f"[!] 答案引用了不存在的编号：{bad}"
                f"（有效范围 1..{len(self.citations)}）"
            )
        if self.sentence_count and self.uncited_sentences:
            lines.append(
                f"[!] {self.sentence_count} 句中 {self.uncited_sentences} 句没有引用标记"
            )
        return "\n".join(lines)
