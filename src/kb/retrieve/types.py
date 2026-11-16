"""检索结果的统一数据结构。

`SearchHit` 是检索层对外的唯一输出类型。BM25 / 稠密 / 融合 / 重排四条
路径都返回它，评测脚本只需要认识这一个类型。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["SearchHit"]


@dataclass
class SearchHit:
    """一条检索结果。

    字段与 `chunks_all.jsonl` 里的元数据一一对应，`line_start`/`line_end`
    用来回链原文（前端可以跳转到 GitHub 对应行）。

    `score` 的含义随 `source` 变化：dense 是余弦相似度，sparse 是 BM25 分，
    hybrid 是 RRF 融合分，rerank 是重排模型的 logit。跨 source 比较分数
    没有意义，比较名次才有意义 —— 这是 RRF 存在的理由。

    多集合相关的两个字段：

    `collection` 是来源集合名（`tech` / `papers` / `profile`）。单索引模式下
    从 chunk 元数据里读，老索引没有这个字段时为空串。

    `locator` 是人类可读的位置：Markdown 是 `L222-297`（起止行号），PDF 是
    `P3` / 跨页 `P3-4`（页码）。老索引没有这个字段时为空串，`location()`
    自动退回用 `line_start`/`line_end` 拼。
    """

    chunk_id: str
    doc_id: str
    repo: str
    rel_path: str
    breadcrumb: str
    line_start: int
    line_end: int
    text: str
    score: float
    source: str
    rank: int = 0  # 1-based；0 表示还没排名
    collection: str = ""
    locator: str = ""
    # 这一条在各路里的名次与原始分：{"dense": {"rank": 1, "score": 0.6229},
    # "sparse": {"rank": 21, "score": 6.31}}。hybrid 下 `score` 是 RRF 融合分，
    # 只有 0.03 量级、看绝对值没有意义；名次和原始分才能回答「为什么这条在前面」。
    # 单路检索时只有对应那一路的键。
    sources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SearchHit":
        return cls(
            chunk_id=d["chunk_id"],
            doc_id=d.get("doc_id", ""),
            repo=d.get("repo", ""),
            rel_path=d.get("rel_path", ""),
            breadcrumb=d.get("breadcrumb", ""),
            line_start=int(d.get("line_start", 0)),
            line_end=int(d.get("line_end", 0)),
            text=d.get("text", ""),
            score=float(d.get("score", 0.0)),
            source=d.get("source", ""),
            rank=int(d.get("rank", 0)),
            collection=d.get("collection", ""),
            locator=d.get("locator", ""),
            sources=dict(d.get("sources") or {}),
        )

    def location(self) -> str:
        """人类可读的来源位置：`repo/rel_path:12-34` 或 `repo/file.pdf:P3`。"""
        path = f"{self.repo}/{self.rel_path}" if self.repo else self.rel_path
        if self.locator:
            return f"{path}:{self.locator}"
        if self.line_start:
            return f"{path}:{self.line_start}-{self.line_end}"
        return path
