"""生成流水线：检索 -> 提示词 -> 模型 -> 引用校验。

## 拒答有两层，触发点不同

第一层在**调模型之前**，按顺序判断两件事：

  1. 检索结果为空            -> 直接返回 `refused=True`，原因「检索无结果」
  2. `min_top_score` 阈值     -> 排序第一的分数低于阈值就拒答

第 2 条有个容易踩的坑：混合检索的 `score` 是 RRF 融合分，量纲和余弦
相似度、BM25 分**完全不同** —— RRF 分大致落在 1/(60+1) ~ 1/(60+50)，
即 0.016 ~ 0.033，只表示名次，不表示相关程度。所以在 `source == "hybrid"`
（以及重排后的 `"rerank"`，分数是 logit）时，任何绝对阈值都没有意义。
本模块在这种情况下显式跳过阈值判断并打一条日志，而不是拿一个 cosine
阈值去卡 RRF 分 —— 那会把所有查询都拒掉。要用阈值请走单路检索
（`use_dense`/`use_sparse` 只开一个），此时 `score` 才是可比的原始分。

第二层在**模型输出之后**：回答里出现 `INSUFFICIENT` 标记即拒答，原因
「资料不足」，正文换成一句人话。这一层挡住的是「检索到了相关块，但块里
没有答案」—— 检索层无法区分这种情况，只有模型看过内容才知道。

## 为什么引用只从「进了提示词的块」生成

见 `prompt.select_hits` 的注释：编号必须和提示词一致。`Answer.hits` 保留
全部原始检索结果（`--show-hits` 要看），`Answer.citations` 只对应真正送
进上下文的那几块，所以 `[n]` 一定能在 citations 里找到（除非模型编了编号，
那正是 `invalid_refs` 要抓的）。

## 后置校验

    invalid_refs       正文里引用了但 citations 里没有的编号
    uncited_sentences  没有 [n] 标记的句子数（供 Phase 3 评测）
    sentence_count     参与统计的句子总数

规则在 `cite.py`，都是纯函数。拒答时这三项置 0：拒答正文是一句固定的话，
不是模型生成的论断，把它算成「未引用句子」会污染评测指标。
"""

from __future__ import annotations

import logging
import time

from ..retrieve.types import SearchHit
from .cite import count_uncited, invalid_refs
from .llm import LLMClient
from .prompt import (
    INSUFFICIENT,
    build_messages,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MAX_CONTEXT_CHARS,
    select_hits,
)
from .types import Answer, Citation, REFUSE_TEXT

__all__ = [
    "Answerer",
    "DEFAULT_CFG",
    "REFUSE_NO_HITS",
    "REFUSE_INSUFFICIENT",
    "REFUSE_LOW_SCORE",
]

log = logging.getLogger(__name__)

# 拒答原因。常量而不是散落的字符串：评测脚本要按原因分组统计。
REFUSE_NO_HITS = "检索无结果"
REFUSE_INSUFFICIENT = "资料不足"
REFUSE_LOW_SCORE = "相关度低于阈值"

# 与 config.toml 的 [generate] 段同名。显式传 cfg 时按 key 覆盖。
DEFAULT_CFG: dict = {
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/chat/completions",
    "api_key_env": "DEEPSEEK_API_KEY",
    "temperature": 0.1,
    "max_tokens": 1024,
    "max_context_chars": DEFAULT_MAX_CONTEXT_CHARS,
    "max_chunk_chars": DEFAULT_MAX_CHUNK_CHARS,
    "top_k": 5,
    "min_top_score": 0.0,
}

# 这些 source 的 score 不是可比的相似度，阈值对它们无效。
_THRESHOLDLESS_SOURCES = ("hybrid", "rerank")


class Answerer:
    """把 Retriever 和 LLMClient 拼成一轮问答。

    `retriever` 只要求有 `search(query, top_k=..., use_rerank=...)` 方法，
    所以测试里可以塞一个假对象，不需要真的索引和 2 GB 模型。
    """

    def __init__(self, retriever, llm: LLMClient, cfg: dict | None = None):
        self.retriever = retriever
        self.llm = llm
        self.cfg = dict(DEFAULT_CFG)
        if cfg:
            unknown = set(cfg) - set(DEFAULT_CFG)
            if unknown:
                # 和 Retriever 一致：不报错，但警告一次，避免「配了没生效」
                print(f"[!] 忽略未知的生成配置项：{sorted(unknown)}")
            self.cfg.update(cfg)

    # ------------------------------------------------------------ 对外

    def answer(self, question: str, *, top_k: int = 5,
               use_rerank: bool = False,
               collections: list[str] | str | None = None) -> Answer:
        """检索并生成带引用的答案。模型调用失败时抛 `LLMError`。

        `collections` 只在多集合模式下有意义：传了就把检索限制在那几个集合里，
        简历（`profile`）不会被带进技术问题的提示词。留空查全部。
        """
        t0 = time.perf_counter()
        question = question or ""
        k = int(top_k) if top_k else int(self.cfg["top_k"])

        hits = self._retrieve(question, k, use_rerank, collections)
        if not hits:
            return self._refuse(question, REFUSE_NO_HITS, hits, t0,
                                detail="检索没有返回任何块")

        if not self._passes_threshold(hits):
            return self._refuse(question, REFUSE_LOW_SCORE, hits, t0,
                                detail=f"top1 score={hits[0].score:.4f}")

        context_hits = select_hits(
            hits,
            max_context_chars=int(self.cfg["max_context_chars"]),
            max_chunk_chars=int(self.cfg["max_chunk_chars"]),
        )
        messages = build_messages(
            question, hits,
            max_context_chars=int(self.cfg["max_context_chars"]),
            max_chunk_chars=int(self.cfg["max_chunk_chars"]),
        )

        text, usage = self.llm.complete(
            messages,
            temperature=float(self.cfg["temperature"]),
            max_tokens=int(self.cfg["max_tokens"]),
        )
        text = (text or "").strip()
        usage = dict(usage or {})

        if INSUFFICIENT in text:
            return self._refuse(question, REFUSE_INSUFFICIENT, hits, t0,
                                usage=usage, context_chunks=len(context_hits))

        citations = [
            Citation.from_hit(hit, ref=i, max_chars=int(self.cfg["max_chunk_chars"]))
            for i, hit in enumerate(context_hits, start=1)
        ]
        bad = invalid_refs(text, len(citations))
        uncited, n_sentences = count_uncited(text)
        return Answer(
            question=question,
            text=text,
            citations=citations,
            refused=False,
            refuse_reason="",
            hits=hits,
            model=getattr(self.llm, "model", "") or "",
            usage=usage,
            latency_ms=_elapsed_ms(t0),
            invalid_refs=bad,
            uncited_sentences=uncited,
            sentence_count=n_sentences,
            context_chunks=len(citations),
        )

    # ------------------------------------------------------------ 内部

    def _retrieve(self, question: str, top_k: int,
                  use_rerank: bool,
                  collections: list[str] | str | None = None) -> list[SearchHit]:
        query = (question or "").strip()
        if not query:
            # 空问题不查索引：BM25 对空串返回空，但没必要走一遍
            return []
        # collections 留空时不传这个参数：单索引的 Retriever 和测试里的假
        # retriever 都只接受前三个关键字参数。
        extra = {"collections": collections} if collections else {}
        return self.retriever.search(query, top_k=top_k, use_rerank=use_rerank,
                                     **extra)

    def _passes_threshold(self, hits: list[SearchHit]) -> bool:
        """第一层的第 2 条：top1 分数是否达到 `min_top_score`。"""
        min_score = self.cfg.get("min_top_score") or 0
        try:
            min_score = float(min_score)
        except (TypeError, ValueError):
            log.warning("min_top_score 不是数字（%r），已忽略", min_score)
            return True
        if min_score <= 0:
            return True  # 0 / 空 = 关闭

        source = hits[0].source if hits else ""
        if source in _THRESHOLDLESS_SOURCES:
            log.warning(
                "min_top_score=%.4f 已跳过：当前 source=%r 的 score 不可跨模式"
                "比较（混合检索是 RRF 分，约 0.016~0.033；重排是 logit）。"
                "阈值只在单路检索（dense 或 sparse）下有效。",
                min_score, source,
            )
            return True

        return hits[0].score >= min_score

    def _refuse(self, question: str, reason: str, hits: list[SearchHit],
                t0: float, *, detail: str = "", usage: dict | None = None,
                context_chunks: int = 0) -> Answer:
        """构造拒答结果。正文固定一句人话，不做引用统计。"""
        log.info("拒答：%s（%s）", reason, detail or "-")
        return Answer(
            question=question,
            text=REFUSE_TEXT,
            citations=[],
            refused=True,
            refuse_reason=reason,
            hits=list(hits),
            model=getattr(self.llm, "model", "") or "",
            usage=dict(usage or {}),
            latency_ms=_elapsed_ms(t0),
            context_chunks=context_chunks,
        )


def _elapsed_ms(t0: float) -> int:
    return int(round((time.perf_counter() - t0) * 1000))
