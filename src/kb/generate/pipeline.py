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

from ..retrieve.tokenize import tokenize
from ..retrieve.types import SearchHit
from .cite import GAP_PREFIX, count_uncited, invalid_refs, split_sentences
from .llm import LLMClient
from .prompt import (
    INSUFFICIENT,
    build_messages,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MAX_CONTEXT_CHARS,
    RETRY_AFTER_INSUFFICIENT,
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
    # 模型说「资料不足」之后，再问一次（见 prompt.RETRY_AFTER_INSUFFICIENT）。
    #
    # `refusal_retry_min_coverage` 默认 0 = 一律复核。原本想用「问题关键词在上下文
    # 里的覆盖度」当闸门，实测这个信号不成立：40 道评测题的覆盖度中位数 0.55、
    # 最低 0.30，而「今天天气怎么样」「你好」这类域外问题反而是 1.00 —— 中文 bigram
    # 在 1,150 万字的语料里几乎全能命中，覆盖率区分不出「域外」和「域内」。
    # 留着这个开关是为了将来换成语料内 IDF 加权覆盖度再试，不是现在的判据。
    "refusal_retry": True,
    "refusal_retry_min_coverage": 0.0,
}

# 这些 source 的 score 不是可比的相似度，阈值对它们无效。
_THRESHOLDLESS_SOURCES = ("hybrid", "rerank")


def _keyword_coverage(question: str, hits: list[SearchHit]) -> float:
    """问题里的关键词有多大比例出现在检索到的上下文里。

    用检索同一套分词，但只保留长度 > 1 的 token：中文单字（「是」「的」「么」）
    在任何上下文里都会命中，算进去只会把这个指标抬到没意义的水平。

    **不要拿它当闸门**：实测「今天天气怎么样」「你好」在技术语料上也是 1.00，
    因为中文 bigram 在 1150 万字的语料里几乎全能命中。它现在只是日志里的一个
    诊断量，真正决定要不要复核的是 `refusal_retry_min_coverage`（默认 0 = 一律复核）。
    """
    wanted = {tok for tok in tokenize(question) if len(tok) > 1}
    if not wanted:
        return 0.0
    context = " ".join(hit.text or "" for hit in hits)
    return len(wanted & set(tokenize(context))) / len(wanted)


def _merge_usage(first: dict, second: dict) -> dict:
    """两次调用的 usage 相加。数字类字段逐项累加，其它字段保留第一次的值。"""
    merged = dict(first or {})
    for key, value in (second or {}).items():
        if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
            merged[key] = merged[key] + value
        elif key not in merged:
            merged[key] = value
    return merged


def _gap_lines(text: str) -> list[str]:
    """取出「资料未覆盖：X」这类缺口声明行。"""
    return [line.strip() for line in (text or "").splitlines()
            if line.strip().startswith(GAP_PREFIX)]


def _is_gap_only(text: str) -> bool:
    """整段回答是不是只有缺口声明（没有任何实质论断）。

    复核之后的模型有时只写一行「资料未覆盖：<整个问题>」—— 这在语义上就是拒答，
    不该算成「答出来了」。把它判成拒答，同时把缺什么留下来给用户看，
    比只回一句「没有找到足够的信息」有信息量。
    """
    if not _gap_lines(text):
        return False
    rest = "\n".join(line for line in (text or "").splitlines()
                     if not line.strip().startswith(GAP_PREFIX))
    return not split_sentences(rest)


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

        retried = False
        if INSUFFICIENT in text:
            coverage = _keyword_coverage(question, context_hits)
            gate = float(self.cfg["refusal_retry_min_coverage"] or 0.0)
            if not self.cfg["refusal_retry"] or coverage < gate:
                # 覆盖度不够 = 真的没资料，再问一次只会多花钱
                return self._refuse(question, REFUSE_INSUFFICIENT, hits, t0,
                                    usage=usage, context_chunks=len(context_hits),
                                    detail=f"关键词覆盖 {coverage:.2f} < {gate:.2f}")

            retry_text, retry_usage = self.llm.complete(
                list(messages) + [{"role": "user", "content": RETRY_AFTER_INSUFFICIENT}],
                temperature=float(self.cfg["temperature"]),
                max_tokens=int(self.cfg["max_tokens"]),
            )
            retry_text = (retry_text or "").strip()
            # 两次调用的 token 都要算进去，否则 usage 会低估一半
            usage = _merge_usage(usage, retry_usage)
            if not retry_text or INSUFFICIENT in retry_text:
                return self._refuse(question, REFUSE_INSUFFICIENT, hits, t0,
                                    usage=usage, context_chunks=len(context_hits),
                                    detail=f"复核后仍拒答（覆盖 {coverage:.2f}）")
            if _is_gap_only(retry_text):
                # 只写「资料未覆盖：X」等于拒答，但把缺口带上，用户知道缺什么
                return self._refuse(question, REFUSE_INSUFFICIENT, hits, t0,
                                    usage=usage, context_chunks=len(context_hits),
                                    detail=f"复核后只声明缺口（覆盖 {coverage:.2f}）",
                                    extra="\n".join(_gap_lines(retry_text)),
                                    refusal_retried=True)
            text = retry_text
            retried = True

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
            refusal_retried=retried,
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
                context_chunks: int = 0, extra: str = "",
                refusal_retried: bool = False) -> Answer:
        """构造拒答结果。正文固定一句人话，不做引用统计。

        `extra` 是补充说明（例如复核后模型声明了哪些子问题资料未覆盖），
        接在固定话术后面另起一行 —— 用户至少知道缺的是什么。
        """
        log.info("拒答：%s（%s）", reason, detail or "-")
        text = REFUSE_TEXT if not extra else f"{REFUSE_TEXT}\n{extra}"
        return Answer(
            question=question,
            text=text,
            citations=[],
            refused=True,
            refuse_reason=reason,
            hits=list(hits),
            model=getattr(self.llm, "model", "") or "",
            usage=dict(usage or {}),
            latency_ms=_elapsed_ms(t0),
            context_chunks=context_chunks,
            refusal_retried=refusal_retried,
        )


def _elapsed_ms(t0: float) -> int:
    return int(round((time.perf_counter() - t0) * 1000))
