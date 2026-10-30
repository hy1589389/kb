"""检索指标计算。

这里的函数全部是纯函数：输入是「gold chunk_id 列表」和「按名次排好的检索结果
chunk_id 列表」，不依赖 Retriever、索引或向量库。所以评测逻辑本身可以脱离
真实索引做单元测试（见 tests/test_eval.py）。

约定：
  · `predicted` 按名次从前往后排列，第 0 个是 rank 1。
  · Recall@k / Hit@k 是"top-k 里命中任意一个 gold 就算命中"。
  · MRR@k 取第一个命中项的名次倒数；一条都没命中记 0（不是 None）。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

DEFAULT_KS = (1, 3, 5, 10)
DIFFICULTY_ORDER = ("easy", "medium", "hard", "unknown")


def first_hit_rank(gold, predicted) -> int | None:
    """第一个命中 gold 的名次（从 1 开始）；没命中返回 None。"""
    gold_set = set(gold)
    for i, cid in enumerate(predicted, start=1):
        if cid in gold_set:
            return i
    return None


def hit_at_k(gold, predicted, k: int) -> bool:
    """top-k 里有没有任意一个 gold。gold 有多个时命中任意一个即算命中。"""
    return first_hit_rank(gold, predicted[:k]) is not None


def recall_at_k(gold, predicted, k: int) -> float:
    """单条问题的 Recall@k，命中返回 1.0，否则 0.0。

    这是二值口径：一条题对应一个（或多个）标准答案位置，只要 top-k 里出现
    任意一个就认为这条题被召回了。平均之后就是通常说的 Recall@k。
    """
    return 1.0 if hit_at_k(gold, predicted, k) else 0.0


def gold_coverage_at_k(gold, predicted, k: int) -> float:
    """top-k 覆盖了 gold 里多大比例（多 gold 题才和 Recall@k 有区别）。"""
    if not gold:
        return 0.0
    gold_set = set(gold)
    found = sum(1 for cid in predicted[:k] if cid in gold_set)
    return found / len(gold_set)


def reciprocal_rank(gold, predicted, k: int = 10) -> float:
    """RR@k：第一个命中项名次的倒数。没命中返回 0.0。"""
    rank = first_hit_rank(gold, predicted[:k])
    return 1.0 / rank if rank else 0.0


def mean_reciprocal_rank(pairs, k: int = 10) -> float:
    """MRR@k。pairs 是 (gold, predicted) 的可迭代对象；空输入返回 0.0。"""
    rrs = [reciprocal_rank(gold, predicted, k) for gold, predicted in pairs]
    if not rrs:
        return 0.0
    return sum(rrs) / len(rrs)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


@dataclass
class QueryOutcome:
    """单条问题的评测结果。用于渲染失败案例、按层归组和跨运行对比。"""

    qid: str
    question: str
    gold: list[str]
    predicted: list[str]
    difficulty: str = "unknown"
    category: str = "未分类"
    note: str = ""
    # chunk_id -> breadcrumb，渲染失败案例时用，缺失也能跑
    breadcrumbs: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def rank(self) -> int | None:
        return first_hit_rank(self.gold, self.predicted)

    @property
    def hit(self) -> bool:
        return self.rank is not None

    def hit_at(self, k: int) -> bool:
        return hit_at_k(self.gold, self.predicted, k)

    def rr(self, k: int = 10) -> float:
        return reciprocal_rank(self.gold, self.predicted, k)

    def gold_breadcrumbs(self) -> list[str]:
        return [self.fmt(cid) for cid in self.gold]

    def top_breadcrumbs(self, n: int = 3) -> list[str]:
        return [self.fmt(cid) for cid in self.predicted[:n]]

    def fmt(self, chunk_id: str) -> str:
        """`breadcrumb  [chunk_id]`。带上 chunk_id 是因为同一小节的多个 chunk
        breadcrumb 完全相同，只看 breadcrumb 分不清 gold 和 top-1 是不是同一块。"""
        bc = self.breadcrumbs.get(chunk_id)
        base = bc if bc and bc != chunk_id else ""
        return f"{base}  [{chunk_id}]" if base else f"[{chunk_id}]"


def build_outcomes(questions, results, breadcrumbs=None) -> list[QueryOutcome]:
    """把 Question 列表和「qid -> 检索结果」拼成 QueryOutcome 列表。

    results: dict[qid] -> list[str]，或 dict[qid] -> 带 chunk_id 属性的对象列表。
    缺失的 qid 记成 error，不静默当成"未命中"。
    """
    breadcrumbs = breadcrumbs or {}
    outcomes = []
    for q in questions:
        if q.qid not in results:
            outcomes.append(
                QueryOutcome(
                    qid=q.qid, question=q.question, gold=list(q.gold), predicted=[],
                    difficulty=q.difficulty, category=q.category, note=q.note,
                    breadcrumbs=breadcrumbs, error="检索没有返回结果",
                )
            )
            continue
        raw = results[q.qid]
        predicted = []
        for item in raw:
            if isinstance(item, str):
                predicted.append(item)
                continue
            cid = getattr(item, "chunk_id", None)
            if cid is None and isinstance(item, dict):
                cid = item.get("chunk_id")
            if cid is None:
                raise TypeError(f"检索结果元素既不是 chunk_id 字符串，也没有 chunk_id 属性：{item!r}")
            predicted.append(cid)
            bc = getattr(item, "breadcrumb", None)
            if bc is None and isinstance(item, dict):
                bc = item.get("breadcrumb")
            if bc:
                breadcrumbs.setdefault(cid, bc)
        outcomes.append(
            QueryOutcome(
                qid=q.qid, question=q.question, gold=list(q.gold), predicted=predicted,
                difficulty=q.difficulty, category=q.category, note=q.note,
                breadcrumbs=breadcrumbs,
            )
        )
    return outcomes


def _grouped(outcomes, key) -> dict[str, list[QueryOutcome]]:
    groups: dict[str, list[QueryOutcome]] = defaultdict(list)
    for o in outcomes:
        groups[getattr(o, key)].append(o)
    return dict(groups)


def _group_metrics(rows: list[QueryOutcome], ks=DEFAULT_KS, mrr_k: int = 10) -> dict:
    return {
        "n": len(rows),
        "recall": {k: _mean(o.hit_at(k) for o in rows) for k in ks},
        "mrr@{}".format(mrr_k): _mean(o.rr(mrr_k) for o in rows),
        "hit@1": _mean(o.hit_at(1) for o in rows),
        "mean_rank": _mean([o.rank for o in rows if o.rank]) if any(o.rank for o in rows) else None,
    }


def summarize(outcomes, ks=DEFAULT_KS, mrr_k: int = 10) -> dict:
    """总指标 + 按难度/题型分层 + 失败案例清单。"""
    outcomes = list(outcomes)
    summary = {
        "n": len(outcomes),
        "ks": list(ks),
        "mrr_k": mrr_k,
        "errors": [
            {"qid": o.qid, "question": o.question, "error": o.error}
            for o in outcomes if o.error
        ],
    }
    summary.update(_group_metrics(outcomes, ks, mrr_k))

    by_difficulty = {}
    for name, rows in _grouped(outcomes, "difficulty").items():
        by_difficulty[name] = _group_metrics(rows, ks, mrr_k)
    summary["by_difficulty"] = dict(
        sorted(by_difficulty.items(), key=lambda kv: (
            DIFFICULTY_ORDER.index(kv[0]) if kv[0] in DIFFICULTY_ORDER else 99, kv[0]))
    )

    summary["by_category"] = {
        name: _group_metrics(rows, ks, mrr_k)
        for name, rows in sorted(_grouped(outcomes, "category").items())
    }

    summary["failures"] = [
        {
            "qid": o.qid,
            "question": o.question,
            "difficulty": o.difficulty,
            "category": o.category,
            "note": o.note,
            "gold": o.gold,
            "gold_breadcrumbs": o.gold_breadcrumbs(),
            "top_breadcrumbs": o.top_breadcrumbs(3),
            "top_chunk_ids": o.predicted[:3],
            "error": o.error,
        }
        for o in outcomes if not o.hit
    ]
    summary["n_failures"] = len(summary["failures"])
    return summary


# --------------------------------------------------------------------------
# 跨运行对比
# --------------------------------------------------------------------------

def to_serializable(outcomes, summary: dict, config: dict) -> dict:
    """评测结果落盘格式。--json / --out 共用，便于 A/B 对比。"""
    return {
        "config": config,
        "summary": summary,
        "per_query": [
            {
                "qid": o.qid,
                "question": o.question,
                "difficulty": o.difficulty,
                "category": o.category,
                "gold": o.gold,
                "predicted": o.predicted,
                "rank": o.rank,
                "rr": round(o.rr(summary.get("mrr_k", 10)), 6),
                "error": o.error,
            }
            for o in outcomes
        ],
    }


_METRIC_KEYS = ("recall", "mrr", "hit@1", "mean_rank", "n")


def _flat_metrics(block: dict) -> dict[str, float | None]:
    flat: dict[str, float | None] = {}
    for key, val in block.items():
        if key == "recall" and isinstance(val, dict):
            for k, v in val.items():
                flat[f"recall@{k}"] = v
        elif key.startswith("mrr@"):
            flat[key] = val
        elif key in ("hit@1", "mean_rank", "n"):
            flat[key] = val
    return flat


def diff_runs(run_a: dict, run_b: dict) -> dict:
    """对比两次运行。

    返回 {'overall': [...], 'by_difficulty': {...}, 'improved': [...],
          'regressed': [...], 'both_failed': [...]}，每行含 a / b / delta / direction。

    delta = B - A；direction 为 'up' / 'down' / 'same'（指标越大越好）。
    注意 mean_rank 是"越小越好"，direction 对它做了反向处理。
    """
    a_sum = run_a.get("summary", run_a)
    b_sum = run_b.get("summary", run_b)
    lower_is_better = {"mean_rank"}

    def rows(block_a: dict, block_b: dict) -> list[dict]:
        fa, fb = _flat_metrics(block_a), _flat_metrics(block_b)
        out = []
        for key in sorted(set(fa) | set(fb)):
            va, vb = fa.get(key), fb.get(key)
            delta = None if va is None or vb is None else vb - va
            if delta is None or abs(delta) < 1e-12:
                direction = "same"
            elif key in lower_is_better:
                direction = "up" if delta < 0 else "down"
            else:
                direction = "up" if delta > 0 else "down"
            out.append({"metric": key, "a": va, "b": vb, "delta": delta, "direction": direction})
        return out

    a_q = {r["qid"]: r for r in run_a.get("per_query", [])}
    b_q = {r["qid"]: r for r in run_b.get("per_query", [])}
    improved, regressed, both_failed = [], [], []
    for qid in sorted(set(a_q) | set(b_q)):
        ra, rb = a_q.get(qid), b_q.get(qid)
        if ra is None or rb is None:
            continue
        ha, hb = ra.get("rank") is not None, rb.get("rank") is not None
        if ha and not hb:
            regressed.append({"qid": qid, "question": rb.get("question", ""),
                              "rank_a": ra.get("rank"), "rank_b": rb.get("rank")})
        elif hb and not ha:
            improved.append({"qid": qid, "question": rb.get("question", ""),
                             "rank_a": ra.get("rank"), "rank_b": rb.get("rank")})
        elif not ha and not hb:
            both_failed.append({"qid": qid, "question": rb.get("question", "")})

    by_diff = {}
    for name in sorted(set(a_sum.get("by_difficulty", {})) | set(b_sum.get("by_difficulty", {}))):
        by_diff[name] = rows(a_sum.get("by_difficulty", {}).get(name, {}),
                             b_sum.get("by_difficulty", {}).get(name, {}))
    return {
        "overall": rows(a_sum, b_sum),
        "by_difficulty": by_diff,
        "improved": improved,
        "regressed": regressed,
        "both_failed": both_failed,
    }
