"""评测体系：评测集加载、指标计算、报告渲染。

分成三层，互不依赖：
  · questions —— 读 questions.jsonl 并严格校验
  · metrics   —— 纯函数指标计算 + 跨运行对比（不碰检索层）
  · runner    —— 加载 Retriever、跑题、出报告（唯一会 import 检索层的地方）
  · report    —— 把结果渲染成能贴进 README 的文本
"""

from __future__ import annotations

from .metrics import (
    DEFAULT_KS,
    QueryOutcome,
    build_outcomes,
    diff_runs,
    first_hit_rank,
    gold_coverage_at_k,
    hit_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
    summarize,
    to_serializable,
)
from .questions import EvalDataError, Question, load_questions, parse_question
from .report import format_diff, format_report
from .runner import (
    DEFAULT_INDEX_DIR,
    RetrievalUnavailable,
    collect_results,
    load_breadcrumbs_from_corpus,
    load_retriever,
    load_run,
    run,
    save,
)

__all__ = [
    "DEFAULT_INDEX_DIR",
    "DEFAULT_KS",
    "EvalDataError",
    "QueryOutcome",
    "Question",
    "RetrievalUnavailable",
    "build_outcomes",
    "collect_results",
    "diff_runs",
    "first_hit_rank",
    "format_diff",
    "format_report",
    "gold_coverage_at_k",
    "hit_at_k",
    "load_breadcrumbs_from_corpus",
    "load_questions",
    "load_retriever",
    "load_run",
    "mean_reciprocal_rank",
    "parse_question",
    "recall_at_k",
    "reciprocal_rank",
    "run",
    "save",
    "summarize",
    "to_serializable",
]
