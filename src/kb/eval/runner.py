"""评测执行：加载检索器 → 跑题 → 出指标。

检索层（`kb.retrieve`）由另一位同学实现，接口已冻结：

    class Retriever:
        def __init__(self, index_dir, embedder=None, reranker=None, cfg=None): ...
        def search(self, query, top_k=10, *, use_dense=True, use_sparse=True,
                   use_rerank=False) -> list[SearchHit]

这里**不 import** 任何 kb.retrieve 的东西，只在真正要跑检索时才尝试导入，
导入失败就抛 RetrievalUnavailable 并给出能照着做的提示。这样即使检索层还没
写完，`python eval/run_eval.py --help` 和评测逻辑的单元测试也照常可用。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .metrics import QueryOutcome, build_outcomes, summarize, to_serializable

DEFAULT_INDEX_DIR = "data/index"


class RetrievalUnavailable(RuntimeError):
    """检索层还不可用。消息里必须写清楚缺什么、怎么办。"""


def load_retriever(index_dir, *, embedder=None, reranker=None, cfg=None, repo_root: Path | None = None):
    """按冻结的接口加载 Retriever。

    失败分三种情况给出不同提示：包不存在 / 类不存在 / 索引目录不存在。
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    src = root / "src"
    import sys

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    try:
        import kb.retrieve as retrieve_mod  # type: ignore
    except ImportError as exc:
        raise RetrievalUnavailable(
            "检索层还没就绪：import kb.retrieve 失败（"
            f"{exc}）。\n"
            f"  期望位置：{src / 'kb' / 'retrieve'}（或 kb/retrieve.py）\n"
            "  这是同事在实现的模块，等它落地并建好索引后再跑评测。\n"
            "  只想验证评测逻辑本身，请运行：python tests/test_eval.py"
        ) from exc

    retriever_cls = getattr(retrieve_mod, "Retriever", None)
    if retriever_cls is None:
        raise RetrievalUnavailable(
            f"{getattr(retrieve_mod, '__file__', 'kb.retrieve')} 里没有 Retriever 类，"
            "接口可能改了。评测脚本按冻结接口调用 Retriever(index_dir, embedder, reranker, cfg)。"
        )

    idx = Path(index_dir)
    if not idx.exists():
        raise RetrievalUnavailable(
            f"索引目录不存在：{idx}\n"
            "  索引要先建好才能跑评测。如果同事的建索引入口还没落地，"
            "先在 tests/test_eval.py 里用假 Retriever 验证评测逻辑。"
        )
    return retriever_cls(idx, embedder=embedder, reranker=reranker, cfg=cfg)


def collect_results(retriever, questions, *, top_k: int, use_dense: bool, use_sparse: bool,
                    use_rerank: bool, progress=None) -> tuple[dict, dict, float]:
    """逐条检索。返回 (results, breadcrumbs, 耗时秒)。

    results[qid] 是 SearchHit 列表；breadcrumbs 是 chunk_id -> breadcrumb 映射，
    渲染失败案例时要显示"gold 在哪个小节、实际返回了哪个小节"。
    """
    results: dict = {}
    breadcrumbs: dict[str, str] = {}
    t0 = time.perf_counter()
    total = len(questions)
    for i, q in enumerate(questions, start=1):
        hits = retriever.search(
            q.question, top_k=top_k,
            use_dense=use_dense, use_sparse=use_sparse, use_rerank=use_rerank,
        )
        results[q.qid] = list(hits)
        for h in hits:
            cid = getattr(h, "chunk_id", None) or (h.get("chunk_id") if isinstance(h, dict) else None)
            bc = getattr(h, "breadcrumb", None) or (h.get("breadcrumb") if isinstance(h, dict) else None)
            if cid and bc:
                breadcrumbs.setdefault(cid, bc)
        if progress:
            progress(i, total, q)
    return results, breadcrumbs, time.perf_counter() - t0


def load_breadcrumbs_from_corpus(corpus_path, gold_ids) -> dict[str, str]:
    """从语料 JSONL 里补 gold 的 breadcrumb（检索结果里不会带 gold 的信息）。

    只读需要的行，语料 45MB 也就几秒。找不到文件时返回空 dict，不报错 ——
    失败案例里退化成显示 chunk_id，指标不受影响。
    """
    want = set(gold_ids)
    if not want:
        return {}
    path = Path(corpus_path)
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("chunk_id")
            if cid in want:
                out[cid] = obj.get("breadcrumb") or obj.get("title") or cid
                if len(out) == len(want):
                    break
    return out


def run(questions, retriever, *, top_k: int, use_dense: bool, use_sparse: bool,
        use_rerank: bool, config: dict | None = None, extra_breadcrumbs=None,
        progress=None) -> tuple[list[QueryOutcome], dict, dict]:
    """跑完一整套评测，返回 (outcomes, summary, config)。"""
    results, breadcrumbs, elapsed = collect_results(
        retriever, questions, top_k=top_k, use_dense=use_dense,
        use_sparse=use_sparse, use_rerank=use_rerank, progress=progress,
    )
    if extra_breadcrumbs:
        for cid, bc in extra_breadcrumbs.items():
            breadcrumbs.setdefault(cid, bc)
    outcomes = build_outcomes(questions, results, breadcrumbs)
    summary = summarize(outcomes)
    full_config = dict(config or {})
    full_config.update({
        "top_k": top_k,
        "use_dense": use_dense,
        "use_sparse": use_sparse,
        "use_rerank": use_rerank,
        "n_questions": len(questions),
        "elapsed_sec": round(elapsed, 2),
    })
    return outcomes, summary, full_config


def save(path, outcomes, summary: dict, config: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = to_serializable(outcomes, summary, config)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_run(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"结果文件不存在：{p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "summary" not in data:
        raise ValueError(f"{p} 不是评测结果文件（缺 summary 字段）")
    return data
