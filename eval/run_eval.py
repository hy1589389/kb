#!/usr/bin/env python
"""检索评测入口。

用法：
    # 跑一次评测（需要同事的检索层 + 建好的索引）
    python eval/run_eval.py --index-dir data/index --top-k 10

    # 只跑前 5 条，顺手存结果
    python eval/run_eval.py --limit 5 --out results/run_baseline.json

    # 只开稠密 / 只开稀疏 / 加重排，做消融
    python eval/run_eval.py --no-sparse --out results/run_dense_only.json
    python eval/run_eval.py --no-dense  --out results/run_sparse_only.json
    python eval/run_eval.py --rerank    --out results/run_rerank.json

    # 对比两次运行
    python eval/run_eval.py --compare results/run_baseline.json results/run_rerank.json

参数说明见 --help。输出是纯文本，可以直接贴进 README。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb.eval import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    DEFAULT_KS,
    EvalDataError,
    RetrievalUnavailable,
    diff_runs,
    format_diff,
    format_report,
    load_breadcrumbs_from_corpus,
    load_questions,
    load_retriever,
    load_run,
    run,
    save,
)

# 语料 JSONL 的常见位置，找得到就用它补 gold 的 breadcrumb
CORPUS_CANDIDATES = (
    Path(r"G:\kb\data\processed\chunks_all.jsonl"),
    ROOT / "data" / "processed" / "chunks_all.jsonl",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval.py",
        description="本地知识库检索评测：Recall@k / MRR@10 / Hit@1 + 分层指标 + 失败案例清单",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"),
                   help="评测集 JSONL（默认 eval/questions.jsonl）")
    p.add_argument("--index-dir", default=DEFAULT_INDEX_DIR,
                   help=f"索引目录（默认 {DEFAULT_INDEX_DIR}）")
    p.add_argument("--corpus", default="",
                   help="语料 JSONL，用来补 gold 的 breadcrumb（默认自动找）")
    p.add_argument("--cfg", default="",
                   help='传给 Retriever 的配置，JSON，如 \'{"model_dir": "G:/kb/models/bge-m3"}\'')
    p.add_argument("--top-k", type=int, default=10, help="检索条数（默认 10）")
    p.add_argument("--no-dense", action="store_true", help="关闭稠密检索")
    p.add_argument("--no-sparse", action="store_true", help="关闭稀疏检索")
    p.add_argument("--rerank", action="store_true", help="开启重排")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    p.add_argument("--json", action="store_true", help="把结果按 JSON 打到 stdout")
    p.add_argument("--out", default="", help="把结果存成 JSON，便于 --compare")
    p.add_argument("--failures", type=int, default=0,
                   help="失败案例清单最多显示几条（默认全部；0 = 全部）")
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                   help="不跑检索，只对比两次已保存的结果")
    return p


def _find_corpus(explicit: str) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for cand in CORPUS_CANDIDATES:
        if cand.is_file():
            return cand
    return None


def _cfg_from_toml() -> dict | None:
    """没给 --cfg 时，从项目根的 config.toml 读 [retrieval] 段。

    这条路是为了省掉在命令行上手写 JSON —— Windows PowerShell 5.1 会把
    参数里的内层双引号吃掉，`--cfg '{"model_dir": "..."}'` 传进去会变成
    非法 JSON，必须写成 `'{\\"model_dir\\": \\"...\\"}'` 才行，没人愿意记这个。
    所以默认走配置文件，`--cfg` 只作为临时覆盖。
    """
    cfg_path = ROOT / "config.toml"
    if not cfg_path.is_file():
        return None
    try:
        import tomllib

        from kb.config import IngestConfig

        parsed = IngestConfig.load(cfg_path)
        out = parsed.retrieval.to_retriever_cfg()
        # 相对路径按项目根展开，否则从别的工作目录调用就会找不到模型
        for key in ("model_dir", "reranker_dir"):
            val = out.get(key)
            if isinstance(val, Path) and not val.is_absolute():
                out[key] = (ROOT / val).resolve()
        return out
    except Exception:  # 配置文件坏了不该让评测挂掉，退回内置默认值
        return None


def cmd_compare(args: argparse.Namespace) -> int:
    path_a, path_b = args.compare
    try:
        run_a = load_run(path_a)
        run_b = load_run(path_b)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[X] {exc}", file=sys.stderr)
        return 2
    cfg_a = run_a.get("config", {})
    cfg_b = run_b.get("config", {})

    def switches(cfg: dict) -> str:
        return (f"dense={cfg.get('use_dense')} sparse={cfg.get('use_sparse')} "
                f"rerank={cfg.get('use_rerank')} top_k={cfg.get('top_k')}")

    print("=" * 72)
    print("  检索评测 · 运行对比")
    print("=" * 72)
    print(f"  A = {path_a}")
    print(f"      {switches(cfg_a)}")
    print(f"  B = {path_b}")
    print(f"      {switches(cfg_b)}")
    print("-" * 72)
    print(format_diff(diff_runs(run_a, run_b), "A", "B"))
    print("=" * 72)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 cp936，中文和 ↑↓ 会炸，这里强制 UTF-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    if args.compare:
        return cmd_compare(args)

    try:
        questions = load_questions(args.questions)
    except EvalDataError as exc:
        print(f"[X] 评测集有问题：{exc}", file=sys.stderr)
        return 2

    total_questions = len(questions)
    if args.limit and args.limit > 0:
        questions = questions[:args.limit]

    use_dense = not args.no_dense
    use_sparse = not args.no_sparse
    if not use_dense and not use_sparse:
        print("[X] --no-dense 和 --no-sparse 不能同时给，那样没有任何召回通道。", file=sys.stderr)
        return 2

    try:
        cfg = json.loads(args.cfg) if args.cfg else _cfg_from_toml()
    except json.JSONDecodeError as exc:
        print(f"[X] --cfg 不是合法 JSON：{exc}", file=sys.stderr)
        print(f"    提示：在 Windows PowerShell 里内层引号要转义，"
              f"写成 --cfg '{{\\\"model_dir\\\": \\\"...\\\"}}'；"
              f"或者干脆别传 --cfg，直接把参数写进 config.toml 的 [retrieval] 段。",
              file=sys.stderr)
        return 2
    if cfg is not None and not isinstance(cfg, dict):
        print("[X] --cfg 必须是一个 JSON 对象，例如 '{\"model_dir\": \"...\"}'", file=sys.stderr)
        return 2

    try:
        retriever = load_retriever(args.index_dir, cfg=cfg, repo_root=ROOT)
    except RetrievalUnavailable as exc:
        print(f"[X] 无法开始评测。\n\n{exc}", file=sys.stderr)
        return 3

    gold_ids = {cid for q in questions for cid in q.gold}
    corpus = _find_corpus(args.corpus)
    gold_breadcrumbs = load_breadcrumbs_from_corpus(corpus, gold_ids) if corpus else {}
    if corpus and not gold_breadcrumbs:
        print(f"[!] 从 {corpus} 里没读到任何 gold 的 breadcrumb，失败案例将显示 chunk_id。",
              file=sys.stderr)

    def progress(i: int, total: int, q) -> None:
        if not args.json:
            print(f"\r  检索中 {i}/{total} …", end="", file=sys.stderr, flush=True)

    try:
        outcomes, summary, config = run(
            questions, retriever,
            top_k=args.top_k, use_dense=use_dense, use_sparse=use_sparse,
            use_rerank=args.rerank,
            config={"questions": str(args.questions), "index_dir": str(args.index_dir),
                    "n_questions_total": total_questions},
            extra_breadcrumbs=gold_breadcrumbs,
            progress=progress,
        )
    except NotImplementedError as exc:
        print(f"\n[X] 检索层调用了未实现的方法：{exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 —— 检索层的问题要原样报出来，不要吞
        print(f"\n[X] 检索过程出错（{type(exc).__name__}: {exc}）", file=sys.stderr)
        return 3
    if not args.json:
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

    if args.out:
        path = save(args.out, outcomes, summary, config)
        if not args.json:
            print(f"结果已保存 -> {path}", file=sys.stderr)

    if args.json:
        from kb.eval import to_serializable
        print(json.dumps(to_serializable(outcomes, summary, config), ensure_ascii=False, indent=2))
        return 0

    print("=" * 72)
    print("  检索评测 · Phase 3")
    print("=" * 72)
    print(format_report(outcomes, summary, config,
                        failure_limit=args.failures or None, ks=DEFAULT_KS))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
