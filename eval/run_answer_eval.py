#!/usr/bin/env python
"""答案侧评测入口：检索 + 生成 +（可选）逐句忠实度判定。

用法：

    # 跑全部题，带判定，存结果
    python eval/run_answer_eval.py --out results/run_answer.json

    # 先小样本试水（每题 ~2 次模型调用，先用 3 条看看通不通）
    python eval/run_answer_eval.py --limit 3 --no-judge

    # 只看拒答和引用（不花判定 token）
    python eval/run_answer_eval.py --no-judge --out results/run_answer_nojudge.json

    # 换集合
    python eval/run_answer_eval.py --collections tech --limit 5

和检索评测（`eval/run_eval.py`）的分工：那边算 Recall/MRR，这边算拒答率、
假拒答率、无引用句占比、无效引用、逐句支持率。两边结果都存成
`config / summary / per_query` 三个顶层键，方便对照。
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
    EvalDataError,
    RetrievalUnavailable,
    load_questions,
    load_retriever,
)
from kb.eval.answer import format_answer_report, run, save_answer_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_answer_eval.py",
        description="答案侧评测：拒答率 / 假拒答 / 引用质量 / 逐句忠实度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"),
                   help="评测集 JSONL（默认 eval/questions.jsonl）")
    p.add_argument("--index-dir", default=DEFAULT_INDEX_DIR,
                   help=f"索引目录（默认 {DEFAULT_INDEX_DIR}）")
    p.add_argument("--cfg", default="",
                   help="传给 Retriever 的 JSON 覆盖，默认读 config.toml 的 [retrieval]")
    p.add_argument("--collections", default="",
                   help="只查这些集合（逗号分隔），默认全部")
    p.add_argument("--top-k", type=int, default=0,
                   help="送进提示词的块数，0 = 用 config.toml 里 [generate].top_k")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    p.add_argument("--no-judge", action="store_true",
                   help="不做逐句判定（省一半 token，只出拒答/引用指标）")
    p.add_argument("--json", action="store_true", help="结果按 JSON 打到 stdout")
    p.add_argument("--out", default="", help="把结果存成 JSON")
    p.add_argument("--failures", type=int, default=5,
                   help="无依据句子清单最多显示几题（默认 5）")
    return p


def _cfg_from_toml() -> dict | None:
    """从 config.toml 的 [retrieval] 段取检索配置，相对路径按项目根展开。"""
    cfg_path = ROOT / "config.toml"
    if not cfg_path.is_file():
        return None
    try:
        from kb.config import IngestConfig

        out = IngestConfig.load(cfg_path).retrieval.to_retriever_cfg()
        for key in ("model_dir", "reranker_dir"):
            val = out.get(key)
            if isinstance(val, Path) and not val.is_absolute():
                out[key] = (ROOT / val).resolve()
        return out
    except Exception:  # 配置文件坏了退回内置默认值，不挡住评测
        return None


def _load_clients():
    """建 DeepSeek 客户端 + Answerer 配置。没有 key 时抛 KeyNotFoundError。"""
    from kb.config import GenerateConfig
    from kb.generate.llm import DeepSeekClient

    gc = GenerateConfig.load(ROOT / "config.toml")
    client = DeepSeekClient(
        model=gc.model, base_url=gc.base_url,
        api_key_env=gc.api_key_env, timeout=gc.timeout,
    )
    return gc, client


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)

    try:
        questions = load_questions(args.questions)
    except EvalDataError as exc:
        print(f"[X] 评测集有问题：{exc}", file=sys.stderr)
        return 2
    if args.limit > 0:
        questions = questions[:args.limit]

    try:
        cfg = json.loads(args.cfg) if args.cfg else _cfg_from_toml()
    except json.JSONDecodeError as exc:
        print(f"[X] --cfg 不是合法 JSON：{exc}", file=sys.stderr)
        return 2

    from kb.generate.llm import KeyNotFoundError, LLMError
    from kb.generate.pipeline import Answerer

    try:
        gc, client = _load_clients()
    except KeyNotFoundError as exc:
        print(f"[X] 找不到 API key：{exc}", file=sys.stderr)
        return 4
    except LLMError as exc:
        print(f"[X] 客户端初始化失败：{exc}", file=sys.stderr)
        return 4

    try:
        retriever = load_retriever(args.index_dir, cfg=cfg, repo_root=ROOT)
    except RetrievalUnavailable as exc:
        print(f"[X] 无法开始评测。\n\n{exc}", file=sys.stderr)
        return 3

    top_k = args.top_k or int(gc.top_k)
    collections = [c.strip() for c in args.collections.split(",") if c.strip()] or None
    answerer = Answerer(retriever, client, cfg=gc.to_answerer_cfg())

    def progress(index: int, total: int, question) -> None:
        print(f"\r  评测中 {index}/{total} …", end="", file=sys.stderr, flush=True)

    if not args.json:
        print(f"  模型 {gc.model}   判定 {'关' if args.no_judge else '开'}   "
              f"top_k={top_k}", file=sys.stderr)

    try:
        outcomes, summary, meta = run(
            questions, answerer, top_k=top_k,
            judge_llm=None if args.no_judge else client,
            collections=collections, progress=progress,
            config={"questions": str(args.questions), "index_dir": str(args.index_dir),
                    "model": gc.model},
        )
    except LLMError as exc:
        print(f"\n[X] 模型调用失败：{exc}", file=sys.stderr)
        return 4
    except Exception as exc:  # noqa: BLE001 —— 检索/生成的问题原样报出来
        print(f"\n[X] 评测出错（{type(exc).__name__}: {exc}）", file=sys.stderr)
        return 3
    if not args.json:
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

    if args.out:
        path = save_answer_run(args.out, outcomes, summary, meta)
        if not args.json:
            print(f"结果已保存 -> {path}", file=sys.stderr)

    if args.json:
        print(json.dumps({"config": meta, "summary": summary,
                          "per_query": [o.to_dict() for o in outcomes]},
                         ensure_ascii=False, indent=2))
        return 0

    print("=" * 72)
    print("  答案侧评测 · 拒答 / 引用 / 忠实度")
    print("=" * 72)
    print(format_answer_report(outcomes, summary, meta,
                               failure_limit=args.failures or None))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
