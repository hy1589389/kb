"""kb 命令行入口：出问题只看这一个文件就能定位流程。

子命令：
    ingest   遍历语料目录，切块并输出 JSONL + 统计      （Phase 1）
    index    建稠密 + 稀疏检索索引                        （Phase 2）
    search   查询索引，人类可读或 --json 输出              （Phase 2）
    ask      检索 + 大模型生成带引用的答案                 （Phase 4）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import IngestConfig
from .ingest.pipeline import build_stats, run_ingest, write_outputs

# src/kb/cli.py -> src/kb -> src -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"


def _fmt(n: int) -> str:
    return f"{n:,}"


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = IngestConfig.load(Path(args.config) if args.config else None)

    if args.raw:
        cfg.raw_dir = Path(args.raw)
    if args.out:
        cfg.out_dir = Path(args.out)
    if args.max_chars is not None:
        cfg.chunk.max_chars = args.max_chars
    if args.min_chars is not None:
        cfg.chunk.min_chars = args.min_chars
    if args.overlap is not None:
        cfg.chunk.overlap = args.overlap
    cfg.resolve(PROJECT_ROOT)

    print("=" * 68)
    print("  Phase 1 · 语料入库")
    print("=" * 68)
    print(f"  原始语料 : {cfg.raw_dir}")
    print(f"  输出目录 : {cfg.out_dir}")
    print(f"  切块参数 : max={cfg.chunk.max_chars} min={cfg.chunk.min_chars} "
          f"overlap={cfg.chunk.overlap} drop_below={cfg.chunk.drop_below}")
    print("-" * 68)

    if not cfg.raw_dir.exists():
        print(f"[X] 原始语料目录不存在：{cfg.raw_dir}")
        print("    先运行:  powershell -File scripts\\bootstrap.ps1")
        return 2

    import time

    t0 = time.perf_counter()
    result = run_ingest(cfg)
    elapsed = time.perf_counter() - t0

    if not result.docs:
        print("[X] 没有解析出任何文档，请检查语料目录。")
        return 1

    stats = build_stats(result, cfg, elapsed)
    written = write_outputs(result, cfg, stats)

    t = stats["totals"]
    cs = stats["chunk_size"]
    print("-" * 68)
    print("  入库结果")
    print("-" * 68)
    print(f"  仓库        : {t['repos']}")
    print(f"  文档        : {_fmt(t['docs'])} 篇")
    print(f"  切块        : {_fmt(t['chunks'])} 个")
    print(f"  总字符      : {_fmt(t['chars'])}  （其中中文 {_fmt(t['cjk_chars'])}）")
    print(f"  估算 token  : {_fmt(t['est_tokens'])}")
    print(f"  块长分布    : min {cs['min']} / p50 {cs['p50']} / p90 {cs['p90']} "
          f"/ p95 {cs['p95']} / max {cs['max']}")
    print(f"  耗时        : {stats['elapsed_sec']}s")
    if stats["skipped_summary"]:
        print(f"  跳过        : {stats['skipped_summary']}")
    print("-" * 68)
    print(f"  stats     -> {written['stats']}")
    print(f"  全量 JSONL-> {written['chunks_all']}")
    print(f"  文档清单  -> {written['manifest']}")
    print("=" * 68)
    return 0


def _load_retrieval_config(args: argparse.Namespace):
    """读 config.toml 的 [retrieval] 段，再被命令行参数覆盖。

    命令行优先级高于配置文件，配置文件的优先级高于代码里的默认值 ——
    这样「跑一次实验」不需要改文件，「长期设置」不需要每次敲参数。
    """
    cfg_path = Path(args.config) if getattr(args, "config", None) else DEFAULT_CONFIG
    ingest_cfg = IngestConfig.load(cfg_path)
    rc = ingest_cfg.retrieval
    rc.resolve(PROJECT_ROOT)

    if getattr(args, "index_dir", None):
        rc.index_dir = Path(args.index_dir).resolve()
    if getattr(args, "model_dir", None):
        rc.model_dir = Path(args.model_dir)
    if getattr(args, "reranker_dir", None):
        rc.reranker_dir = Path(args.reranker_dir)
    if getattr(args, "batch_size", None):
        rc.embed_batch_size = int(args.batch_size)
    if getattr(args, "max_length", None):
        rc.embed_max_length = int(args.max_length)
    return rc


def cmd_index(args: argparse.Namespace) -> int:
    from .retrieve.pipeline import _make_embedder, build_index

    rc = _load_retrieval_config(args)
    chunks_path = Path(args.chunks).resolve() if args.chunks else (
        PROJECT_ROOT / "data" / "processed" / "chunks_all.jsonl"
    )
    index_dir = rc.index_dir

    print("=" * 68)
    print("  Phase 2 · 建检索索引")
    print("=" * 68)
    print(f"  语料     : {chunks_path}")
    print(f"  索引目录 : {index_dir}")
    print(f"  embedder : {args.embedder}"
          + (f"  (model_dir={rc.model_dir})" if args.embedder == "onnx" else ""))
    print(f"  批大小   : {rc.embed_batch_size}   截断长度: {rc.embed_max_length}"
          + (f"   limit={args.limit}" if args.limit else ""))
    print("-" * 68)

    try:
        if args.embedder == "hashing":
            # HashingEmbedder 的 idf 要从语料现算，所以必须先读一遍文本。
            # 用 --limit 时只读前 N 条，否则 44 MB 全读进来只为统计 df。
            # 建索引那一步会再读一遍同样的文件（idf 必须先于编码算完，
            # 顺序无法合并），多读一次约 1 秒，可接受。
            from .retrieve.pipeline import load_chunks

            chunks = load_chunks(chunks_path, limit=args.limit)
            embedder = _make_embedder(
                "hashing", None, rc.embed_max_length, rc.embed_batch_size,
                texts_for_idf=[c.get("embed_text") or c.get("text") or "" for c in chunks],
            )
            del chunks
        else:
            embedder = _make_embedder("onnx", rc.model_dir, rc.embed_max_length,
                                      rc.embed_batch_size)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[X] {exc}")
        return 2

    # hashing 且限量的场景下，build_index 也要限量：把前 N 条写成一个临时
    # JSONL，避免为了自检把 18k 条全嵌一遍（也无从「只嵌前 N 条」）。
    # 临时文件放在索引目录内，结束就删，不会污染语料目录。
    effective_chunks = chunks_path
    tmp_path = None
    if args.limit:
        from .retrieve.pipeline import write_limited_chunks

        tmp_path = index_dir / "_limit_chunks.jsonl"
        written = write_limited_chunks(chunks_path, tmp_path, args.limit)
        effective_chunks = tmp_path
        print(f"  自检模式 : 只处理前 {written:,} 条")

    try:
        info = build_index(
            effective_chunks, index_dir, embedder,
            batch_size=rc.embed_batch_size,
            max_length=rc.embed_max_length,
            progress=True,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[X] 建索引失败：{exc}")
        return 2
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    print("-" * 68)
    print(f"  块数        : {info['n_chunks']:,}")
    print(f"  嵌入维度    : {info['dim']}")
    print(f"  词表        : {info['vocab_size']:,} term / {info['n_postings']:,} posting")
    print(f"  平均块长    : {info['avgdl']} token")
    print(f"  耗时        : {info['elapsed_sec']}s")
    print(f"  索引目录    : {index_dir}")
    print("-" * 68)
    print(f"  试查        : python ingest.py search \"三次握手\" --index-dir \"{index_dir}\"")
    print("=" * 68)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .retrieve.embedder import HashingEmbedder
    from .retrieve.pipeline import Retriever

    rc = _load_retrieval_config(args)
    index_dir = rc.index_dir
    if not index_dir.exists():
        print(f"[X] 索引目录不存在：{index_dir}")
        print(f"    先建索引：python ingest.py index --chunks <chunks.jsonl>")
        return 2

    cfg = rc.to_retriever_cfg()
    if args.use_rerank:
        cfg["use_reranker"] = True
    # 命令行 --rerank 优先于配置；配置里 use_reranker=false 时不重排
    want_rerank = bool(args.use_rerank or rc.use_reranker)

    # 判断索引是哪种 embedder 建的：hashing 索引不能用 onnx 查询（向量空间
    # 不同），反之亦然。Retriever 里也有一道检查，这里提前决定省一次加载。
    try:
        index_info = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(f"[X] {index_dir / 'index_meta.json'} 读不出来，索引可能不完整，请重建。")
        return 2

    embedder = None
    if "hashing" in str(index_info.get("embedder", "")).lower():
        embedder = HashingEmbedder(dim=int(index_info.get("dim", 512)))

    try:
        retriever = Retriever(index_dir, embedder=embedder, cfg=cfg)
        hits = retriever.search(
            args.query, top_k=args.top_k,
            use_dense=not args.no_dense,
            use_sparse=not args.no_sparse,
            use_rerank=want_rerank,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[X] {exc}")
        return 2

    if args.json:
        payload = {
            "query": args.query,
            "index_dir": str(index_dir),
            "n_hits": len(hits),
            "hits": [h.to_dict() for h in hits],
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return 0

    if not hits:
        print(f"没有命中：{args.query}")
        print("（检查 --no-dense / --no-sparse 是否把两路都关了，或者换个查询词）")
        return 0

    print(f"查询: {args.query}")
    print(f"索引: {index_dir}   返回 {len(hits)} 条   "
          f"模式: {'hybrid' if not args.no_dense and not args.no_sparse else ('dense' if not args.no_dense else 'sparse')}"
          f"{' + rerank' if want_rerank else ''}")
    print("=" * 68)
    for hit in hits:
        print(f"[{hit.rank:>2}] {hit.score:>8.4f}  {hit.breadcrumb or '(无标题)'}")
        print(f"     {hit.location()}")
        print(f"     {_summarize(hit.text)}")
        print("-" * 68)
    return 0


def _summarize(text: str, width: int = 160) -> str:
    """把正文压成单行摘要：去掉换行和多余空白，超长截断加省略号。"""
    flat = " ".join((text or "").split())
    if len(flat) <= width:
        return flat
    return flat[:width] + f"... [共 {len(text)} 字符]"


def _load_generate_config(args: argparse.Namespace):
    """读 config.toml 的 [generate] 段，再被命令行参数覆盖。

    和 `_load_retrieval_config` 同一套优先级：命令行 > 配置文件 > 代码默认。
    """
    cfg_path = Path(args.config) if getattr(args, "config", None) else DEFAULT_CONFIG
    gc = IngestConfig.load(cfg_path).generate
    if getattr(args, "model", None):
        gc.model = args.model
    if getattr(args, "temperature", None) is not None:
        gc.temperature = float(args.temperature)
    if getattr(args, "max_context_chars", None):
        gc.max_context_chars = int(args.max_context_chars)
    if getattr(args, "top_k", None):
        gc.top_k = int(args.top_k)
    return gc


def cmd_ask(args: argparse.Namespace) -> int:
    """检索 -> 大模型 -> 带引用的答案。

    输出里不打印 key，`DeepSeekClient` 的异常消息也经过脱敏；找不到 key 时
    只提示设置方法。
    """
    from .generate.llm import DeepSeekClient, KeyNotFoundError, LLMError
    from .generate.pipeline import Answerer
    from .retrieve.embedder import HashingEmbedder
    from .retrieve.pipeline import Retriever

    rc = _load_retrieval_config(args)
    gc = _load_generate_config(args)
    index_dir = rc.index_dir
    if not index_dir.exists():
        print(f"[X] 索引目录不存在：{index_dir}")
        print("    先建索引：python ingest.py index --chunks <chunks.jsonl>")
        return 2

    try:
        index_info = json.loads(
            (index_dir / "index_meta.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        print(f"[X] {index_dir / 'index_meta.json'} 读不出来，索引可能不完整，请重建。")
        return 2

    embedder = None
    if "hashing" in str(index_info.get("embedder", "")).lower():
        embedder = HashingEmbedder(dim=int(index_info.get("dim", 512)))

    # --no-rerank 优先：配置里开了 use_reranker 也能被它关掉
    want_rerank = (bool(args.use_rerank) or bool(rc.use_reranker)) \
        and not args.no_rerank

    try:
        retriever = Retriever(index_dir, embedder=embedder,
                              cfg=rc.to_retriever_cfg())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[X] {exc}")
        return 2

    try:
        llm = DeepSeekClient(
            model=gc.model,
            base_url=gc.base_url,
            api_key_env=gc.api_key_env,
            timeout=gc.timeout,
        )
    except KeyNotFoundError as exc:
        print(f"[X] {exc}")
        return 3
    except LLMError as exc:
        print(f"[X] 初始化模型客户端失败：{exc}")
        return 2

    try:
        answerer = Answerer(retriever, llm, cfg=gc.to_answerer_cfg())
        ans = answerer.answer(args.question, top_k=gc.top_k,
                              use_rerank=want_rerank)
    except LLMError as exc:
        # exc 的消息里不含请求头；这里也只打印消息本身
        print(f"[X] 调用模型失败：{exc}")
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        # embedder / reranker 是懒加载的，缺模型文件会在第一次检索时才炸
        print(f"[X] 检索失败：{exc}")
        if "模型目录不存在" in str(exc):
            print("    模型不在配置的路径下时，用 --model-dir 指向实际目录。")
        return 2
    finally:
        llm.close()

    if args.json:
        sys.stdout.write(
            json.dumps(ans.to_dict(), ensure_ascii=False, indent=2) + "\n"
        )
        return 0

    mode = "hybrid" + (" + rerank" if want_rerank else "")
    print(f"问题: {ans.question}")
    print(f"索引: {index_dir}")
    print(f"模型: {ans.model}   检索模式: {mode}   "
          f"上下文块数: {ans.context_chunks}/{len(ans.hits)}   "
          f"耗时: {ans.latency_ms} ms")
    if ans.usage:
        print(f"token: prompt={ans.usage.get('prompt_tokens', '-')} "
              f"completion={ans.usage.get('completion_tokens', '-')} "
              f"total={ans.usage.get('total_tokens', '-')}")
    print("=" * 68)
    print(ans.render())
    print("=" * 68)

    if args.show_hits:
        print(f"检索原始结果（{len(ans.hits)} 条）：")
        for hit in ans.hits:
            print(f"[{hit.rank:>2}] {hit.score:>8.4f}  {hit.source:<7} "
                  f"{hit.breadcrumb or '(无标题)'}")
            print(f"     {hit.location()}")
            print(f"     {_summarize(hit.text)}")
            print("-" * 68)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kb",
        description="本地知识库 —— Phase 1 语料入库 / Phase 2 检索",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="遍历语料目录，切块并输出 JSONL + 统计")
    p_ingest.add_argument("--config", help="config.toml 路径（默认项目根目录下）")
    p_ingest.add_argument("--raw", help="原始语料目录")
    p_ingest.add_argument("--out", help="输出目录")
    p_ingest.add_argument("--max-chars", type=int, dest="max_chars", help="单块字符上限")
    p_ingest.add_argument("--min-chars", type=int, dest="min_chars", help="合并阈值")
    p_ingest.add_argument("--overlap", type=int, help="二次切分的重叠字符数")
    p_ingest.set_defaults(func=cmd_ingest)

    p_index = sub.add_parser("index", help="建检索索引（稠密 + BM25）")
    p_index.add_argument("--config", help="config.toml 路径（默认项目根目录下）")
    p_index.add_argument("--chunks", help="chunks_all.jsonl 路径")
    p_index.add_argument("--index-dir", dest="index_dir", help="索引输出目录")
    p_index.add_argument("--model-dir", dest="model_dir", help="bge-m3 ONNX 目录")
    p_index.add_argument("--embedder", choices=["onnx", "hashing"], default="onnx",
                         help="onnx=bge-m3（需要模型）；hashing=零依赖自检用")
    p_index.add_argument("--batch-size", type=int, dest="batch_size", help="嵌入批大小")
    p_index.add_argument("--max-length", type=int, dest="max_length",
                         help="嵌入截断长度（默认 512）")
    p_index.add_argument("--limit", type=int,
                         help="只处理前 N 条，用于快速自检")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="查询索引")
    p_search.add_argument("query", help="查询词")
    p_search.add_argument("--config", help="config.toml 路径（默认项目根目录下）")
    p_search.add_argument("--index-dir", dest="index_dir", help="索引目录")
    p_search.add_argument("--model-dir", dest="model_dir", help="bge-m3 ONNX 目录")
    p_search.add_argument("--reranker-dir", dest="reranker_dir", help="重排模型目录")
    p_search.add_argument("--top-k", type=int, default=10, dest="top_k",
                          help="返回条数（默认 10）")
    p_search.add_argument("--no-dense", action="store_true", help="关闭稠密检索")
    p_search.add_argument("--no-sparse", action="store_true", help="关闭 BM25 检索")
    p_search.add_argument("--rerank", action="store_true", dest="use_rerank",
                          help="开启交叉编码重排")
    p_search.add_argument("--json", action="store_true", help="输出 JSON")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="检索 + 大模型生成带引用的答案（Phase 4）")
    p_ask.add_argument("question", help="问题")
    p_ask.add_argument("--config", help="config.toml 路径（默认项目根目录下）")
    p_ask.add_argument("--index-dir", dest="index_dir", help="索引目录")
    p_ask.add_argument("--model-dir", dest="model_dir", help="bge-m3 ONNX 目录")
    p_ask.add_argument("--reranker-dir", dest="reranker_dir", help="重排模型目录")
    p_ask.add_argument("--top-k", type=int, dest="top_k",
                       help="送进提示词的块数（默认取 [generate].top_k）")
    p_ask.add_argument("--model", help="模型名（默认取 [generate].model）")
    p_ask.add_argument("--temperature", type=float,
                       help="采样温度（默认取 [generate].temperature）")
    p_ask.add_argument("--max-context-chars", type=int, dest="max_context_chars",
                       help="上下文总字符上限（默认取 [generate].max_context_chars）")
    p_ask.add_argument("--rerank", action="store_true", dest="use_rerank",
                       help="强制开启重排")
    p_ask.add_argument("--no-rerank", action="store_true", dest="no_rerank",
                       help="关闭重排（优先于配置和 --rerank）")
    p_ask.add_argument("--json", action="store_true", help="输出 Answer.to_dict()")
    p_ask.add_argument("--show-hits", action="store_true", dest="show_hits",
                       help="额外打印检索原始结果")
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 cp936，中文输出会炸，这里强制 UTF-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
