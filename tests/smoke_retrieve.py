"""真实语料自检：用 HashingEmbedder 在受限语料上走通整条检索链路。

存在的意义：bge-m3 的 ONNX 权重还在下载（2 GB），在它到位之前需要一条
不依赖模型、不依赖网络的路径，把「读 JSONL -> 建 BM25 + 稠密索引 -> 查询
-> RRF 融合 -> 输出 SearchHit」整条链路跑一遍。跑通了说明检索层的代码没
问题，之后换 OnnxEmbedder 只需要改一个参数。

它**不是**效果评测：HashingEmbedder 只做「共词相似度」，没有语义能力，
所以下面这 5 个查询的 top-5 只用来验证链路和排序合理性，数字不能当指标。

用法：
    python tests\\smoke_retrieve.py
    python tests\\smoke_retrieve.py --limit 3000 --top-k 5
    python tests\\smoke_retrieve.py --chunks G:\\kb\\data\\processed\\chunks_all.jsonl
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb.retrieve import HashingEmbedder, Retriever, build_index  # noqa: E402
from kb.retrieve.pipeline import load_chunks, write_limited_chunks  # noqa: E402

# 默认语料：只读，不要改。沙箱外的那份在 G:\kb\data\processed\。
DEFAULT_CHUNKS = Path(r"G:\kb\data\processed\chunks_all.jsonl")

# 覆盖中文长词、英文缩写、中英混排三类查询，用来验证 tokenizer 的三种路径
QUERIES = [
    "TCP 三次握手为什么需要三次",
    "HashMap 的红黑树转换阈值",
    "Redis 持久化 AOF 和 RDB 的区别",
    "线程池核心参数与拒绝策略",
    "JVM 内存模型与垃圾回收器",
]


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="真实语料检索链路自检")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--index-dir", default=str(ROOT / ".tmp" / "smoke-index"))
    parser.add_argument("--limit", type=int, default=3000,
                        help="只索引前 N 条（默认 3000，全量 18k 用 hashing 约 2 分钟）")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dim", type=int, default=512, help="HashingEmbedder 维度")
    parser.add_argument("--keep-index", action="store_true",
                        help="保留索引目录（默认跑完删掉）")
    args = parser.parse_args(argv)

    chunks_path = Path(args.chunks)
    index_dir = Path(args.index_dir)

    print("=" * 72)
    print("  检索链路自检（HashingEmbedder，不需要模型）")
    print("=" * 72)
    print(f"  语料     : {chunks_path}")
    print(f"  索引目录 : {index_dir}")
    print(f"  限制条数 : {args.limit}")
    print(f"  向量维度 : {args.dim}")
    print("-" * 72)

    if not chunks_path.exists():
        print(f"[X] 语料不存在：{chunks_path}")
        return 2

    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)

    t0 = time.perf_counter()
    chunks = load_chunks(chunks_path, limit=args.limit)
    texts = [c.get("embed_text") or c.get("text") or "" for c in chunks]
    total_chars = sum(len(t) for t in texts)
    print(f"  读到 {len(chunks):,} 个块，共 {total_chars:,} 字符"
          f"（{total_chars / max(len(chunks), 1):.0f} 字符/块）")

    embedder = HashingEmbedder(dim=args.dim)
    embedder.build_idf(texts)
    print(f"  idf 统计完成：{len(embedder.df):,} 个不同 token")

    # build_index 的签名是冻结的（没有 limit），「只索引前 N 条」靠先截一个
    # 临时 JSONL 实现；截出来的行与原文逐字节相同，所以 doc_index 不变。
    index_dir.mkdir(parents=True, exist_ok=True)
    limited_path = index_dir / "_limited_chunks.jsonl"
    written = write_limited_chunks(chunks_path, limited_path, args.limit)
    print(f"  自检语料 : {written:,} 行 -> {limited_path.name}")

    info = build_index(limited_path, index_dir, embedder,
                       batch_size=16, progress=True)
    limited_path.unlink(missing_ok=True)

    t_build = time.perf_counter() - t0
    print(f"  建索引完成，耗时 {t_build:.1f}s"
          f"（{info['vocab_size']:,} term / {info['n_postings']:,} posting）")
    print("-" * 72)

    retriever = Retriever(index_dir, embedder=embedder,
                          cfg={"dense_top_k": 50, "sparse_top_k": 50,
                               "fusion_top_k": 50})

    for q_index, query in enumerate(QUERIES, start=1):
        t1 = time.perf_counter()
        hits = retriever.search(query, top_k=args.top_k)
        elapsed_ms = (time.perf_counter() - t1) * 1000

        print(f"\n查询 {q_index}: {query}    ({elapsed_ms:.0f} ms)")
        print("-" * 72)
        if not hits:
            print("  (无命中)")
            continue
        for hit in hits:
            crumb = hit.breadcrumb or "(无标题)"
            print(f"  [{hit.rank}] {hit.score:.4f}  {crumb}")
            print(f"      {hit.location()}")
            snippet = " ".join(hit.text.split())
            print(f"      {snippet[:110]}{'...' if len(snippet) > 110 else ''}")

    print("\n" + "=" * 72)
    print("  链路自检通过。换真实模型：")
    print(f"    python ingest.py index --chunks \"{chunks_path}\" "
          f"--index-dir data\\index --embedder onnx")
    print("=" * 72)

    if not args.keep_index:
        shutil.rmtree(index_dir, ignore_errors=True)
        print(f"  已删除临时索引 {index_dir}（--keep-index 可保留）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
