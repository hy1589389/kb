"""检索流水线：建索引 + 查询。

## 索引目录的内容

`build_index()` 在一个目录里产出四个文件，`Retriever` 只读它们：

    dense_embeddings.npy   (N, dim) float32，已 L2 归一化，支持 mmap
    bm25_postings.npz      BM25 倒排表的 numpy 平行数组
    bm25_vocab.json        token -> term_id
    bm25_meta.json         N / avgdl / k1 / b / idf
    chunks_meta.jsonl      第 i 行 = 第 i 个 chunk 的元数据（与向量行号一一对应）
    index_meta.json        建索引时的配置（embedder 类型、dim、参数）

`chunks_meta.jsonl` 与 `chunks_all.jsonl` 的区别是**丢掉 embed_text**。
embed_text（breadcrumb + 正文）只用于建索引时的嵌入，检索阶段用不到，而它
占原文件近一半体积；正文 `text` 保留，因为重排和结果展示都需要它，不存
的话每次查询都得回读 44 MB 的原始 JSONL。整体索引目录约为原始语料的 1.5 倍
（75 MB 向量 + 倒排表 + 约 22 MB 元数据）。

## dense 与 sparse 的行号是同一个 doc_index

两个索引都按 chunks 文件的**行序**建，所以 `doc_index == chunks_meta.jsonl
的行号`。RRF 融合两路结果时不需要任何 ID 映射，直接用 doc_index。这条
不变式在 `build_index()` 里是靠「同一个迭代顺序、同一份 meta 列表」保证
的；一旦有人给 BM25 加过滤条件而稠密那路没加，两边的行号就会错位 ——
所以 `Retriever` 加载时会断言两个索引的长度相同。

## 查询流程

    query
      ├─ dense:  embedder.encode([query]) -> DenseIndex.search(top_k=dense_top_k)
      └─ sparse: BM25Index.search(top_k=sparse_top_k)
                    ↓
              rrf_fuse(k=rrf_k)  -> fusion_top_k 条
                    ↓
              [可选] OnnxReranker.score(query, passages) -> 重排 -> final_top_k

重排只对融合后的 `fusion_top_k`（默认 50）条生效，因为交叉编码器需要对
每个 (query, passage) 对做一次完整前向，成本与候选数线性相关，而对全部
18k 块做交叉编码在 CPU 上是几十分钟，不现实。

## 多集合（多索引）

`Retriever` 的 `index_dir` 参数接受两种形态：

    Retriever(Path("data/index"))                      # 单索引，向后兼容
    Retriever({"tech": Path("data/index"),
               "profile": Path("data/index/profile")})  # 多集合

索引目录是**物理隔离**的：技术文档 18k 块在 `data/index/`，简历几十块在
`data/index/profile/`，各自独立建、独立删。查询时每个集合内部先做
dense+sparse 的 RRF 融合，再用 `rrf_fuse` 跨集合合并一次，权重来自
`cfg["collections"]`（`{集合名: 权重}`，默认 1.0）。所有索引在 `__init__` 里
加载一次，请求期不再读盘。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .dense import DenseIndex
from .embedder import Embedder, HashingEmbedder, OnnxEmbedder, OnnxReranker
from .hybrid import rrf_fuse
from .sparse import BM25Index
from .types import SearchHit

__all__ = ["Retriever", "build_index", "load_chunks", "write_limited_chunks",
           "DEFAULT_CFG"]

# 检索参数的默认值。与 config.toml 的 [retrieval] 段同名，
# 显式传 cfg 时按 key 覆盖。
DEFAULT_CFG: dict = {
    "dense_top_k": 50,
    "sparse_top_k": 50,
    "fusion_top_k": 50,
    "final_top_k": 10,
    "rrf_k": 60,
    "dense_weight": 1.0,
    "sparse_weight": 1.0,
    "use_reranker": False,
    "rerank_batch_size": 8,
    "embed_batch_size": 16,
    "embed_max_length": 512,
    # 单索引模式下给结果打的集合名（多集合模式用 dict 的 key，不看这个）
    "collection": "",
    # 多集合模式下每个集合的权重：{"tech": 1.0, "profile": 0.8}
    "collections": {},
}

_META_NAME = "chunks_meta.jsonl"
_INDEX_META = "index_meta.json"


def _json_failure(path: Path, exc: Exception) -> str:
    """JSON 解析失败的报错文本：指出文件、开头内容，以及「是不是 LFS 指针」。

    不加这个的时候，线上只有一句 `[X] Expecting value: line 1 column 1 (char 0)`，
    既不知道是哪个文件，也看不出索引是没构建好还是大文件没拉下来。
    """
    head = ""
    try:
        with path.open("rb") as fh:
            head = fh.read(60).decode("utf-8", "replace").replace("\n", " ")
    except OSError:
        pass
    extra = ""
    if head.startswith("version https://git-lfs"):
        extra = ("\n这是 LFS 指针，不是文件内容：构建/部署时大文件没被拉下来。"
                 "用 python tools/fetch_index.py data/index --repo <owner>/<space> 重拉一份。")
    return f"{path} 不是合法 JSON：{exc}\n开头是：{head!r}{extra}"


# 元数据里保留的字段。`embed_text` 不进索引元数据（它只用于建索引时的
# 嵌入，检索时用不到），但 `text` 要留 —— 重排需要正文，展示也需要正文，
# 不存的话每次查询都得回读 44 MB 的 chunks_all.jsonl。
#
# `collection` / `locator` / `low_quality` 是后加的：老索引里没有这三列，
# 读取时用 `_META_DEFAULTS` 兜底，所以旧索引不用重建也能查（只是地点显示
# 退回行号、低质量标记为空）。
_META_KEYS = (
    "chunk_id", "doc_id", "repo", "rel_path", "breadcrumb",
    "line_start", "line_end", "char_count", "cjk_count", "text",
    "collection", "locator", "low_quality",
)

_META_DEFAULTS: dict = {
    "line_start": 0,
    "line_end": 0,
    "low_quality": False,
}


def load_chunks(chunks_path: Path, limit: int | None = None) -> list[dict]:
    """读 JSONL，返回按行序的 dict 列表。

    逐行 `json.loads` 而不是 `read_text().splitlines()`：44 MB 全读进内存
    再切会多占一份内存，而且在 `--limit` 场景下白读后面几万行。
    """
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"语料文件不存在：{chunks_path}\n"
            f"这是 Phase 1 的产物，先在语料仓库目录跑：  python ingest.py ingest"
        )
    out: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    if not out:
        raise ValueError(f"语料文件为空：{chunks_path}")
    return out


def write_limited_chunks(chunks_path: Path, out_path: Path,
                         limit: int) -> int:
    """把 JSONL 的前 `limit` 行原样写到 `out_path`，返回实际写出的行数。

    `build_index` 只接受一个 chunks 路径，没有 limit 参数（签名是冻结的）。
    「只索引前 3000 条做自检」这个需求靠这个函数实现：先截出一个临时
    JSONL，再把它当正常语料传给 `build_index`。截出来的行是原始字节的
    拷贝，不做解析和重写，所以行号（= doc_index）与原文件一致。
    """
    chunks_path = Path(chunks_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with chunks_path.open("r", encoding="utf-8") as src, \
            out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(line if line.endswith("\n") else line + "\n")
            written += 1
            if written >= limit:
                break
    return written


def _meta_of(chunk: dict) -> dict:
    out: dict = {}
    for key in _META_KEYS:
        if key in chunk:
            out[key] = chunk[key]
        else:
            out[key] = _META_DEFAULTS.get(key, "")
    return out


def _make_embedder(kind: str, model_dir: Path | None, max_length: int,
                   batch_size: int, texts_for_idf: list[str] | None = None) -> Embedder:
    """按名字构造 embedder。`hashing` 需要语料来统计 idf。"""
    kind = (kind or "onnx").lower()
    if kind == "hashing":
        emb = HashingEmbedder(dim=512)
        if texts_for_idf:
            emb.build_idf(texts_for_idf)
        return emb
    if kind == "onnx":
        if model_dir is None:
            raise ValueError("embedder=onnx 时必须提供 model_dir")
        return OnnxEmbedder(model_dir, max_length=max_length, batch_size=batch_size)
    raise ValueError(f"未知 embedder 类型：{kind}（可选 onnx / hashing）")


def build_index(
    chunks_path: Path,
    index_dir: Path,
    embedder: Embedder,
    *,
    batch_size: int = 16,
    max_length: int = 512,
    progress: bool = True,
) -> dict:
    """建稠密 + 稀疏双索引，写入 `index_dir`。

    参数
    ----
    chunks_path : Phase 1 产出的 chunks_all.jsonl
    index_dir   : 输出目录（不存在会创建）
    embedder    : 任何 Embedder 实现。用 HashingEmbedder 可以在没有模型的
                  情况下先跑通整条链路自检
    batch_size  : 嵌入的批大小（CPU 上 16 一般比 1 快 3~5 倍）
    max_length  : 嵌入输入截断长度
    progress    : 是否打印进度（tqdm；没装就退化成朴素打印）

    返回
    ----
    dict，含 n_chunks / dim / vocab_size / n_postings / elapsed_sec 等，
    同时原样写进 `index_meta.json` 供后续查询和评测记录。
    """
    chunks_path = Path(chunks_path)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    chunks = load_chunks(chunks_path)
    n = len(chunks)
    # 嵌入的是 embed_text（= breadcrumb + 正文），不是 text。这是 Phase 1
    # 就定下的约定：把标题路径拼进正文，查询词命中标题时也能召回正文。
    texts = [c.get("embed_text") or c.get("text") or "" for c in chunks]
    meta = [_meta_of(c) for c in chunks]

    # ---- 稀疏索引 ----
    if progress:
        print(f"[1/3] BM25 建索引：{n:,} 篇 ...")
    bm25 = BM25Index(k1=1.5, b=0.75)
    bm25.build(texts)
    bm25.save(index_dir)
    if progress:
        print(f"      词表 {bm25.vocab_size:,} 个 term，"
              f"{bm25.n_postings:,} 条 posting，avgdl={bm25.avgdl:.1f}")

    # ---- 稠密索引 ----
    if progress:
        print(f"[2/3] 稠密嵌入：dim={embedder.dim}，batch={batch_size}，"
              f"max_length={max_length} ...")
    emb = _encode_with_progress(embedder, texts, batch_size, progress)
    if emb.shape != (n, embedder.dim):
        raise RuntimeError(
            f"embedder 返回的 shape {emb.shape} 与预期 {(n, embedder.dim)} 不符"
        )
    DenseIndex(emb, meta).save(index_dir)
    mem_mb = emb.nbytes / 1024 / 1024
    if progress:
        print(f"      向量矩阵 {emb.shape[0]:,} × {emb.shape[1]} "
              f"= {mem_mb:.1f} MB")

    # ---- 元数据 ----
    if progress:
        print(f"[3/3] 写元数据 ...")
    with (index_dir / _META_NAME).open("w", encoding="utf-8") as fh:
        for m in meta:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t0
    info = {
        "n_chunks": n,
        "dim": int(embedder.dim),
        "embedder": type(embedder).__name__,
        "vocab_size": bm25.vocab_size,
        "n_postings": bm25.n_postings,
        "avgdl": round(bm25.avgdl, 3),
        "batch_size": batch_size,
        "max_length": max_length,
        "chunks_path": str(chunks_path),
        "elapsed_sec": round(elapsed, 2),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (index_dir / _INDEX_META).write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if progress:
        print(f"      索引目录：{index_dir}   耗时 {elapsed:.1f}s")
    return info


def _encode_with_progress(embedder: Embedder, texts: list[str], batch_size: int,
                          progress: bool) -> np.ndarray:
    """分批编码并显示进度。

    分块编码而不是一次 `encode(texts)`：18k 条文本一次性交给 tokenizers
    会先构造 18k 个 Encoding 对象（每个带 ids/mask/offsets/type_ids），
    内存峰值比向量本身还大。按 batch 走可以把峰值压到 batch 级别。
    """
    n = len(texts)
    bs = max(1, int(batch_size))
    outs: list[np.ndarray] = []

    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=n, unit="chunk", desc="      encode",
                       dynamic_ncols=True)
        except ImportError:
            bar = None

    try:
        for start in range(0, n, bs):
            part = texts[start:start + bs]
            outs.append(np.asarray(embedder.encode(part, batch_size=bs),
                                   dtype=np.float32))
            if bar is not None:
                bar.update(len(part))
            elif progress and (start // bs) % 20 == 0:
                print(f"      {min(start + bs, n):,}/{n:,}")
    finally:
        if bar is not None:
            bar.close()

    return np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]


class _IndexPart:
    """一个索引目录加载后的只读视图。

    单索引模式只有一个 part（名字取 `cfg["collection"]`，通常为空串），多集合
    模式每个集合一个。`bm25` / `dense` / `meta` 三条数据的行号必须一一对应，
    加载时就断言，别等到检索时拿到错位的元数据。
    """

    def __init__(self, name: str, index_dir: Path, cfg: dict):
        self.name = name
        self.index_dir = Path(index_dir)
        self.cfg = cfg

        meta_path = self.index_dir / _INDEX_META
        if not meta_path.exists():
            raise FileNotFoundError(
                f"索引目录 {self.index_dir} 里没有 {_INDEX_META}\n"
                f"先建索引：  python ingest.py index --chunks <chunks.jsonl> "
                f"--index-dir {self.index_dir}"
            )
        try:
            self.index_info = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(_json_failure(meta_path, exc)) from exc

        chunks_meta = self.index_dir / _META_NAME
        if not chunks_meta.exists():
            raise FileNotFoundError(
                f"索引目录 {self.index_dir} 里没有 {_META_NAME}，重新建索引"
            )
        meta: list[dict] = []
        try:
            with chunks_meta.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        meta.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(_json_failure(chunks_meta, exc)) from exc
        self.meta = meta

        self.bm25 = BM25Index.load(self.index_dir)
        self.dense = DenseIndex.load(self.index_dir, mmap=True)

        if len(self.bm25) != self.dense.n_docs or len(self.meta) != self.dense.n_docs:
            raise ValueError(
                f"索引不一致（{self.label()}）：bm25={len(self.bm25)} "
                f"dense={self.dense.n_docs} meta={len(self.meta)}。"
                f"索引目录可能被部分覆盖，请重新建索引。"
            )

    def label(self) -> str:
        return f"集合 {self.name}" if self.name else str(self.index_dir)

    @property
    def n_docs(self) -> int:
        return int(self.dense.n_docs)

    def is_low_quality(self, doc_index: int) -> bool:
        """这一块的元数据是否被标了 low_quality（老索引没有该字段 -> False）。"""
        return bool(self.meta[doc_index].get("low_quality"))

    def collection_of(self, doc_index: int) -> str:
        """来源集合名：优先用建索引时记的，退回索引自己的名字。"""
        return str(self.meta[doc_index].get("collection") or self.name or "")


class Retriever:
    """加载索引并执行检索。

    典型用法：

        r = Retriever(index_dir, embedder=HashingEmbedder())   # 自检
        hits = r.search("TCP 三次握手", top_k=5)

        # 生产：不传 embedder，用建索引时记录的 embedder 类型自动加载
        r = Retriever(index_dir, cfg={"model_dir": r"G:\\kb\\models\\bge-m3"})
        hits = r.search("...", top_k=10, use_rerank=True)

        # 多集合：各自检索 + 跨集合 RRF，结果带 collection 标记
        r = Retriever({"tech": Path("data/index"),
                       "profile": Path("data/index/profile")},
                      cfg={"collections": {"tech": 1.0, "profile": 0.8}})
        hits = r.search("项目经历", collections=["tech", "profile"])

    `embedder` 不传时会尝试按 `index_meta.json` 里记录的 embedder 类型
    自动构造。hashing 索引只查得到 hashing 的查询向量，onnx 索引同理 ——
    建索引和查询用的 embedder 必须是同一个，否则向量空间不对齐，检索结果
    是噪声（而且不会报错）。`build_index` 把类型记进 `index_meta.json`
    就是为了防这个，`_ensure_embedder` 在类型不匹配时直接抛错。

    多集合时所有索引在 `__init__` 里加载一次（`self.parts`），查询期不再读盘。
    """

    def __init__(self, index_dir: Path | Mapping[str, Path],
                 embedder: Embedder | None = None,
                 reranker: OnnxReranker | None = None, cfg: dict | None = None):
        self.cfg = dict(DEFAULT_CFG)
        if cfg:
            unknown = set(cfg) - set(DEFAULT_CFG) - {"model_dir", "reranker_dir"}
            if unknown:
                # 不直接报错：评测脚本可能顺手塞别的键进来。但静默忽略
                # 拼错的键会导致「配了没生效」的假象，所以警告一次。
                print(f"[!] 忽略未知的检索配置项：{sorted(unknown)}")
            self.cfg.update(cfg)
        self.model_dir = self.cfg.get("model_dir")
        self.reranker_dir = self.cfg.get("reranker_dir")

        # index_dir 可以是单个路径（向后兼容），也可以是 {集合名: 路径}
        self.multi = isinstance(index_dir, Mapping)
        if self.multi:
            dirs = {str(k): Path(v) for k, v in index_dir.items()}
            if not dirs:
                raise ValueError("多集合检索至少要给一个索引目录")
        else:
            dirs = {str(self.cfg.get("collection") or ""): Path(index_dir)}
        self.index_dirs = dirs

        self.embedder = embedder
        self.reranker = reranker
        self._reranker_checked = reranker is not None
        # 模型懒加载的锁（见 `_ensure_embedder`）：后台预热和请求线程会同时走到那里
        self._embedder_lock = threading.Lock()

        self.parts: dict[str, _IndexPart] = {
            name: _IndexPart(name, path, self.cfg) for name, path in dirs.items()
        }

        # 兼容单索引时代的属性：web / 评测脚本读 index_info、meta、index_dir
        first = next(iter(self.parts.values()))
        self.index_info = first.index_info
        self.index_dir = first.index_dir
        self.bm25 = first.bm25
        self.dense = first.dense
        if self.multi:
            self.meta = [row for part in self.parts.values() for row in part.meta]
        else:
            self.meta = first.meta

        self._check_embedders()

    # --------------------------------------------------------------- 加载

    def _check_embedders(self) -> None:
        """每个索引单独校验 embedder；多集合还要求它们是同一种。"""
        recorded_kinds: set[str] = set()
        dims: set[int] = set()
        for part in self.parts.values():
            recorded = part.index_info.get("embedder")
            dim = part.index_info.get("dim")
            if self.embedder is not None:
                if recorded:
                    _assert_embedder_matches(self.embedder, recorded, part.index_info)
                if dim and int(self.embedder.dim) != int(dim):
                    raise ValueError(
                        f"embedder 维度({self.embedder.dim}) 与索引维度({dim}) "
                        f"不一致（{part.label()}），检索结果会是噪声。"
                        f"请用建索引时的同一个 embedder。"
                    )
            if recorded:
                recorded_kinds.add(str(recorded).lower())
            if dim:
                dims.add(int(dim))

        if self.embedder is None and (len(recorded_kinds) > 1 or len(dims) > 1):
            raise ValueError(
                f"多个集合的索引不是用同一个 embedder 建的："
                f"{sorted(recorded_kinds)} / dim={sorted(dims)}。\n"
                f"跨集合查询只有一个查询向量，必须统一；请显式传 embedder。"
            )

    def _ensure_embedder(self) -> Embedder:
        """懒加载嵌入器。

        加锁是因为 Web 启动方式变了：端口先监听，模型在后台线程预热。这段时间
        若有请求进来，请求线程和预热线程会同时走到这里 —— 不加锁会各自构造一份
        ONNX session（2 GB 模型 × 2，直接 OOM）。锁只保护构造过程，编码本身由
        `OnnxEmbedder._run_batch` 里的锁串行化。
        """
        if self.embedder is not None:
            return self.embedder
        with self._embedder_lock:
            if self.embedder is not None:      # 双检：等锁时别人已经构造好了
                return self.embedder
            self.embedder = self._build_embedder()
            return self.embedder

    def _build_embedder(self) -> Embedder:
        kind = (self.index_info.get("embedder") or "OnnxEmbedder").lower()
        if kind == "hashingembedder":
            return HashingEmbedder(dim=int(self.index_info.get("dim", 512)))
        if self.model_dir is None:
            raise ValueError(
                "稠密检索需要 embedder，但配置里没有 model_dir。\n"
                "在 config.toml 的 [retrieval] 段设置 model_dir，或构造 Retriever 时传 "
                "cfg={'model_dir': ...}；\n"
                "只想先自检的话，用建索引时的 hashing 索引 + HashingEmbedder()。"
            )
        return OnnxEmbedder(
            self.model_dir,
            max_length=int(self.cfg["embed_max_length"]),
            batch_size=int(self.cfg["embed_batch_size"]),
        )

    def _ensure_reranker(self) -> OnnxReranker | None:
        if self._reranker_checked:
            return self.reranker
        self._reranker_checked = True
        if self.reranker_dir is None:
            print("[!] --rerank 已开启，但配置里没有 reranker_dir，跳过重排。")
            return None
        self.reranker = OnnxReranker(
            self.reranker_dir, max_length=int(self.cfg["embed_max_length"])
        )
        return self.reranker

    # --------------------------------------------------------------- 查询

    def collection_names(self) -> list[str]:
        """已加载的集合名（单索引模式下可能是一个空串）。"""
        return list(self.parts)

    def collection_weight(self, name: str) -> float:
        """跨集合 RRF 时该集合的权重，默认 1.0。

        `cfg["collections"]` 支持两种写法：直接给数字，或给一张带 `weight`
        的表（与 TOML 的 `[collections.<name>]` 对齐）。
        """
        overrides = self.cfg.get("collections") or {}
        raw = overrides.get(name, 1.0) if isinstance(overrides, Mapping) else 1.0
        if isinstance(raw, Mapping):
            raw = raw.get("weight", 1.0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 1.0

    def _select_parts(self, collections) -> list[_IndexPart]:
        if collections is None:
            return list(self.parts.values())
        if isinstance(collections, str):
            names = [x.strip() for x in collections.split(",")]
        else:
            names = [str(x).strip() for x in collections]
        names = [n for n in names if n]
        if not names:
            return list(self.parts.values())
        # 单索引模式没有集合名可比（名字是空串），任何请求都当「未指定」
        if "" in self.parts:
            return list(self.parts.values())
        missing = [n for n in names if n not in self.parts]
        if missing:
            raise ValueError(
                f"没有这些集合的索引：{', '.join(missing)}；"
                f"现有：{', '.join(self.parts)}。\n"
                f"先建索引：python ingest.py index --collection {missing[0]}"
            )
        return [self.parts[n] for n in names]

    def _query_vector(self, query: str) -> np.ndarray:
        return self._ensure_embedder().encode([query], batch_size=1)[0]

    def _part_rank(self, part: _IndexPart, query: str, qvec: np.ndarray | None,
                   use_dense: bool, use_sparse: bool,
                   include_low_quality: bool
                   ) -> tuple[list[int], dict[int, float], str, dict[int, dict]]:
        """一个集合内部：dense / sparse 两路 -> RRF -> 名次列表。

        低质量块在这一层就过滤掉，不能等到融合之后：被过滤掉的块如果参与了
        名次计算，会把正常块往后挤，等于让垃圾数据占着名次不放。

        返回 `(名次列表, 融合分, source, 每条的来源明细)`。明细形如
        `{doc_index: {"dense": {"rank": 1, "score": 0.6229},
                     "sparse": {"rank": 21, "score": 6.31}}}`，
        用来回答「这条为什么排第一」—— hybrid 的 `score` 是 RRF 分，
        只有 0.03 量级，光看它看不出名次从哪来。
        """
        rankings: list[list[int]] = []
        weights: list[float] = []
        raw_scores: dict[int, float] = {}
        detail: dict[int, dict] = {}
        fusion_k = int(self.cfg["fusion_top_k"])

        if use_dense and qvec is not None:
            dense_hits = part.dense.search(qvec, top_k=int(self.cfg["dense_top_k"]))
            if not include_low_quality:
                dense_hits = [(i, s) for i, s in dense_hits if not part.is_low_quality(i)]
            rankings.append([i for i, _ in dense_hits])
            weights.append(float(self.cfg["dense_weight"]))
            raw_scores = dict(dense_hits)
            for position, (doc, value) in enumerate(dense_hits, start=1):
                detail.setdefault(doc, {})["dense"] = {"rank": position,
                                                       "score": float(value)}
        if use_sparse:
            sparse_hits = part.bm25.search(query, top_k=int(self.cfg["sparse_top_k"]))
            if not include_low_quality:
                sparse_hits = [(i, s) for i, s in sparse_hits if not part.is_low_quality(i)]
            rankings.append([i for i, _ in sparse_hits])
            weights.append(float(self.cfg["sparse_weight"]))
            if not use_dense:
                raw_scores = dict(sparse_hits)
            for position, (doc, value) in enumerate(sparse_hits, start=1):
                detail.setdefault(doc, {})["sparse"] = {"rank": position,
                                                        "score": float(value)}

        if len(rankings) == 1:
            # 只有一路时不做 RRF：直接用原始分数。余弦相似度和 BM25 分数
            # 各自都有意义，评测单路效果时要用原始分数；RRF 会把它压成
            # 1/(60+rank) 这种只有名次含义的数，信息量反而变少。
            return (rankings[0][:fusion_k], raw_scores,
                    ("dense" if use_dense else "sparse"), detail)

        fused_all = rrf_fuse(rankings, k=int(self.cfg["rrf_k"]), weights=weights)
        return ([i for i, _ in fused_all[:fusion_k]], dict(fused_all), "hybrid", detail)

    def search(self, query: str, top_k: int = 10, *, use_dense: bool = True,
               use_sparse: bool = True, use_rerank: bool = False,
               collections: list[str] | str | None = None,
               include_low_quality: bool = False) -> list[SearchHit]:
        """执行检索。

        `top_k` 覆盖 cfg 里的 `final_top_k`。`use_dense` / `use_sparse`
        至少要开一个（两个都关没有候选集）。

        `collections` 只在多集合模式下有意义，只查指定的那几个；留空查全部。

        `include_low_quality=False`（默认）排除被质量指标标成 `low_quality`
        的块；查简历 / 扫描件时要复查这些块就传 True。

        多集合时 `score` 是**跨集合 RRF 融合分**，`source` 统一为 `hybrid`：
        名次才是可比量，绝对分数在集合之间、在 dense/sparse 之间都没有可比性。
        """
        if not use_dense and not use_sparse:
            raise ValueError("use_dense 和 use_sparse 不能同时为 False")
        query = (query or "").strip()
        if not query:
            return []
        final_k = int(top_k) if top_k else int(self.cfg["final_top_k"])
        parts = self._select_parts(collections)
        qvec = self._query_vector(query) if use_dense else None

        if len(parts) == 1:
            part = parts[0]
            order, scores, source, detail = self._part_rank(
                part, query, qvec, use_dense, use_sparse, include_low_quality
            )
            hits = [self._to_hit(part, i, scores.get(i, 0.0), source, rank,
                                 detail.get(i))
                    for rank, i in enumerate(order, start=1)]
            if use_rerank:
                hits = self._rerank(query, hits, final_k)
            return hits[:final_k]

        # ---- 多集合：各集合内部先融合，再按集合权重做一次跨集合 RRF ----
        # 各集合的行号是各自的 0..N-1，先加上偏移变成全局 id 再交给 rrf_fuse，
        # 否则 tech 的第 3 块和 profile 的第 3 块会被当成同一个文档。
        rankings: list[list[int]] = []
        weights_list: list[float] = []
        mapping: dict[int, tuple[_IndexPart, int]] = {}
        details: dict[int, dict] = {}
        offset = 0
        for part in parts:
            order, _scores, _source, detail = self._part_rank(
                part, query, qvec, use_dense, use_sparse, include_low_quality
            )
            rankings.append([offset + i for i in order])
            for i in order:
                mapping[offset + i] = (part, i)
                # 明细跟着全局 id 走，否则多集合时各行各路的名次会串
                details[offset + i] = detail.get(i) or {}
            weights_list.append(self.collection_weight(part.name))
            offset += part.n_docs

        fused = rrf_fuse(rankings, k=int(self.cfg["rrf_k"]), weights=weights_list)
        hits = []
        for rank, (gid, score) in enumerate(
                fused[:int(self.cfg["fusion_top_k"])], start=1):
            part, local_index = mapping[gid]
            hits.append(self._to_hit(part, local_index, float(score), "hybrid", rank,
                                     details.get(gid)))

        if use_rerank:
            hits = self._rerank(query, hits, final_k)
        return hits[:final_k]

    def _to_hit(self, part: _IndexPart, doc_index: int, score: float,
                source: str, rank: int, sources: dict | None = None) -> SearchHit:
        m = part.meta[doc_index]
        return SearchHit(
            chunk_id=m.get("chunk_id", ""),
            doc_id=m.get("doc_id", ""),
            repo=m.get("repo", ""),
            rel_path=m.get("rel_path", ""),
            breadcrumb=m.get("breadcrumb", ""),
            line_start=int(m.get("line_start", 0) or 0),
            line_end=int(m.get("line_end", 0) or 0),
            text=m.get("text", "") or "",
            score=float(score),
            source=source,
            rank=rank,
            collection=part.collection_of(doc_index),
            locator=str(m.get("locator") or ""),
            sources=dict(sources or {}),
        )

    def _rerank(self, query: str, hits: list[SearchHit], final_k: int) -> list[SearchHit]:
        reranker = self._ensure_reranker()
        if reranker is None or not hits:
            return hits
        passages = [h.text for h in hits]
        scores = reranker.score(query, passages,
                               batch_size=int(self.cfg["rerank_batch_size"]))
        order = np.argsort(-np.asarray(scores), kind="stable")
        out: list[SearchHit] = []
        for new_rank, pos in enumerate(order, start=1):
            hit = hits[int(pos)]
            hit.score = float(scores[int(pos)])
            hit.source = "rerank"
            hit.rank = new_rank
            out.append(hit)
        return out


def _assert_embedder_matches(embedder: Embedder, recorded: str, info: dict) -> None:
    """建索引与查询必须用同一种 embedder。

    这个检查值得单独写：两种 embedder 的向量空间完全无关（一个是语义模型，
    一个是哈希桶），混用不会抛任何异常，只会安静地返回垃圾结果。
    """
    want = "HashingEmbedder" if "hashing" in recorded.lower() else "OnnxEmbedder"
    got = type(embedder).__name__
    if want != got:
        raise ValueError(
            f"embedder 类型不匹配：索引是用 {recorded} 建的，查询传的是 {got}。\n"
            f"两种 embedder 的向量空间不同，混用不会报错但结果全是噪声。\n"
            f"请用同一种：建索引时 --embedder {'hashing' if want == 'HashingEmbedder' else 'onnx'}。"
        )
