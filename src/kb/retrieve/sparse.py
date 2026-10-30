"""Okapi BM25 倒排索引，只依赖 numpy。

## 为什么是 BM25 而不是 TF-IDF

BM25 相比「TF-IDF 余弦」多了一个文档长度归一化项 `tf/(tf + k1*(1-b+b*dl/avgdl))`。
语料里块长差异极大：`chunks_all.jsonl` 的块从 40 字符到 2415 字符，最长是
最短的 60 倍。没有长度归一化，长块因为「词多所以命中词多」而系统性排在
前面，短小精悍的答案块被压下去。b=0.75 让长块里的词频收益递减，这是
BM25 在短文本检索上稳定压过 TF-IDF 余弦的主要原因，也是本文件不省掉这
一项的理由。

    score(q, d) = Σ_{t∈q} IDF(t) · tf(t,d)·(k1+1)
                              / (tf(t,d) + k1·(1 - b + b·dl(d)/avgdl))

IDF 用带 +1 的 Robertson-Sparck-Jones 平滑形式：

    IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

加 1 保证 IDF 恒正。不修改这个的话，当一个词出现在超过一半文档里时
IDF 变负，负分词的贡献会把整体排序搅乱（常见于中文 unigram 里的「的」）。

## 存储设计：为什么是 numpy 平行数组而不是 pickle

持久化用三个等长的 numpy 平行数组描述整个倒排表：

    term_ids : int32  每个 (term, doc) posting 的 term_id
    doc_ids  : int32  该 posting 的 doc_index
    tfs      : int16  该 posting 的词频

外加 `vocab.json`（term -> term_id）和 `meta.json`（N / avgdl / k1 / b /
doc_lengths）。

不用 pickle 的两个理由：
  · 安全。pickle 反序列化会执行任意代码，索引文件一旦来自别处就是 RCE。
    本项目的 `np.load(..., allow_pickle=False)` 明确拒绝 object 数组。
  · 设计上的整洁。pickle 存的是「Python 对象的内存镜像」，跨版本、跨
    numpy 大版本都可能失效；平行数组是纯数据，任何语言都能读。

不用 JSON 存倒排表的理由：18,276 篇文档的 posting 数量在千万级，JSON
文本化后体积和解析时间都不可接受（而且 int 会变成字符串）。

## 存储的排序不变式（加载后查询快的原因）

`build()` 结束时按 `term_ids` 做了 argsort，所以三个数组是 term 有序的。
由此每个 term 的 posting 落在**一段连续区间**里，区间边界可以由一次
`np.searchsorted` 或 `np.unique(..., return_index=True)` 算出来，存成
`seg_starts`（长度 vocab_size+1）。查询时：

  1. `vocab.json` 把查询 token 映射成 term_id；
  2. `np.isin(qids, vocab_ids)` 还是 `np.searchsorted(vocab_ids, qids, 'left')`
     定位每个 term 在 vocab 里的位置 —— 只对「有几个查询词」这个规模做，
     与文档数无关；
  3. 用 `seg_starts[tid]:seg_starts[tid+1]` 切片拿到 posting 区间。

查询里不存在于词表的词直接丢弃（DF=0），不需要扫全表。真正耗时的只有
最后那一步按 term 逐个累加 BM25 分，候选集是「命中了至少一个查询词的
文档数」，对 5~10 个 token 的查询来说是几千量级，几十毫秒。

## term_id 与 doc_index 的取值约定

  · `term_ids` / `doc_ids` / `tfs` 里的 doc 位置统一用 `doc_index`
    （0-based，等于 JSONL 里的行号），pipeline 层负责把它映射回 chunk 元数据。
  · `vocab.json` 的 term_id 在 `build()` 内按 token 首次出现顺序分配；
    分配后立即排序落盘，加载时词表按 term_id 升序，可直接 `searchsorted`。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .tokenize import tokenize_counts

__all__ = ["BM25Index"]

# int16 存词频，超出就截断。BM25 里 tf 到几千以后贡献已经饱和
# （k1=1.5 时 tf=100 与 tf=1000 的分差 < 4%），截断无实质影响。
_TF_MAX = 32767

_VOCAB_NAME = "vocab.json"
_POSTINGS_NAME = "postings.npz"
_META_NAME = "meta.json"


class BM25Index:
    """Okapi BM25 倒排索引。

    用法：

        idx = BM25Index(k1=1.5, b=0.75)
        idx.build(["文本一", "文本二"])
        for doc_index, score in idx.search("查询词", top_k=50):
            ...
        idx.save(index_dir)               # 目录，不是文件
        idx2 = BM25Index.load(index_dir)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0:
            raise ValueError(f"k1 必须 > 0，收到 {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b 必须在 [0, 1]，收到 {b}")
        self.k1 = float(k1)
        self.b = float(b)
        self.reset()

    # ------------------------------------------------------------------ 状态

    def reset(self) -> None:
        self.vocab: dict[str, int] = {}
        self.term_ids = np.zeros(0, dtype=np.int32)
        self.doc_ids = np.zeros(0, dtype=np.int32)
        self.tfs = np.zeros(0, dtype=np.int16)
        # 长度 vocab_size+1，seg_starts[t]:seg_starts[t+1] 是 term t 的 posting 区间
        self.seg_starts = np.zeros(1, dtype=np.int64)
        self.doc_lengths = np.zeros(0, dtype=np.int32)
        self.idf = np.zeros(0, dtype=np.float32)
        # 预算好的 doc 长度归一化分母：k1 * (1 - b + b * dl / avgdl)
        self._len_norm = np.zeros(0, dtype=np.float32)
        self.n_docs = 0
        self.avgdl = 0.0

    def __len__(self) -> int:
        return self.n_docs

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def n_postings(self) -> int:
        return int(self.term_ids.shape[0])

    # ------------------------------------------------------------------ 建索引

    def build(self, texts: list[str]) -> None:
        """对 `texts` 建倒排索引。`texts[i]` 的 doc_index 就是 i。

        单遍扫描：分词 -> Counter -> 累加 posting 到 list of (term_id, doc, tf)。
        Python 层逐条 append 在 18k 文档上约几秒，可以接受；用 numpy 做
        「token 矩阵 -> 稀疏矩阵」反而要先生成千万级中间数组，更慢更占内存。
        """
        self.reset()
        n = len(texts)
        if n == 0:
            return

        vocab = self.vocab
        lengths = np.zeros(n, dtype=np.int32)
        postings_t: list[int] = []
        postings_d: list[int] = []
        postings_f: list[int] = []
        df_counter: dict[int, int] = defaultdict(int)

        for doc_index, text in enumerate(texts):
            counts = tokenize_counts(text or "")
            lengths[doc_index] = sum(counts.values())
            for term, tf in counts.items():
                tid = vocab.get(term)
                if tid is None:
                    tid = len(vocab)
                    vocab[term] = tid
                postings_t.append(tid)
                postings_d.append(doc_index)
                postings_f.append(tf if tf < _TF_MAX else _TF_MAX)
                df_counter[tid] += 1

        n_terms = len(vocab)
        term_ids = np.asarray(postings_t, dtype=np.int32)
        doc_ids = np.asarray(postings_d, dtype=np.int32)
        tfs = np.asarray(postings_f, dtype=np.int16)

        # 按 term_id 排序，让同一 term 的 posting 连续
        order = np.argsort(term_ids, kind="stable")
        self.term_ids = term_ids[order]
        self.doc_ids = doc_ids[order]
        self.tfs = tfs[order]

        self.doc_lengths = lengths
        self.n_docs = n
        self.avgdl = float(lengths.mean()) if n else 0.0

        # term 有序之后，term t 的区间起点 = 第一个 >= t 的位置
        seg = np.searchsorted(self.term_ids, np.arange(n_terms, dtype=np.int32), side="left")
        self.seg_starts = np.append(seg, self.term_ids.shape[0]).astype(np.int64)

        # 词表重新按 term_id 排序落成列表，供 searchsorted 使用
        self._rebuild_vocab_arrays()

        # IDF：ln(1 + (N - df + 0.5)/(df + 0.5))
        df = np.zeros(n_terms, dtype=np.float64)
        for tid, d in df_counter.items():
            df[tid] = d
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)

        self._recompute_len_norm()

    def _rebuild_vocab_arrays(self) -> None:
        """从 self.vocab 生成「按 term_id 升序的 term 列表」。

        build() 里 tid 是按首次出现顺序分配的，vocab.values() 恰好是
        0..n-1；load() 后 vocab 从 JSON 读回来，值同样是 0..n-1。
        这里只按 tid 摆正位置，方便调试时反查 term，不参与查询路径
        （查询走 dict，O(1)）。
        """
        terms = [""] * len(self.vocab)
        for term, tid in self.vocab.items():
            terms[tid] = term
        self._terms_by_id = terms

    def _recompute_len_norm(self) -> None:
        if self.n_docs == 0 or self.avgdl <= 0:
            self._len_norm = np.zeros(self.n_docs, dtype=np.float32)
            return
        dl = self.doc_lengths.astype(np.float32)
        self._len_norm = (self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)).astype(np.float32)

    # ------------------------------------------------------------------ 查询

    def lookup(self, term: str) -> int | None:
        """词 -> term_id，不存在返回 None。"""
        return self.vocab.get(term)

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """返回 [(doc_index, score)]，按分数降序。

        与文档数无关的预处理（词表映射、区间定位）只做 O(|query|) 次，
        真正的累加只遍历命中词的 posting。
        """
        if self.n_docs == 0 or top_k <= 0:
            return []

        counts = tokenize_counts(query or "")
        if not counts:
            return []

        # 查询词 -> term_id 去重（同一个词重复出现在查询里只算一次，
        # 这也是标准 BM25 的做法：查询词频不进公式）
        qids: list[int] = []
        for term in counts:
            tid = self.vocab.get(term)
            if tid is not None:
                qids.append(tid)
        if not qids:
            return []

        scores = np.zeros(self.n_docs, dtype=np.float32)
        for tid in qids:
            start = int(self.seg_starts[tid])
            end = int(self.seg_starts[tid + 1])
            if start == end:
                continue
            docs = self.doc_ids[start:end]
            tf = self.tfs[start:end].astype(np.float32)
            contrib = tf * (self.k1 + 1.0) / (tf + self._len_norm[docs])
            scores[docs] += self.idf[tid] * contrib

        k = min(top_k, self.n_docs)
        # argpartition 拿到前 k 的无序集合，再只对 k 个元素排序。
        # 全量 argsort 是 O(N log N)，N=18k 时能看到差别但不致命；
        # 规模再大一个量级时这一步就是分水岭。
        cand = np.argpartition(-scores, k - 1)[:k]
        cand = cand[scores[cand] > 0]  # 0 分等于没命中，不返回
        order = np.argsort(-scores[cand], kind="stable")
        cand = cand[order]
        return [(int(d), float(scores[d])) for d in cand]

    def score_all(self, query: str) -> np.ndarray:
        """返回长度 N 的稠密分数数组（调试 / 评测用，一般用 search）。"""
        out = np.zeros(self.n_docs, dtype=np.float32)
        for doc_index, score in self.search(query, top_k=self.n_docs):
            out[doc_index] = score
        return out

    # ------------------------------------------------------------------ 持久化

    def save(self, path: Path) -> None:
        """写入目录 `path`：postings.npz + vocab.json + meta.json。

        `path` 是目录（与 DenseIndex 共用同一个 index_dir 时文件名不同，
        互不覆盖）。
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.n_docs == 0:
            raise ValueError("索引为空，先调 build()")

        np.savez(
            path / _POSTINGS_NAME,
            term_ids=self.term_ids,
            doc_ids=self.doc_ids,
            tfs=self.tfs,
            seg_starts=self.seg_starts,
            doc_lengths=self.doc_lengths,
            idf=self.idf,
        )
        (path / _VOCAB_NAME).write_text(
            json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8"
        )
        (path / _META_NAME).write_text(
            json.dumps(
                {
                    "n_docs": self.n_docs,
                    "avgdl": self.avgdl,
                    "k1": self.k1,
                    "b": self.b,
                    "vocab_size": len(self.vocab),
                    "n_postings": self.n_postings,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        path = Path(path)
        meta_path = path / _META_NAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"BM25 索引不存在：{meta_path}\n"
                f"先建索引：  python ingest.py index --chunks <chunks.jsonl> --index-dir {path}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        obj = cls(k1=meta.get("k1", 1.5), b=meta.get("b", 0.75))
        obj.n_docs = int(meta["n_docs"])
        obj.avgdl = float(meta["avgdl"])
        obj.vocab = json.loads((path / _VOCAB_NAME).read_text(encoding="utf-8"))

        # allow_pickle=False 是硬要求：索引文件可能来自别处，不能让
        # np.load 有机会反序列化任意对象
        with np.load(path / _POSTINGS_NAME, allow_pickle=False) as data:
            obj.term_ids = data["term_ids"]
            obj.doc_ids = data["doc_ids"]
            obj.tfs = data["tfs"]
            obj.seg_starts = data["seg_starts"]
            obj.doc_lengths = data["doc_lengths"]
            obj.idf = data["idf"]

        obj._rebuild_vocab_arrays()
        obj._recompute_len_norm()
        return obj
