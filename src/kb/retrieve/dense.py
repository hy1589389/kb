"""稠密向量检索：暴力余弦 + argpartition + mmap。不用向量库。

## 为什么不上向量库（FAISS / Qdrant / LanceDB / Chroma）

规模算一下就有答案。语料是 18,276 个块，bge-m3 的稠密向量是 1024 维
float32：

    单个向量   1024 × 4 B          = 4 KB
    全量矩阵   18,276 × 1024 × 4 B  = 74.9 MB
    一次全量打分  18,276 × 1024 的矩阵乘 1024 维查询向量
                 ≈ 1.9 × 10^7 次乘加

在单核 CPU 上 numpy 的 sgemv 大约 5~10 GFLOP/s，也就是**一次查询几毫秒
到几十毫秒**（实测含 argpartition 与排序约 15 ms，见 docs/design-notes.md）。
这个数字比任何向量库的网络/序列化/进程内跳转开销都低：Qdrant 起一个
容器，一次 HTTP 往返本身就要几毫秒，还得维护一套额外的数据副本和 ID
映射。索引规模在十万级以下时，向量库是纯粹的负担 —— 它解决的问题
（十亿级向量的近似最近邻）在这里根本不存在。

75 MB 也完全放得进内存。本机内存 16 GB 起，即使按 10 倍语料增长
（18 万块、750 MB）也还是一次矩阵乘的事。真正需要换方案的门槛是
百万块以上（4 GB 矩阵），那时再谈 IVF/HNSW 不迟。

## mmap

`load(..., mmap=True)`（默认）用 `np.load(mmap_mode="r")` 把矩阵映射成
只读内存映射，而不是读进堆。好处有两个：

  · 页缓存命中时启动零拷贝。进程启动即用，不需要等 75 MB 读完；
  · 多个进程（FastAPI 的 worker、评测脚本）共享同一份物理页，不会
    每个进程各占 75 MB。

代价：每次 `search` 仍然会遍历整个矩阵（这是暴力检索的定义），所以
mmap 省的是**内存占用和启动时间**，不是查询时间。顺序扫描对页缓存很
友好，实测 mmap 与常驻内存的查询耗时差异在噪声范围内。

## 取 top-k 的方式

`np.argpartition(-sims, k-1)` 是 O(N) 的选择算法，只保证前 k 个在正确
位置上、彼此无序；再对选出来的 k 个做一次 O(k log k) 排序。相比
`np.argsort(-sims)[:k]`（O(N log N)）在 N=18k 时省掉约 40% 的时间，
N 越大收益越明显。

`np.argpartition` 要求 k <= N 且 k-1 < N，k == N 时退化为全排序，代码里
用 `min(top_k, n)` 兜住。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["DenseIndex"]

_EMB_NAME = "embeddings.npy"
_META_NAME = "dense_meta.json"


class DenseIndex:
    """稠密向量索引。向量在 `__init__` 时就要求已 L2 归一化。

    归一化后内积等于余弦相似度，省掉每次查询的除法。Embedder 的实现
    （见 embedder.py）保证输出已归一化，这条不变式在这里断言一次。
    """

    def __init__(self, embeddings: np.ndarray, meta: list[dict]):
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"embeddings 必须是二维 (N, dim)，收到 shape={emb.shape}")
        if len(meta) != emb.shape[0]:
            raise ValueError(
                f"meta 条数({len(meta)}) 与向量行数({emb.shape[0]}) 不一致"
            )
        self.embeddings = emb  # (N, dim) float32，已 L2 归一化
        self.meta = meta
        self.dim = int(emb.shape[1])

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def n_docs(self) -> int:
        return self.__len__()

    # ------------------------------------------------------------------ 查询

    def search(self, query_vec: np.ndarray, top_k: int = 50) -> list[tuple[int, float]]:
        """返回 [(doc_index, cosine)]，按分数降序。

        `query_vec` 会被就地归一化（传进来的是临时向量，改它没关系；
        如果传的是共享数组，请自己先 copy）。
        """
        if self.n_docs == 0 or top_k <= 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(
                f"查询向量维度 {q.shape[0]} 与索引维度 {self.dim} 不一致；"
                f"检查建索引和查询用的是不是同一个 embedder"
            )
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm

        sims = self.embeddings @ q  # (N,) 一次矩阵乘出全部相似度
        k = min(top_k, self.n_docs)
        if k == self.n_docs:
            cand = np.argsort(-sims, kind="stable")
        else:
            cand = np.argpartition(-sims, k - 1)[:k]
            cand = cand[np.argsort(-sims[cand], kind="stable")]
        return [(int(i), float(sims[i])) for i in cand]

    def search_batch(self, query_vecs: np.ndarray, top_k: int = 50) -> list[list[tuple[int, float]]]:
        """批量查询：一次 (N, dim) @ (dim, B) 矩阵乘，评测集跑批时快得多。"""
        qs = np.asarray(query_vecs, dtype=np.float32)
        if qs.ndim == 1:
            qs = qs[None, :]
        norms = np.linalg.norm(qs, axis=1, keepdims=True)
        qs = qs / np.maximum(norms, 1e-12)
        sims = self.embeddings @ qs.T  # (N, B)
        out = []
        for b in range(sims.shape[1]):
            col = sims[:, b]
            k = min(top_k, self.n_docs)
            cand = (np.argsort(-col, kind="stable") if k == self.n_docs
                    else np.argpartition(-col, k - 1)[:k])
            cand = cand[np.argsort(-col[cand], kind="stable")]
            out.append([(int(i), float(col[i])) for i in cand])
        return out

    # ------------------------------------------------------------------ 持久化

    def save(self, path: Path) -> None:
        """写入 `embeddings.npy` + `dense_meta.json`。

        用 `np.save` 而不是 `np.savez`：单个 .npy 文件天然支持 mmap，
        压缩的 npz 必须先整体解压，等于放弃 mmap 的意义。
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / _EMB_NAME, self.embeddings)
        (path / _META_NAME).write_text(
            json.dumps(
                {"n_docs": self.n_docs, "dim": self.dim},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, mmap: bool = True) -> "DenseIndex":
        """加载索引。`mmap=True`（默认）用只读内存映射，省内存和启动时间。

        `mmap=True` 时 `self.embeddings` 的 `base` 是一个 memmap，矩阵乘
        照常工作；但**不要**对它做跨进程传递或就地写。
        """
        path = Path(path)
        meta_path = path / _META_NAME
        emb_path = path / _EMB_NAME
        if not meta_path.exists() or not emb_path.exists():
            raise FileNotFoundError(
                f"稠密索引不存在：{emb_path}\n"
                f"先建索引：  python ingest.py index --chunks <chunks.jsonl> --index-dir {path}"
            )
        meta_info = json.loads(meta_path.read_text(encoding="utf-8"))
        mode = "r" if mmap else None
        embeddings = np.load(emb_path, mmap_mode=mode, allow_pickle=False)

        obj = cls.__new__(cls)
        obj.embeddings = embeddings
        obj.dim = int(meta_info.get("dim", embeddings.shape[1]))
        # 元数据不随稠密索引存（一份就够，见 pipeline 的 chunks_meta.json）。
        # 这里填等长占位列表，保证 len(index.meta) == len(index) 这条不变式
        # 在加载后依然成立；pipeline 会用真实元数据覆盖它。
        obj.meta = [None] * int(embeddings.shape[0])
        return obj
