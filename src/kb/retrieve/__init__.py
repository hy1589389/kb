"""Phase 2 检索层。

对外只暴露这些名字，评测脚本和 Web 层都从这里导入：

    from kb.retrieve import Retriever, build_index, SearchHit
    from kb.retrieve import BM25Index, DenseIndex, rrf_fuse
    from kb.retrieve import Embedder, HashingEmbedder, OnnxEmbedder, OnnxReranker

导入本包**不**会拉起 onnxruntime（`embedder.py` 里是延迟导入），所以
只用 BM25 的场景启动开销和 Phase 1 一样。
"""

from __future__ import annotations

from .dense import DenseIndex
from .embedder import Embedder, HashingEmbedder, OnnxEmbedder, OnnxReranker
from .hybrid import rrf_fuse
from .pipeline import Retriever, build_index, load_chunks
from .sparse import BM25Index
from .tokenize import tokenize, tokenize_counts
from .types import SearchHit

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "OnnxEmbedder",
    "OnnxReranker",
    "BM25Index",
    "DenseIndex",
    "Retriever",
    "SearchHit",
    "build_index",
    "load_chunks",
    "rrf_fuse",
    "tokenize",
    "tokenize_counts",
]
