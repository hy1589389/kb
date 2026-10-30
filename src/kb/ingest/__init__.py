"""语料入库子包：清洗 → 结构感知切块 → 落盘。"""

from .chunker import Chunk, Section, chunk_document, make_chunk_id, make_doc_id
from .pipeline import IngestResult, build_stats, run_ingest, write_outputs

__all__ = [
    "Chunk",
    "Section",
    "chunk_document",
    "make_chunk_id",
    "make_doc_id",
    "IngestResult",
    "run_ingest",
    "build_stats",
    "write_outputs",
]
