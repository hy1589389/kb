"""语料入库子包：清洗 → 结构感知切块（Markdown / PDF）→ 落盘。"""

from .chunker import (Chunk, Section, chunk_document, chunk_sections,
                      make_chunk_id, make_doc_id)
from .pdf import PdfParseResult, parse_pdf
from .pipeline import (DocRecord, FileTask, IngestResult, build_stats,
                       ingest_collection, load_fingerprints, run_ingest,
                       write_outputs)
from .quality import QualityReport, assess_texts, garbled_ratio

__all__ = [
    "Chunk",
    "Section",
    "chunk_document",
    "chunk_sections",
    "make_chunk_id",
    "make_doc_id",
    "parse_pdf",
    "PdfParseResult",
    "DocRecord",
    "FileTask",
    "IngestResult",
    "run_ingest",
    "ingest_collection",
    "load_fingerprints",
    "build_stats",
    "write_outputs",
    "QualityReport",
    "assess_texts",
    "garbled_ratio",
]
