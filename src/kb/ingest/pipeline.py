"""Phase 1 · 语料落库管道：遍历 → 解析 → 切块 → JSONL + stats.json。

这一阶段的唯一目标：把原始语料变成「结构化、可回链、可统计」的 chunk。
不涉及嵌入、不涉及向量库 —— 那些是 Phase 2 的事。

为什么先做这个？
  数据管道的质量决定整个系统的上限。检索效果差，九成时候不是模型不行，
  而是 chunk 里有徽章、有导航、有半截代码。先把这一层做扎实，
  后面调检索参数时才有干净的对照组。
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import MARKDOWN_SUFFIXES, IngestConfig
from .chunker import chunk_document, make_chunk_id, make_doc_id

# --------------------------------------------------------------------------- #
# 统计口径
# --------------------------------------------------------------------------- #
_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK 统一表意文字
    (0x3400, 0x4DBF),  # 扩展 A
    (0x3000, 0x303F),  # CJK 标点
    (0xFF00, 0xFFEF),  # 全角字符
)


def count_cjk(text: str) -> int:
    return sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES))


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文约 1 字 1 token，英文约 3.5 字符 1 token。

    只用于「语料规模」这一层沟通，不用于计费。
    """
    cjk = count_cjk(text)
    other = len(text) - cjk
    return int(cjk + other / 3.5)


# --------------------------------------------------------------------------- #
# 结果结构
# --------------------------------------------------------------------------- #
@dataclass
class SkippedFile:
    repo: str
    rel_path: str
    reason: str


@dataclass
class DocRecord:
    doc_id: str
    repo: str
    rel_path: str
    title: str
    n_chunks: int
    char_count: int


@dataclass
class IngestResult:
    chunks: list[dict] = field(default_factory=list)
    docs: list[DocRecord] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 遍历
# --------------------------------------------------------------------------- #
def discover_repos(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def iter_markdown(repo_dir: Path, cfg: IngestConfig):
    """遍历一个仓库下的所有 Markdown 文件，产出 (绝对路径, 相对仓库路径)。"""
    skip_dirs = set(cfg.skip_dirs)
    skip_files = {s.lower() for s in cfg.skip_files}

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for name in files:
            lower = name.lower()
            if not lower.endswith(MARKDOWN_SUFFIXES):
                continue
            if lower in skip_files:
                continue
            if lower.startswith(("license", "changelog", "contributing", "code_of_conduct")):
                continue
            abs_path = Path(root) / name
            yield abs_path, abs_path.relative_to(repo_dir).as_posix()


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def ingest_file(repo: str, abs_path: Path, rel_path: str, cfg: IngestConfig) -> tuple[list[dict], DocRecord | None, str]:
    """处理单个文件，返回 (chunk 字典列表, 文档记录或 None, 跳过原因)。"""
    try:
        raw = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # 权限 / 文件被占用
        return [], None, f"read_error: {exc.__class__.__name__}"

    if not raw.strip():
        return [], None, "empty_file"

    title, chunks = chunk_document(
        raw,
        max_chars=cfg.chunk.max_chars,
        min_chars=cfg.chunk.min_chars,
        drop_below=cfg.chunk.drop_below,
        overlap=cfg.chunk.overlap,
        code_block_max_chars=cfg.chunk.code_block_max_chars,
    )
    if not chunks:
        return [], None, "no_chunks_after_clean"

    doc_id = make_doc_id(repo, rel_path)
    out: list[dict] = []
    for c in chunks:
        out.append(
            {
                "chunk_id": make_chunk_id(doc_id, c.index, c.text),
                "doc_id": doc_id,
                "repo": repo,
                "rel_path": rel_path,
                "title": title,
                "section_path": list(c.path),
                "breadcrumb": c.breadcrumb,
                "part": c.part,
                "n_parts": c.n_parts,
                "merged_paths": [list(p) for p in c.merged_paths],
                "chunk_index": c.index,
                "line_start": c.line_start,
                "line_end": c.line_end,
                "char_count": len(c.text),
                "cjk_count": count_cjk(c.text),
                "est_tokens": estimate_tokens(c.text),
                # text 是原文正文，embed_text 是真正送去嵌入的内容（含面包屑）
                "text": c.text,
                "embed_text": c.embed_text,
            }
        )

    total_chars = sum(x["char_count"] for x in out)
    doc = DocRecord(
        doc_id=doc_id,
        repo=repo,
        rel_path=rel_path,
        title=title,
        n_chunks=len(out),
        char_count=total_chars,
    )
    return out, doc, ""


def run_ingest(cfg: IngestConfig, *, verbose: bool = True) -> IngestResult:
    result = IngestResult()
    repos = discover_repos(cfg.raw_dir)

    if not repos:
        if verbose:
            print(f"[!] {cfg.raw_dir} 下没有找到任何语料仓库。")
            print("    先执行 scripts/bootstrap.ps1，或手动 git clone 到该目录。")
        return result

    for repo_dir in repos:
        repo = repo_dir.name
        n_files = n_ok = n_skip = 0
        for abs_path, rel_path in iter_markdown(repo_dir, cfg):
            n_files += 1
            chunks, doc, reason = ingest_file(repo, abs_path, rel_path, cfg)
            if doc is None:
                n_skip += 1
                result.skipped.append(SkippedFile(repo=repo, rel_path=rel_path, reason=reason))
                continue
            result.chunks.extend(chunks)
            result.docs.append(doc)
            n_ok += 1
        if verbose:
            print(f"  [{repo:<16}] 文件 {n_files:>5}  入库 {n_ok:>5}  跳过 {n_skip:>4}")
    return result


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(int(len(ordered) * q), len(ordered) - 1)
    return ordered[idx]


def build_stats(result: IngestResult, cfg: IngestConfig, elapsed: float) -> dict:
    chunks = result.chunks
    sizes = [c["char_count"] for c in chunks]
    all_text_len = sum(sizes)

    per_repo: dict[str, dict] = {}
    for doc in result.docs:
        r = per_repo.setdefault(doc.repo, {"docs": 0, "chunks": 0, "chars": 0})
        r["docs"] += 1
        r["chunks"] += doc.n_chunks
        r["chars"] += doc.char_count

    top_docs = sorted(result.docs, key=lambda d: d.char_count, reverse=True)[:10]

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(elapsed, 2),
        "source": str(cfg.raw_dir),
        "chunk_config": asdict(cfg.chunk),
        "totals": {
            "repos": len(per_repo),
            "docs": len(result.docs),
            "chunks": len(chunks),
            "chars": all_text_len,
            "cjk_chars": sum(c["cjk_count"] for c in chunks),
            "est_tokens": sum(c["est_tokens"] for c in chunks),
        },
        "chunk_size": {
            "min": min(sizes) if sizes else 0,
            "p50": _percentile(sizes, 0.50),
            "p90": _percentile(sizes, 0.90),
            "p95": _percentile(sizes, 0.95),
            "max": max(sizes) if sizes else 0,
            "mean": round(statistics.fmean(sizes), 1) if sizes else 0,
        },
        "per_repo": per_repo,
        "top_docs_by_chars": [asdict(d) for d in top_docs],
        "skipped_summary": _summarize_skipped(result),
        "skipped_examples": [asdict(s) for s in result.skipped[:20]],
    }


def _summarize_skipped(result: IngestResult) -> dict[str, int]:
    summary: dict[str, int] = {}
    for s in result.skipped:
        key = s.reason.split(":")[0]
        summary[key] = summary.get(key, 0) + 1
    return summary


def write_outputs(result: IngestResult, cfg: IngestConfig, stats: dict) -> dict[str, Path]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # 按仓库分片写 JSONL，方便后续做增量索引（只重跑变动的仓库）
    by_repo: dict[str, list[dict]] = {}
    for c in result.chunks:
        by_repo.setdefault(c["repo"], []).append(c)

    for repo, rows in by_repo.items():
        path = cfg.out_dir / f"chunks_{repo}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        written[f"chunks_{repo}"] = path

    # 全量文件（跨仓库检索用）
    all_path = cfg.out_dir / "chunks_all.jsonl"
    with all_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in result.chunks:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    written["chunks_all"] = all_path

    manifest_path = cfg.out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        for doc in result.docs:
            fh.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    written["manifest"] = manifest_path

    stats_path = cfg.out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    written["stats"] = stats_path

    return written
