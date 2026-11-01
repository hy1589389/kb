"""Phase 1 · 语料落库管道：遍历 → 解析 → 切块 → JSONL + stats.json。

这一阶段的唯一目标：把原始语料变成「结构化、可回链、可统计」的 chunk。
不涉及嵌入、不涉及向量库 —— 那些是 Phase 2 的事。

为什么先做这个？
  数据管道的质量决定整个系统的上限。检索效果差，九成时候不是模型不行，
  而是 chunk 里有徽章、有导航、有半截代码。先把这一层做扎实，
  后面调检索参数时才有干净的对照组。

## 多集合

每个集合（`[collections.<name>]`）有自己的 `raw_dir` 和 `index_dir`，遍历时
互不可见：

    data/raw/tech/JavaGuide/...   -> chunks_tech.jsonl   -> data/index/
    data/raw/papers/*.pdf         -> chunks_papers.jsonl -> data/index/papers/
    data/raw/profile/*.pdf        -> chunks_profile.jsonl-> data/index/profile/

`index_dir` 恰好是 `data/index`（即 tech）时，**额外**写一份历史的
`chunks_all.jsonl` / `chunks_<repo>.jsonl` / `manifest.jsonl` / `stats.json`，
保持建索引入口 `python ingest.py index --chunks data/processed/chunks_all.jsonl`
不变。别的集合绝不碰这些文件。

## 文件指纹幂等

每个文件记 `data/processed/fingerprints_<collection>.json`：

    {"<相对路径>": {"sha256": "...", "mtime": 1234567890, "chunks": 12}}

  · hash 未变 -> 跳过，打印「跳过（未变更）」，直接复用上次的块
  · hash 变了 -> 重新解析，替换该文件的块
  · 新文件     -> 新增
  · raw_dir 里已删除的文件 -> 块不再写入（同步删除）

指纹的 key 是**相对 raw_dir 的路径**，所以同一个 repo 下的同名文件不会撞。

## dry-run

`ingest --dry-run` 不写任何文件，逐个文件打印：块数、平均字符数、前 2 个块的
breadcrumb + 摘要，以及质量警告。指纹照样比对，只是不落盘。
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import (
    MARKDOWN_SUFFIXES,
    PARSER_AUTO,
    PARSER_MARKDOWN,
    PARSER_PDF,
    PDF_SUFFIXES,
    CollectionConfig,
    IngestConfig,
)
from .chunker import Chunk, chunk_document, make_chunk_id, make_doc_id
from .pdf import page_locator, parse_pdf
from .quality import QualityReport, assess_texts, mark_low_quality

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


def legacy_index_dir(cfg: IngestConfig) -> Path:
    """tech 沿用的旧索引目录（`data/index`），由 out_dir 的父目录推出来。

    用它判断某个集合要不要额外写 `chunks_all.jsonl` 这类历史文件：
    `index_dir == data/index` 的集合就是 tech。
    """
    return (Path(cfg.out_dir).parent / "index").resolve()


# --------------------------------------------------------------------------- #
# 结果结构
# --------------------------------------------------------------------------- #
@dataclass
class SkippedFile:
    repo: str
    rel_path: str
    reason: str
    collection: str = ""


@dataclass
class DocRecord:
    doc_id: str
    repo: str
    rel_path: str
    title: str
    n_chunks: int
    char_count: int
    collection: str = ""
    parser: str = ""
    low_quality: bool = False


@dataclass
class FileTask:
    """待处理的单个文件。"""

    abs_path: Path
    raw_rel: str      # 相对 raw_dir 的路径（指纹的 key）
    repo: str
    rel_path: str
    kind: str         # "markdown" / "pdf"


@dataclass
class FileReport:
    """单个文件的处理报告，dry-run 打印和日志都用它。"""

    collection: str
    raw_rel: str
    kind: str
    status: str        # 新增 / 变更 / 未变更
    n_chunks: int = 0
    avg_chars: int = 0
    n_pages: int = 0
    quality: QualityReport | None = None
    preview: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reason: str = ""   # 非空 = 解析失败 / 跳过原因


@dataclass
class IngestResult:
    collection: str = ""
    chunks: list[dict] = field(default_factory=list)
    docs: list[DocRecord] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)
    reports: list[FileReport] = field(default_factory=list)
    fingerprints: dict[str, dict] = field(default_factory=dict)
    unchanged: int = 0
    added: int = 0
    updated: int = 0
    removed: list[str] = field(default_factory=list)
    missing_raw_dir: bool = False

    @property
    def low_quality_docs(self) -> list[DocRecord]:
        return [d for d in self.docs if d.low_quality]


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


def _iter_tree(raw_dir: Path, cfg: IngestConfig):
    """递归遍历 raw_dir，产出 (绝对路径, 相对 raw_dir 的 posix 路径)。"""
    skip_dirs = set(cfg.skip_dirs)
    for root, dirs, files in os.walk(raw_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            abs_path = Path(root) / name
            yield abs_path, abs_path.relative_to(raw_dir).as_posix()


def iter_collection_files(raw_dir: Path, parser: str,
                          cfg: IngestConfig) -> list[FileTask]:
    """按集合的 parser 配置列出待处理文件。

    · markdown 集合沿用旧逻辑：只认 `.md/.markdown/.mdx`，`repo` = raw_dir 下的
      一级子目录名（raw_dir 根下的散文件不收，和以前一致）。
    · pdf 集合走平铺：raw_dir 下（含子目录）所有 `.pdf`，`repo` = 文件名（不含扩展名）。
    · auto 按扩展名分派：`.pdf` 同上；Markdown 若在一级子目录下则 `repo` 取子目录名，
      直接在根下的取文件名。
    """
    tasks: list[FileTask] = []

    if parser == PARSER_MARKDOWN:
        for repo_dir in discover_repos(raw_dir):
            for abs_path, rel_path in iter_markdown(repo_dir, cfg):
                tasks.append(
                    FileTask(abs_path, f"{repo_dir.name}/{rel_path}",
                             repo_dir.name, rel_path, "markdown")
                )
        return tasks

    if parser == PARSER_PDF:
        for abs_path, rel_path in _iter_tree(raw_dir, cfg):
            if not abs_path.name.lower().endswith(PDF_SUFFIXES):
                continue
            tasks.append(FileTask(abs_path, rel_path, abs_path.stem, rel_path, "pdf"))
        return tasks

    if parser != PARSER_AUTO:
        raise ValueError(f"未知的 parser：{parser!r}")

    skip_files = {s.lower() for s in cfg.skip_files}
    for abs_path, rel_path in _iter_tree(raw_dir, cfg):
        name = abs_path.name
        lower = name.lower()
        parts = Path(rel_path).parts
        if lower.endswith(PDF_SUFFIXES):
            tasks.append(FileTask(abs_path, rel_path, abs_path.stem, rel_path, "pdf"))
            continue
        if not lower.endswith(MARKDOWN_SUFFIXES) or lower in skip_files:
            continue
        if lower.startswith(("license", "changelog", "contributing", "code_of_conduct")):
            continue
        if len(parts) > 1:
            repo = parts[0]
            rel_to_repo = "/".join(parts[1:])
        else:
            repo = abs_path.stem
            rel_to_repo = name
        tasks.append(FileTask(abs_path, rel_path, repo, rel_to_repo, "markdown"))

    return tasks


# --------------------------------------------------------------------------- #
# 指纹
# --------------------------------------------------------------------------- #
def file_signature(path: Path) -> tuple[str, int]:
    """(sha256, mtime 秒)。按 1 MB 分块读，44 MB 的仓库也不会一口气进内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest(), int(Path(path).stat().st_mtime)


def fingerprints_path(out_dir: Path, collection: str) -> Path:
    return Path(out_dir) / f"fingerprints_{collection}.json"


def load_fingerprints(out_dir: Path, collection: str) -> dict[str, dict]:
    path = fingerprints_path(out_dir, collection)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_existing_chunks(out_dir: Path, collection: str) -> dict[tuple[str, str], list[dict]]:
    """读上一次的 `chunks_<collection>.jsonl`，按 `(repo, rel_path)` 分组。

    未变更的文件直接复用这些行，不再解析 —— 这是幂等的关键：跳过时不重新
    切块，chunk_id 就不会因为切块器的小改动而漂移。

    用 `(repo, rel_path)` 而不是「相对 raw_dir 的路径」做 key，是因为后者
    要从记录里反推：Markdown 的 rel_path 相对仓库、PDF 的 rel_path 相对
    raw_dir、auto 集合里根目录下的 Markdown 又是第三种情况，反推迟早出错。
    这两个字段是记录里原样存在的，直接拿来当 key 最稳。
    """
    path = Path(out_dir) / f"chunks_{collection}.jsonl"
    if not path.exists():
        return {}
    out: dict[tuple[str, str], list[dict]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # 半截行（上次写到一半崩了）直接丢，后面会重新解析
            key = (str(row.get("repo", "") or ""), str(row.get("rel_path", "") or ""))
            out.setdefault(key, []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: int(r.get("chunk_index", 0) or 0))
    return out


# --------------------------------------------------------------------------- #
# 单个文件 → chunk 记录
# --------------------------------------------------------------------------- #
def _chunk_params(cfg: IngestConfig) -> dict:
    return {
        "max_chars": cfg.chunk.max_chars,
        "min_chars": cfg.chunk.min_chars,
        "drop_below": cfg.chunk.drop_below,
        "overlap": cfg.chunk.overlap,
        "code_block_max_chars": cfg.chunk.code_block_max_chars,
    }


def _records(collection: str, repo: str, rel_path: str, title: str,
             chunks: list[Chunk], kind: str) -> list[dict]:
    """把 `Chunk` 列表转成 JSONL 记录（现有字段 + collection + locator）。"""
    doc_id = make_doc_id(repo, rel_path)
    out: list[dict] = []
    for c in chunks:
        if kind == "pdf":
            locator = page_locator(c.line_start, c.line_end)
        else:
            locator = f"L{c.line_start}-{c.line_end}"
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
                # PDF 没有行号，line_start/line_end 填页码（字段名保持不变）
                "line_start": c.line_start,
                "line_end": c.line_end,
                "char_count": len(c.text),
                "cjk_count": count_cjk(c.text),
                "est_tokens": estimate_tokens(c.text),
                # text 是原文正文，embed_text 是真正送去嵌入的内容（含面包屑）
                "text": c.text,
                "embed_text": c.embed_text,
                "collection": collection,
                "locator": locator,
            }
        )
    return out


def _doc_of(collection: str, repo: str, rel_path: str, title: str,
            rows: list[dict], kind: str) -> DocRecord:
    return DocRecord(
        doc_id=str(rows[0].get("doc_id", "")) if rows else make_doc_id(repo, rel_path),
        repo=repo,
        rel_path=rel_path,
        title=title or (str(rows[0].get("title", "")) if rows else ""),
        n_chunks=len(rows),
        char_count=sum(int(r.get("char_count", 0) or 0) for r in rows),
        collection=collection,
        parser=kind,
        low_quality=any(bool(r.get("low_quality")) for r in rows),
    )


def ingest_markdown_file(task: FileTask, cfg: IngestConfig,
                         collection: str) -> tuple[list[dict], str, str]:
    """Markdown 文件 -> (chunk 记录, 文档标题, 跳过原因)。"""
    try:
        raw = task.abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # 权限 / 文件被占用
        return [], "", f"read_error: {exc.__class__.__name__}"

    if not raw.strip():
        return [], "", "empty_file"

    title, chunks = chunk_document(raw, **_chunk_params(cfg))
    if not chunks:
        return [], title, "no_chunks_after_clean"
    rows = _records(collection, task.repo, task.rel_path, title, chunks, "markdown")
    return rows, title, ""


def ingest_pdf_file(task: FileTask, cfg: IngestConfig,
                    collection: str) -> tuple[list[dict], str, str, list[str], int]:
    """PDF 文件 -> (chunk 记录, 标题, 跳过原因, 备注, 页数)。"""
    parsed = parse_pdf(task.abs_path, **_chunk_params(cfg))
    if parsed.reason:
        return [], parsed.title, parsed.reason, list(parsed.notes), parsed.n_pages
    rows = _records(collection, task.repo, task.rel_path, parsed.title,
                    parsed.chunks, "pdf")
    return rows, parsed.title, "", list(parsed.notes), parsed.n_pages


def _assess(rows: list[dict]) -> QualityReport:
    return assess_texts([str(r.get("text", "")) for r in rows])


# --------------------------------------------------------------------------- #
# 主流程：一个集合
# --------------------------------------------------------------------------- #
def ingest_collection(cfg: IngestConfig, collection: CollectionConfig, *,
                      verbose: bool = True, dry_run: bool = False) -> IngestResult:
    """入库一个集合。dry_run=True 时只读不写。"""
    result = IngestResult(collection=collection.name)
    raw_dir = Path(collection.raw_dir)
    if not raw_dir.exists():
        result.missing_raw_dir = True
        if verbose:
            print(f"  [!] 集合 {collection.name} 的 raw_dir 不存在：{raw_dir}")
        return result

    old_fingerprints = load_fingerprints(cfg.out_dir, collection.name)
    old_chunks = load_existing_chunks(cfg.out_dir, collection.name)
    tasks = iter_collection_files(raw_dir, collection.parser, cfg)
    seen: set[str] = set()

    for task in tasks:
        seen.add(task.raw_rel)
        prev = old_fingerprints.get(task.raw_rel) or {}
        try:
            digest, mtime = file_signature(task.abs_path)
        except OSError as exc:
            result.skipped.append(SkippedFile(task.repo, task.rel_path,
                                              f"read_error: {exc.__class__.__name__}",
                                              collection.name))
            continue

        unchanged = (prev.get("sha256") == digest) and (
            (task.repo, task.rel_path) in old_chunks)
        status = "未变更" if unchanged else ("变更" if prev else "新增")

        if unchanged and not dry_run:
            rows = old_chunks[(task.repo, task.rel_path)]
            result.chunks.extend(rows)
            result.docs.append(_doc_of(collection.name, task.repo, task.rel_path,
                                       str(rows[0].get("title", "")), rows, task.kind))
            result.fingerprints[task.raw_rel] = {
                "sha256": digest, "mtime": mtime, "chunks": len(rows),
            }
            result.unchanged += 1
            if verbose:
                print(f"  [跳过（未变更）] {task.raw_rel}")
            continue

        n_pages = 0
        if task.kind == "pdf":
            rows, title, reason, notes, n_pages = ingest_pdf_file(task, cfg, collection.name)
        else:
            rows, title, reason = ingest_markdown_file(task, cfg, collection.name)
            notes = []

        if reason:
            result.skipped.append(SkippedFile(task.repo, task.rel_path, reason,
                                              collection.name))
            result.reports.append(FileReport(
                collection=collection.name, raw_rel=task.raw_rel, kind=task.kind,
                status=status, n_pages=n_pages, notes=notes, reason=reason,
            ))
            if verbose:
                print(f"  [跳过] {task.raw_rel}：{reason}")
            continue

        report = _assess(rows)
        mark_low_quality(rows, report.low_quality)
        doc = _doc_of(collection.name, task.repo, task.rel_path, title, rows, task.kind)
        result.chunks.extend(rows)
        result.docs.append(doc)
        result.fingerprints[task.raw_rel] = {
            "sha256": digest, "mtime": mtime, "chunks": len(rows),
        }
        if unchanged:
            result.unchanged += 1
        elif prev:
            result.updated += 1
        else:
            result.added += 1

        result.reports.append(FileReport(
            collection=collection.name, raw_rel=task.raw_rel, kind=task.kind,
            status=status, n_chunks=len(rows),
            avg_chars=int(round(report.avg_chars)), n_pages=n_pages,
            quality=report, preview=rows[:2], notes=notes,
        ))
        if verbose:
            print(f"  [{status}] {task.raw_rel}  块 {len(rows)}  "
                  f"平均 {report.avg_chars:.0f} 字符"
                  + (f"  {n_pages} 页" if n_pages else ""))
            if report.low_quality:
                print(f"      [!] 质量警告：{'；'.join(report.reasons)}")

    # 源文件已删除 -> 它的块不再写入（同步删除）
    for raw_rel in sorted(set(old_fingerprints) - seen):
        result.removed.append(raw_rel)
        if verbose:
            print(f"  [移除] {raw_rel}（源文件已删除）")

    return result


def run_ingest(cfg: IngestConfig, *, collections: list[str] | None = None,
               verbose: bool = True, dry_run: bool = False) -> list[IngestResult]:
    """按配置入库若干集合，一个集合一个 `IngestResult`。

    `collections=None` 时用 `[collections].default_search`。旧调用
    （`run_ingest(cfg)`）在没配 `[collections]` 时就是「只有 tech」，行为不变。
    """
    results: list[IngestResult] = []
    for collection in cfg.collections.select(collections):
        if verbose:
            print(f"  集合 {collection.name}  parser={collection.parser}  "
                  f"raw_dir={collection.raw_dir}")
        results.append(
            ingest_collection(cfg, collection, verbose=verbose, dry_run=dry_run)
        )
    return results


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
        "collection": result.collection,
        "source": str(cfg.collections.get(result.collection).raw_dir)
        if result.collection in cfg.collections.items else str(cfg.raw_dir),
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
        "files": {
            "added": result.added,
            "updated": result.updated,
            "unchanged": result.unchanged,
            "removed": len(result.removed),
            "skipped": len(result.skipped),
        },
        "low_quality_docs": [asdict(d) for d in result.low_quality_docs],
        "removed_files": list(result.removed),
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


def _summarize_text(text: str, width: int = 90) -> str:
    flat = " ".join((text or "").split())
    if len(flat) <= width:
        return flat
    return flat[:width] + f"... [共 {len(text)} 字符]"


def print_reports(result: IngestResult, *, preview: int = 2,
                  width: int = 90) -> None:
    """dry-run 预览：每个文件的块数、平均字符数、前 N 块、质量警告。"""
    if not result.reports:
        return
    print("-" * 68)
    print("  预览（dry-run，不写文件）")
    print("-" * 68)
    for report in result.reports:
        if report.reason:
            print(f"  [{report.status}] {report.raw_rel}")
            print(f"      [X] 跳过原因：{report.reason}")
            continue
        head = (f"  [{report.status}] {report.raw_rel}  块 {report.n_chunks}  "
                f"平均 {report.avg_chars} 字符")
        if report.n_pages:
            head += f"  {report.n_pages} 页"
        print(head)
        if report.notes:
            print(f"      解析: {' · '.join(report.notes)}")
        if report.quality is not None:
            print(f"      质量: {report.quality.summary()}")
        for i, row in enumerate(report.preview[:preview], start=1):
            crumb = str(row.get("breadcrumb") or "(无标题)")
            loc = str(row.get("locator") or "")
            print(f"      [{i}] {crumb}" + (f"   ({loc})" if loc else ""))
            print(f"          {_summarize_text(str(row.get('text', '')), width)}")


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(result: IngestResult, cfg: IngestConfig,
                  stats: dict) -> dict[str, Path]:
    """落盘。返回 {名字: 路径}，CLI 直接打印。

    `collection.index_dir == data/index`（tech）时，额外写历史的
    `chunks_all.jsonl` / `chunks_<repo>.jsonl` / `manifest.jsonl` / `stats.json`，
    保证「建索引 → chunks_all.jsonl」这条入口不变。
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    name = result.collection or "tech"
    written: dict[str, Path] = {}

    chunks_path = cfg.out_dir / f"chunks_{name}.jsonl"
    _write_jsonl(chunks_path, result.chunks)
    written["chunks"] = chunks_path

    manifest_path = cfg.out_dir / f"manifest_{name}.jsonl"
    _write_jsonl(manifest_path, (asdict(doc) for doc in result.docs))
    written["manifest"] = manifest_path

    fingerprints_path(cfg.out_dir, name).write_text(
        json.dumps(result.fingerprints, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written["fingerprints"] = fingerprints_path(cfg.out_dir, name)

    stats_path = cfg.out_dir / f"stats_{name}.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    written["stats"] = stats_path

    collection = cfg.collections.items.get(name)
    is_legacy = collection is not None and \
        Path(collection.index_dir).resolve() == legacy_index_dir(cfg)
    if not is_legacy:
        return written

    # ---- 以下四个文件只为 tech（data/index）保留，别的集合绝不写 ----
    by_repo: dict[str, list[dict]] = {}
    for c in result.chunks:
        by_repo.setdefault(c["repo"], []).append(c)
    for repo, rows in by_repo.items():
        path = cfg.out_dir / f"chunks_{repo}.jsonl"
        _write_jsonl(path, rows)
        written[f"chunks_{repo}"] = path

    all_path = cfg.out_dir / "chunks_all.jsonl"
    _write_jsonl(all_path, result.chunks)
    written["chunks_all"] = all_path

    legacy_manifest = cfg.out_dir / "manifest.jsonl"
    _write_jsonl(legacy_manifest, (asdict(doc) for doc in result.docs))
    written["legacy_manifest"] = legacy_manifest

    legacy_stats = cfg.out_dir / "stats.json"
    legacy_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    written["legacy_stats"] = legacy_stats
    return written
