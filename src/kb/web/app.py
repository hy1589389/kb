"""FastAPI 应用工厂：检索 / 问答 / 编辑器回链。

设计上只有一个硬约束：**retriever 和 answerer 从外面传进来**。加载一次索引
要 6~70 秒（18k 块的元数据 + BM25 倒排 + 2 GB 模型的首次前向），每个请求
重建一次就等于服务不可用。`create_app` 只保存引用，不构造、不复制；路由里
用的永远是同一个对象，所以 `python ingest.py web` 启动一次之后，单次检索
只剩「编码 query + 矩阵乘 + BM25」的时间。

路由：

    GET  /                 单文件前端
    GET  /api/search       q / top_k / mode(hybrid|dense|sparse) / collections
    POST /api/ask          {"q": ..., "top_k": ...} -> Answer.to_dict()
    GET  /api/open         chunk_id -> 编辑器打开源文件
    GET  /api/collections  集合列表（名字 / 块数 / 有没有索引）+ 默认选中的集合
    GET  /api/health       块数 / embedder / 就绪状态 / 集合列表
    GET  /api/stats        索引统计（按集合分组 + 顶层汇总）
    GET  /api/review/today 今日复习：按日期挑一道评测失败题 + 参考原文

## 多集合

`Retriever` 可以是多索引模式（`{集合名: 索引目录}`），此时一个查询会先在
每个集合内部各自融合，再按集合权重做一次跨集合 RRF。Web 层要做的只有两件事：

1. 从 retriever 上把「集合名 -> 索引目录」拿出来（`_index_dir_map`），读各自的
   `index_meta.json` 得到块数，`create_app` 时缓存进 `app.state.collections`；
2. `/api/search` 把 `collections` 参数校验后原样透传给 `Retriever.search`，
   结果里每条带 `collection` 字段。

集合列表和块数**只在 create_app 时读一次**：`/api/collections` 每个请求都只是
把 `app.state.collections` 抄一遍。索引目录是启动后就不再变的东西，重算没有
意义，还会让一个前端 chip 的渲染变成 2 次磁盘读。

`/api/stats` 要读一遍 24 MB 的 `chunks_meta.jsonl`，所以也只在 `create_app`
时算一次（多集合时每个集合算一次），结果缓存在 `app.state.stats`；之后每个
请求只是把缓存抄一遍。这是启动期的一次性成本（真实索引 18276 行约 0.3 秒），
换掉的是「每次有人点统计按钮就阻塞事件循环扫一遍文件」。

`/api/open` 需要知道 chunk 的 repo/rel_path/行号。`Retriever` 自己持有全部
元数据（`.meta`），这里做两件事：优先查「本次进程里见过的检索结果」缓存，
查不到再按 chunk_id 扫一遍 retriever.meta 建一次索引。缓存有上限，避免
长时间运行的进程把每个请求的正文都攒在内存里。
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..generate.llm import LLMError
from .editor import detect_editor, open_in_editor, resolve_source_path
from .review import REVIEW_MAX_CHUNKS, load_review_bank, parse_review_day

__all__ = ["create_app", "STATIC_DIR", "TEXT_PREVIEW_CHARS"]

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# `/api/search` 里每条结果的正文上限。前端「展开全文」用的是 text_full。
TEXT_PREVIEW_CHARS = 400

MAX_TOP_K = 100

# 见过的检索结果上限。超过就整体清空而不是做 LRU：这里存的是元数据 +
# 正文，单条平均 1 KB，5000 条约 5 MB，清空的代价只是「旧 chunk 要重新检索
# 一次才能打开」，不影响正确性。
_HIT_CACHE_LIMIT = 5000


class AskRequest(BaseModel):
    """`POST /api/ask` 的请求体。字段都有默认值，缺省也能进来再报错。"""

    q: str = ""
    top_k: int = 5
    collections: list[str] | None = None


def _as_dict(hit: Any) -> dict:
    """`SearchHit` / dict / 其它鸭子类型统一成 dict。"""
    if isinstance(hit, dict):
        return dict(hit)
    to_dict = getattr(hit, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {name: getattr(hit, name, None) for name in (
        "chunk_id", "repo", "rel_path", "breadcrumb",
        "line_start", "line_end", "text", "score", "source", "rank",
        "collection", "locator",
    )}


def _mode_flags(mode: str) -> tuple[bool, bool]:
    """`mode` -> (use_dense, use_sparse)。非法值直接 400。"""
    normalized = (mode or "hybrid").strip().lower()
    if normalized == "hybrid":
        return True, True
    if normalized == "dense":
        return True, False
    if normalized == "sparse":
        return False, True
    raise HTTPException(
        status_code=400,
        detail=f"未知的 mode：{mode!r}（可选 hybrid / dense / sparse）",
    )


def _count_chunks(retriever: Any) -> int:
    """块数：索引元数据 > 加载的 meta > 稠密索引行数。"""
    info = getattr(retriever, "index_info", None) or {}
    for value in (info.get("n_chunks"), info.get("n_docs")):
        try:
            if value:
                return int(value)
        except (TypeError, ValueError):
            pass
    meta = getattr(retriever, "meta", None)
    if meta:
        return len(meta)
    dense = getattr(retriever, "dense", None)
    return int(getattr(dense, "n_docs", 0) or 0)


def _int_or(value: Any, default: int = 0) -> int:
    """`int()` 的宽容版：索引里的 char_count 可能是 None / 字符串 / 浮点。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iter_chunk_meta(path: Path):
    """逐行产出 `chunks_meta.jsonl` 里的 dict；空行和坏行直接跳过。

    单独抽成生成器有两个好处：24 MB 的文件不会一次性读进内存；测试可以
    mock 它来数「到底扫了几遍索引」。
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def _dir_size_mb(path: Path, skip: tuple[Path, ...] = ()) -> float:
    """目录总大小（MiB，保留 1 位）。单个文件 stat 失败不影响整体。

    `skip` 是「已经算进别的集合」的子目录。真实索引就是这种情况：`tech` 的
    索引目录是 `data/index`，而 `profile` 在 `data/index/profile` 下，不排掉
    的话顶层汇总会把 profile 的大小算两遍。
    """
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if not item.is_file():
                    continue
                if any(root != path and item.is_relative_to(root) for root in skip):
                    continue
                total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0.0
    return round(total / (1024 * 1024), 1)


# 没有 repo 字段的块归到这个桶里，保证「各仓库块数之和 == chunks」恒成立
_UNKNOWN_REPO = "(未标注仓库)"


def _safe_path(value: Any) -> Path | None:
    """任意值 -> Path；空值 / 非法路径返回 None（索引目录来自配置，可能写错）。"""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return Path(text)
    except (OSError, ValueError):
        return None


def _read_index_meta(index_dir: Path | None) -> dict | None:
    """读 `index_meta.json`；读不到或不是 JSON 对象都返回 None（当作没索引）。"""
    if index_dir is None:
        return None
    try:
        data = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _index_dir_map(retriever: Any) -> dict[str, Path | None]:
    """`{集合名: 索引目录}`，多集合模式下就是 retriever 真正加载的那几份。

    取值顺序：`retriever.index_dirs`（多索引模式）> `retriever.parts`（内部
    加载结果）> `retriever.index_dir`（单索引兼容）。单索引且没有集合名时
    键是空串 —— 这与 `Retriever._select_parts` 里「名字是空串 = 未指定」的
    约定一致，任何集合名都会被当成「查全部」。
    """
    out: dict[str, Path | None] = {}

    raw = getattr(retriever, "index_dirs", None)
    if isinstance(raw, Mapping):
        for name, path in raw.items():
            out[str(name)] = _safe_path(path)
        if out:
            return out

    parts = getattr(retriever, "parts", None)
    if isinstance(parts, Mapping) and parts:
        for name, part in parts.items():
            out[str(name)] = _safe_path(getattr(part, "index_dir", None))
        if out:
            return out

    names: list[str] = []
    getter = getattr(retriever, "collection_names", None)
    if callable(getter):
        try:
            names = [str(name) for name in getter() if str(name)]
        except (TypeError, ValueError, AttributeError):
            names = []
    out[names[0] if names else ""] = _safe_path(getattr(retriever, "index_dir", None))
    return out


def _collection_rows(retriever: Any, name: str, *, single: bool) -> list:
    """某个集合已经加载到内存里的块元数据（读不到索引文件时的退路）。"""
    parts = getattr(retriever, "parts", None)
    if isinstance(parts, Mapping) and name in parts:
        rows = getattr(parts[name], "meta", None)
        if rows:
            return list(rows)
    if single:
        return list(getattr(retriever, "meta", None) or [])
    return []


def _collection_count(retriever: Any, name: str, *, single: bool) -> int:
    """某个集合的块数退路：part.n_docs > 索引元数据 > 内存 meta 行数。"""
    parts = getattr(retriever, "parts", None)
    if isinstance(parts, Mapping) and name in parts:
        try:
            number = int(getattr(parts[name], "n_docs", 0) or 0)
        except (TypeError, ValueError):
            number = 0
        if number:
            return number
    return _count_chunks(retriever) if single else 0


def _collection_entries(retriever: Any, dirs: Mapping[str, Path | None]) -> list[dict]:
    """`/api/collections` 的集合列表：块数取自各索引的 `index_meta.json`。

    只在 `create_app` 里调用一次。索引文件读不到时 `has_index=False`，块数
    退回内存里的元数据（单集合模式），不抛异常 —— 索引目录半残缺时服务
    仍然要能起来，前端至少能显示「这个集合没有索引」。
    """
    single = len(dirs) == 1
    entries: list[dict] = []
    for name, index_dir in dirs.items():
        info = _read_index_meta(index_dir)
        has_index = info is not None
        chunks = _int_or((info or {}).get("n_chunks"))
        if not chunks:
            chunks = _collection_count(retriever, name, single=single)
        entries.append({
            "name": str(name),
            "chunks": int(chunks),
            "has_index": bool(has_index),
        })
    return entries


def _one_collection_stats(retriever: Any, name: str, index_dir: Path | None, *,
                          single: bool, skip: tuple[Path, ...] = ()) -> dict:
    """单个集合的统计。返回体里多一个 `total_chars`，给顶层汇总算平均用。"""
    counts: dict[str, int] = {}
    chunks = 0
    total_chars = 0

    meta_path = index_dir / "chunks_meta.jsonl" if index_dir is not None else None
    if meta_path is not None and meta_path.is_file():
        try:
            for row in _iter_chunk_meta(meta_path):
                chunks += 1
                repo = str(row.get("repo") or "").strip() or _UNKNOWN_REPO
                counts[repo] = counts.get(repo, 0) + 1
                size = row.get("char_count")
                if size is None:
                    size = len(str(row.get("text") or ""))
                total_chars += max(0, _int_or(size))
        except OSError:
            counts.clear()
            chunks = 0
            total_chars = 0

    if not chunks:
        # 假 retriever / 索引目录不完整时的退路：不读文件，用已经加载的元数据
        for item in _collection_rows(retriever, name, single=single):
            data = _as_dict(item)
            repo = str(data.get("repo") or "").strip() or _UNKNOWN_REPO
            counts[repo] = counts.get(repo, 0) + 1
            size = data.get("char_count")
            if size is None:
                size = len(str(data.get("text") or ""))
            total_chars += max(0, _int_or(size))
            chunks += 1
        if not chunks:
            chunks = _collection_count(retriever, name, single=single)
        if chunks and not counts:
            counts[_UNKNOWN_REPO] = chunks

    repos = [
        {"name": repo, "chunks": number}
        for repo, number in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return {
        "name": str(name),
        "chunks": chunks,
        "repos": repos,
        "avg_chars": int(round(total_chars / chunks)) if chunks else 0,
        "index_dir": str(index_dir) if index_dir is not None else "",
        "index_size_mb": _dir_size_mb(index_dir, skip) if (
            index_dir is not None and index_dir.is_dir()) else 0.0,
        "total_chars": total_chars,
    }


def _build_stats(retriever: Any, dirs: Mapping[str, Path | None] | None = None) -> dict:
    """算出 `/api/stats` 的响应体：按集合分组 + 旧版顶层汇总字段。

    只在 `create_app` 里调用一次。`chunks_meta.jsonl` 是主数据源；读不到时
    退回 retriever 内存里的 `meta`，再不行用 `_count_chunks`。任何一步失败
    都只是让统计变粗，不能让服务起不来。

    顶层 `chunks` / `repos` / `avg_chars` / `index_size_mb` / `index_dir` 是
    多集合之前就有的字段，一条都不能少（旧前端和旧测试都读它们），取值是
    所有集合的汇总；新增的 `collections` 才是分组明细。
    """
    if dirs is None:
        dirs = _index_dir_map(retriever)
    single = len(dirs) == 1

    entries: list[dict] = []
    repo_totals: dict[str, int] = {}
    total_chunks = 0
    total_chars = 0
    total_size = 0.0

    for name, index_dir in dirs.items():
        # 别的集合的索引目录若在这个目录下面，算大小时要排掉（避免重复计）
        skip = tuple(
            path for other, path in dirs.items()
            if other != name and path is not None and index_dir is not None
            and path != index_dir and path.is_relative_to(index_dir)
        )
        entry = _one_collection_stats(retriever, name, index_dir,
                                      single=single, skip=skip)
        total_chars += entry.pop("total_chars")
        total_chunks += entry["chunks"]
        total_size += entry["index_size_mb"]
        entries.append(entry)
        for repo in entry["repos"]:
            repo_totals[repo["name"]] = repo_totals.get(repo["name"], 0) + repo["chunks"]

    repos = [
        {"name": repo, "chunks": number}
        for repo, number in sorted(repo_totals.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    first_dir = next((path for path in dirs.values() if path is not None), None)
    return {
        "total_chunks": total_chunks,
        "chunks": total_chunks,
        "repos": repos,
        "avg_chars": int(round(total_chars / total_chunks)) if total_chunks else 0,
        "index_dir": str(first_dir) if first_dir is not None else "",
        "index_size_mb": round(total_size, 1),
        "collections": entries,
    }


def create_app(retriever: Any, answerer: Any = None, *,
               raw_dir: Path | str, editor: str | None = None,
               default_collections: list[str] | None = None,
               review_file: Path | str | None = None) -> FastAPI:
    """构造 FastAPI 应用。

    参数
    ----
    retriever : 任何有 `search(query, top_k=..., use_dense=..., use_sparse=...,
                 collections=...)` 的对象。**必须是已经构造好的实例**，本函数
                 不负责加载。多索引模式下它的 `index_dirs` 决定有哪些集合。
    answerer  : `Answerer` 或鸭子类型（有 `answer(question, top_k=...)`）。
                为 None 时 `/api/ask` 返回 503，检索照常可用。
    raw_dir   : 语料仓库根目录（`<raw_dir>/<collection>/<repo>/<rel_path>`）。
    editor    : 'code' / 'notepad' / 可执行文件路径；None 表示启动时自动探测一次。
    default_collections : `/api/search` 不传 `collections` 时查哪些集合。留空
                 等于「全部已加载的集合」；传了不存在的名字会被忽略。
    review_file : 评测结果 JSON（`results/run_hybrid.json`），「今日复习」的题库。
                 不传 / 读不出来时该入口返回 `available=false` 和一句原因，
                 其它功能不受影响。
    """
    if retriever is None:
        raise ValueError("create_app 需要已经构造好的 retriever，不能传 None")

    raw_root = Path(raw_dir).resolve()
    # 探测器只跑一次：每次请求都 which 一遍没必要，结果在整个进程里都一样。
    active_editor = (editor or "").strip() or detect_editor()

    # 集合信息只在启动时读一次：名字 -> 索引目录 -> index_meta.json 里的块数
    collection_dirs = _index_dir_map(retriever)
    collections = _collection_entries(retriever, collection_dirs)
    collection_names = [entry["name"] for entry in collections]
    wanted = [str(name) for name in (default_collections or [])
              if str(name) in collection_names]
    defaults = wanted or list(collection_names)

    app = FastAPI(
        title="kb · 本地知识库",
        version="0.1.0",
        description="检索 / 问答 / 在编辑器里打开源文件",
    )
    # 放在 state 里，测试和调试时能直接确认「还是同一个对象」
    app.state.retriever = retriever
    app.state.answerer = answerer
    app.state.raw_dir = raw_root
    app.state.editor = active_editor
    # 一次算好；`/api/collections` / `/api/search` 只读这些缓存，不再碰磁盘
    app.state.collections = collections
    app.state.collection_names = collection_names
    app.state.collection_dirs = collection_dirs
    app.state.default_collections = defaults
    # 一次算好；`/api/stats` 只读这个缓存，不再碰磁盘
    app.state.stats = _build_stats(retriever, collection_dirs)
    # 「今日复习」题库：评测结果只在这里读一遍，之后每个请求只是算日期取模
    app.state.review = load_review_bank(review_file)

    hit_cache: dict[str, dict] = {}
    meta_index: dict[str, dict] | None = None

    def remember(hit: Any) -> dict:
        data = _as_dict(hit)
        chunk_id = str(data.get("chunk_id") or "")
        if chunk_id:
            if len(hit_cache) >= _HIT_CACHE_LIMIT:
                hit_cache.clear()
            hit_cache[chunk_id] = data
        return data

    def from_meta(chunk_id: str) -> dict | None:
        nonlocal meta_index
        if meta_index is None:
            meta_index = {}
            for item in getattr(retriever, "meta", None) or []:
                data = _as_dict(item)
                key = str(data.get("chunk_id") or "")
                if key:
                    meta_index[key] = data
        return meta_index.get(chunk_id)

    def lookup(chunk_id: str) -> dict | None:
        return hit_cache.get(chunk_id) or from_meta(chunk_id)

    def to_result(hit: Any, fallback_rank: int) -> dict:
        data = remember(hit)
        text = str(data.get("text") or "")
        truncated = len(text) > TEXT_PREVIEW_CHARS
        path = resolve_source_path(data, raw_root)
        return {
            "chunk_id": str(data.get("chunk_id") or ""),
            "rank": int(data.get("rank") or fallback_rank),
            "score": float(data.get("score") or 0.0),
            # 多集合：这条结果来自哪个集合。单索引老索引里可能是空串。
            "collection": str(data.get("collection") or ""),
            "breadcrumb": str(data.get("breadcrumb") or ""),
            "repo": str(data.get("repo") or ""),
            "rel_path": str(data.get("rel_path") or ""),
            "line_start": int(data.get("line_start") or 0),
            "line_end": int(data.get("line_end") or 0),
            # Markdown 是 L222-297，PDF 是 P3 / P3-4；老索引没有这个字段
            "locator": str(data.get("locator") or ""),
            "text": (text[:TEXT_PREVIEW_CHARS] + "…") if truncated else text,
            "text_full": text if truncated else None,
            "text_chars": len(text),
            "truncated": truncated,
            "source": str(data.get("source") or ""),
            "abs_path": str(path) if path is not None else None,
        }

    def pick_collections(raw: list[str] | None) -> list[str]:
        """`collections` 查询参数 -> 实际要查的集合名列表。

        · 不传（或全是空串）-> `default_collections`
        · `all`             -> 所有已加载的集合
        · 有不存在名字      -> 400，detail 里列出可用集合（前端 chip 也可能过期）
        """
        names: list[str] = []
        for group in raw or []:
            for part in str(group or "").split(","):
                part = part.strip()
                if part and part not in names:
                    names.append(part)
        if not names:
            return list(app.state.default_collections)
        if any(name.lower() == "all" for name in names):
            return list(collection_names)
        missing = [name for name in names if name not in collection_names]
        if missing:
            available = "、".join(collection_names) or "(空)"
            raise HTTPException(
                status_code=400,
                detail=f"不存在的集合：{', '.join(missing)}。可用集合：{available}",
            )
        return names

    def review_chunks(item: dict) -> tuple[list[dict], str]:
        """复习题的参考原文：优先取 gold 块，取不到再实时检索。

        gold 是评测集标注的正确段落，id 已经在 `retriever.meta` 里，直接查
        等于 0 次查询编码，比重新检索更准也更快。只有 chunk_id 对不上
        （索引重建过、id 变了）时才退到检索；这时返回的是「系统认为相关的
        段落」，`chunks_source` 会如实写 `search`，前端照实标注来源。
        """
        chunks: list[dict] = []
        for cid in [str(c) for c in (item.get("gold") or [])][:REVIEW_MAX_CHUNKS]:
            data = from_meta(cid)
            if data is None:
                chunks = []
                break
            chunks.append(to_result(data, len(chunks) + 1))
        if chunks:
            return chunks, "gold"

        try:
            hits = retriever.search(item["question"], top_k=REVIEW_MAX_CHUNKS)
        except (FileNotFoundError, RuntimeError, ValueError):
            return [], "none"
        return ([to_result(hit, index) for index, hit in enumerate(hits, start=1)],
                "search")

    # ------------------------------------------------------------- 路由

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        try:
            html = INDEX_HTML.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"前端文件读不出来：{INDEX_HTML}（{exc}）",
            ) from exc
        return HTMLResponse(html)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/collections")
    def api_collections() -> dict:
        """集合列表 + 默认选中项。块数在 create_app 时就读好了，这里只抄缓存。"""
        return {
            "collections": [dict(entry) for entry in app.state.collections],
            "default": list(app.state.default_collections),
        }

    @app.get("/api/search")
    def api_search(
        q: str = Query("", description="查询词"),
        top_k: int = Query(10, ge=1, le=MAX_TOP_K, description="返回条数"),
        mode: str = Query("hybrid", description="hybrid / dense / sparse"),
        collections: list[str] | None = Query(
            None,
            description="只查这些集合，逗号分隔、可重复；all=全部；不传用默认集合",
        ),
    ) -> dict:
        query = (q or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="查询词 q 不能为空")

        use_dense, use_sparse = _mode_flags(mode)
        selected = pick_collections(collections)
        started = time.perf_counter()
        try:
            hits = retriever.search(
                query, top_k=top_k,
                use_dense=use_dense, use_sparse=use_sparse,
                collections=selected or None,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"检索失败：{exc}") from exc
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))

        results = [to_result(hit, i) for i, hit in enumerate(hits, start=1)]
        return {
            "query": query,
            "mode": (mode or "hybrid").strip().lower(),
            "top_k": top_k,
            "collections": selected,
            "elapsed_ms": elapsed_ms,
            "count": len(results),
            "results": results,
        }

    @app.post("/api/ask")
    def api_ask(payload: AskRequest) -> dict:
        if answerer is None:
            raise HTTPException(
                status_code=503,
                detail="问答未启用：启动时没有可用的模型客户端"
                       "（检查 DEEPSEEK_API_KEY 是否设置）。检索功能不受影响。",
            )
        question = (payload.q or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题 q 不能为空")
        top_k = max(1, min(int(payload.top_k or 5), MAX_TOP_K))
        # 和 /api/search 用同一套解析：不传走默认集合，all 查全部，未知名字 400。
        # 这样「问技术问题」时简历不会被检索进提示词。
        # 单索引模式下集合名是空串，这里滤掉 —— 空名不是真的集合，传给
        # answerer 只会让鸭子类型的假对象多收一个没用的参数。
        chosen = [name for name in pick_collections(payload.collections) if name]

        started = time.perf_counter()
        try:
            answer = answerer.answer(question, top_k=top_k, collections=chosen)
        except LLMError as exc:
            # LLMError 的消息在 llm.py 里过了一遍脱敏，不含 key / 请求头
            raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
        except TypeError as exc:
            # 自定义 answerer 可能只有 answer(question, top_k=...)。退回不带集合的调用，
            # 不能因为多传了一个关键字参数就让整个问答不可用。
            if "collections" not in str(exc):
                raise
            answer = answerer.answer(question, top_k=top_k)
            chosen = []
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"问答失败：{exc}") from exc
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))

        to_dict = getattr(answer, "to_dict", None)
        body = dict(to_dict()) if callable(to_dict) else dict(answer)
        for hit in body.get("hits") or []:
            remember(hit)
        body["elapsed_ms"] = elapsed_ms
        body["collections"] = chosen
        return body

    @app.get("/api/open")
    def api_open(chunk_id: str = Query("", description="检索结果里的 chunk_id")) -> dict:
        cid = (chunk_id or "").strip()
        if not cid:
            raise HTTPException(status_code=400, detail="缺少 chunk_id")
        hit = lookup(cid)
        if hit is None:
            raise HTTPException(
                status_code=404,
                detail=f"没有找到 chunk_id={cid}：先检索一次，或确认这个 id 属于当前索引",
            )

        line = int(hit.get("line_start") or 1)
        path = resolve_source_path(hit, raw_root)
        if path is None:
            return {
                "ok": False,
                "chunk_id": cid,
                "command": "",
                "path": None,
                "line": line,
                "error": "源文件不存在，或解析后落在 raw_dir 之外"
                         "（检查 --raw-dir 是否指向语料仓库的上一级目录）",
            }

        ok, command = open_in_editor(path, line, editor=app.state.editor)
        return {
            "ok": bool(ok),
            "chunk_id": cid,
            "command": command,
            "path": str(path),
            "line": line,
            "error": None if ok else command,
        }

    @app.get("/api/health")
    def api_health() -> dict:
        info = getattr(retriever, "index_info", None) or {}
        recorded = str(info.get("embedder") or "")
        loaded_embedder = getattr(retriever, "embedder", None)
        # 多集合时 `_count_chunks` 只看得到第一份索引，所以优先用集合块数之和
        chunks = sum(entry["chunks"] for entry in app.state.collections)
        if not chunks:
            chunks = _count_chunks(retriever)

        model_dir = getattr(retriever, "model_dir", None)
        model_dir_exists = bool(model_dir) and Path(model_dir).is_dir()
        index_ready = chunks > 0
        # hashing 索引的 embedder 是现成的，不需要模型文件；
        # onnx 索引只有模型目录在（或已经懒加载过）才算就绪。
        embedder_ready = (
            loaded_embedder is not None
            or "hashing" in recorded.lower()
            or model_dir_exists
        )
        return {
            "ok": True,
            "ready": bool(index_ready and embedder_ready),
            "index_ready": index_ready,
            "chunks": chunks,
            "collections": list(collection_names),
            "n_collections": len(collection_names),
            "embedder": recorded or (type(loaded_embedder).__name__
                                     if loaded_embedder is not None else ""),
            "embedder_loaded": loaded_embedder is not None,
            "model_dir": str(model_dir) if model_dir else None,
            "model_dir_exists": model_dir_exists,
            "index_dir": str(getattr(retriever, "index_dir", "") or ""),
            "raw_dir": str(raw_root),
            "raw_dir_exists": raw_root.is_dir(),
            "answerer": answerer is not None,
            "editor": active_editor,
            "dim": int(info.get("dim") or 0),
        }

    @app.get("/api/stats")
    def api_stats() -> dict:
        """索引统计。数值在 `create_app` 时算好，这里只做一次浅拷贝。"""
        stats = getattr(app.state, "stats", None) or {}
        body = dict(stats)
        body["repos"] = [dict(repo) for repo in body.get("repos") or []]
        body["collections"] = [
            {**entry, "repos": [dict(repo) for repo in entry.get("repos") or []]}
            for entry in body.get("collections") or []
        ]
        return body

    @app.get("/api/review/today")
    def api_review_today(
        day: str = Query("", description="覆盖日期 YYYY-MM-DD，默认今天"),
    ) -> dict:
        """今日复习：按日期从「评测失败题」里挑一道，附参考原文。

        没有用户系统也没有定时任务：同一天永远是同一道，隔天换一道，题库
        轮完再从头开始。题库本身在 `create_app` 时读好了，这里只做取模。

        题库不可用（没配文件 / 文件坏了 / 没有失败题）时返回 200 +
        `available=false`，原因放在 `reason` 里 —— 这是正常状态，不是请求错误，
        前端可以照常渲染一句话，不用走错误分支。
        """
        bank = app.state.review or {}
        try:
            target = parse_review_day(day)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        items = bank.get("items") or []
        body = {
            "available": False,
            "date": target.isoformat(),
            "source": str(bank.get("source") or ""),
            "reason": str(bank.get("reason") or "没有可复习的题目"),
            "total": len(items),
            "n_evaluated": int(bank.get("n_evaluated") or 0),
            "top_k": int(bank.get("top_k") or 5),
            "eval_top_k": int(bank.get("eval_top_k") or 0),
            # 不可用时也给空列表，前端不用先判断键在不在
            "chunks_source": "",
            "chunks": [],
        }
        if not items:
            return body

        position = target.toordinal() % len(items)
        item = items[position]
        chunks, chunks_source = review_chunks(item)
        return {
            **body,
            "available": True,
            "reason": "",
            "seq": position + 1,
            "qid": item["qid"],
            "question": item["question"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "rank": item["rank"],
            "chunks_source": chunks_source,
            "chunks": chunks,
        }

    return app
