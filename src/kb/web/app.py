"""FastAPI 应用工厂：检索 / 问答 / 编辑器回链。

设计上只有一个硬约束：**retriever 和 answerer 从外面传进来**。加载一次索引
要 6~70 秒（18k 块的元数据 + BM25 倒排 + 2 GB 模型的首次前向），每个请求
重建一次就等于服务不可用。`create_app` 只保存引用，不构造、不复制；路由里
用的永远是同一个对象，所以 `python ingest.py web` 启动一次之后，单次检索
只剩「编码 query + 矩阵乘 + BM25」的时间。

路由：

    GET  /                 单文件前端
    GET  /api/search       q / top_k / mode(hybrid|dense|sparse)
    POST /api/ask          {"q": ..., "top_k": ...} -> Answer.to_dict()
    GET  /api/open         chunk_id -> 编辑器打开源文件
    GET  /api/health       块数 / embedder / 就绪状态

`/api/open` 需要知道 chunk 的 repo/rel_path/行号。`Retriever` 自己持有全部
元数据（`.meta`），这里做两件事：优先查「本次进程里见过的检索结果」缓存，
查不到再按 chunk_id 扫一遍 retriever.meta 建一次索引。缓存有上限，避免
长时间运行的进程把每个请求的正文都攒在内存里。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..generate.llm import LLMError
from .editor import detect_editor, open_in_editor, resolve_source_path

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
    """`POST /api/ask` 的请求体。两个字段都有默认值，缺省也能进来再报错。"""

    q: str = ""
    top_k: int = 5


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


def create_app(retriever: Any, answerer: Any = None, *,
               raw_dir: Path | str, editor: str | None = None) -> FastAPI:
    """构造 FastAPI 应用。

    参数
    ----
    retriever : 任何有 `search(query, top_k=..., use_dense=..., use_sparse=...)`
                的对象。**必须是已经构造好的实例**，本函数不负责加载。
    answerer  : `Answerer` 或鸭子类型（有 `answer(question, top_k=...)`）。
                为 None 时 `/api/ask` 返回 503，检索照常可用。
    raw_dir   : 语料仓库根目录（`<raw_dir>/<repo>/<rel_path>` 就是源文件）。
    editor    : 'code' / 'notepad' / 可执行文件路径；None 表示启动时自动探测一次。
    """
    if retriever is None:
        raise ValueError("create_app 需要已经构造好的 retriever，不能传 None")

    raw_root = Path(raw_dir).resolve()
    # 探测器只跑一次：每次请求都 which 一遍没必要，结果在整个进程里都一样。
    active_editor = (editor or "").strip() or detect_editor()

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
            "breadcrumb": str(data.get("breadcrumb") or ""),
            "repo": str(data.get("repo") or ""),
            "rel_path": str(data.get("rel_path") or ""),
            "line_start": int(data.get("line_start") or 0),
            "line_end": int(data.get("line_end") or 0),
            "text": (text[:TEXT_PREVIEW_CHARS] + "…") if truncated else text,
            "text_full": text if truncated else None,
            "text_chars": len(text),
            "truncated": truncated,
            "source": str(data.get("source") or ""),
            "abs_path": str(path) if path is not None else None,
        }

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

    @app.get("/api/search")
    def api_search(
        q: str = Query("", description="查询词"),
        top_k: int = Query(10, ge=1, le=MAX_TOP_K, description="返回条数"),
        mode: str = Query("hybrid", description="hybrid / dense / sparse"),
    ) -> dict:
        query = (q or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="查询词 q 不能为空")

        use_dense, use_sparse = _mode_flags(mode)
        started = time.perf_counter()
        try:
            hits = retriever.search(
                query, top_k=top_k,
                use_dense=use_dense, use_sparse=use_sparse,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"检索失败：{exc}") from exc
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))

        results = [to_result(hit, i) for i, hit in enumerate(hits, start=1)]
        return {
            "query": query,
            "mode": (mode or "hybrid").strip().lower(),
            "top_k": top_k,
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

        started = time.perf_counter()
        try:
            answer = answerer.answer(question, top_k=top_k)
        except LLMError as exc:
            # LLMError 的消息在 llm.py 里过了一遍脱敏，不含 key / 请求头
            raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"问答失败：{exc}") from exc
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))

        to_dict = getattr(answer, "to_dict", None)
        body = dict(to_dict()) if callable(to_dict) else dict(answer)
        for hit in body.get("hits") or []:
            remember(hit)
        body["elapsed_ms"] = elapsed_ms
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

    return app
