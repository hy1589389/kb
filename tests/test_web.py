"""Phase 5 Web 层测试：编辑器回链 + FastAPI 应用（标准库 unittest，不加载模型）。

运行：
    python tests/test_web.py
    python -m unittest tests.test_web -v

不碰真实索引和 2 GB 模型：retriever / answerer 都是鸭子类型的假对象。
真实路径解析用工作区内的临时语料目录（系统 %TEMP% 在本机不可写）。

覆盖的点，每条都对应一个真会出错的地方：

  · rel_path 越界        —— `..` 能逃出 raw_dir 时，接口就变成任意文件启动器
  · 绝对路径 rel_path    —— Windows 上直接给 `C:\\...` 也要挡住
  · 文件不存在           —— 索引旧了、语料删了，要返回 None 而不是抛异常
  · code / notepad       —— 前者的参数是 `-g file:line`，后者没有行号参数
  · 编辑器不存在         —— 要返回 (False, 说明)，不能让 CLI 崩
  · retriever 只构造一次 —— 这是整个 Web 模式的性能前提
  · /api/open 未知 id    —— 404，而不是 500 或者悄悄打开别的文件
  · 集合列表 / 默认集合   —— 块数取自各索引的 index_meta.json，且只在启动时读一次
  · ?collections= 参数    —— 不传用默认、all 查全部、不存在的名字 400 并列出可用集合
  · /api/stats 按集合分组 —— 旧顶层字段必须还在（只增不减）
  · cmd_web 的索引发现     —— 没索引的集合跳过、全都没有才报错、--index-dir 退化成单集合
  · 前端集合 chip / 徽标   —— 没有构建工具，id、localStorage、请求参数只能静态检查
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows 控制台默认 cp936，中文断言消息会炸，这里强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from kb.generate.types import REFUSE_TEXT, Answer, Citation  # noqa: E402
from kb.retrieve.types import SearchHit  # noqa: E402
from kb.web import app as web_app  # noqa: E402
from kb.web.app import create_app  # noqa: E402
from kb.web.editor import detect_editor, open_in_editor, resolve_source_path  # noqa: E402
from kb.cli import _load_ingest_config, _resolve_web_indexes, build_parser  # noqa: E402

TMP_ROOT = ROOT / ".tmp"

HIT_TEXT = (
    "HashMap 在链表长度达到 8 时转红黑树，低于 6 时退回链表。"
    "红黑树是为了把最坏情况下的查找从 O(n) 降到 O(log n)。"
)


def make_hit(index: int = 1, *, repo: str = "JavaGuide",
             rel_path: str = "docs/hashmap.md", line_start: int = 12,
             line_end: int = 40, text: str | None = None,
             score: float = 0.03, collection: str = "",
             locator: str = "") -> SearchHit:
    """构造一条检索结果，repo/rel_path/行号都与临时语料目录对得上。"""
    return SearchHit(
        chunk_id=f"chunk-{index}",
        doc_id=f"doc-{index}",
        repo=repo,
        rel_path=rel_path,
        breadcrumb=f"JavaGuide > Map（重要） > HashMap 的底层实现 {index}",
        line_start=line_start,
        line_end=line_end,
        text=HIT_TEXT if text is None else text,
        score=score,
        source="hybrid",
        rank=index,
        collection=collection,
        locator=locator,
    )


def make_answer(hit: SearchHit, *, refused: bool = False,
                reason: str = "") -> Answer:
    citations = [] if refused else [Citation.from_hit(hit, ref=1)]
    return Answer(
        question="RAG 是什么",
        text=REFUSE_TEXT if refused else "RAG 是检索增强生成，先检索再生成 [1]。",
        citations=citations,
        refused=refused,
        refuse_reason=reason,
        hits=[hit],
        model="fake-model",
        usage={"total_tokens": 42},
        latency_ms=12,
        uncited_sentences=0,
        sentence_count=0 if refused else 1,
        context_chunks=0 if refused else 1,
    )


class FakeRetriever:
    """鸭子类型的检索器：记录调用参数，返回预设结果。

    `created` 是类变量，用来断言「整个进程只构造了一次」。

    `index_dirs` 模拟多索引的 `Retriever.index_dirs`（`{集合名: 索引目录}`）；
    不传就不设这个属性，`create_app` 会退回 `index_dir` 的单集合路径。
    """

    created = 0

    def __init__(self, hits: list[SearchHit], *,
                 index_dirs: dict[str, str | Path] | None = None):
        type(self).created += 1
        self.hits = list(hits)
        self.calls: list[dict] = []
        self.index_dir = str(TMP_ROOT / "index")
        if index_dirs is not None:
            self.index_dirs = {
                str(name): Path(path) for name, path in index_dirs.items()
            }
            first = next(iter(self.index_dirs.values()), None)
            if first is not None:
                self.index_dir = str(first)
        self.index_info = {
            "n_chunks": len(self.hits),
            "dim": 1024,
            "embedder": "OnnxEmbedder",
        }
        # embedder 已经加载过，health 里 ready 才会是 True
        self.embedder = object()
        self.model_dir = str(TMP_ROOT / "models" / "bge-m3")
        self.meta = [h.to_dict() for h in self.hits]

    def search(self, query: str, top_k: int = 10, *, use_dense: bool = True,
               use_sparse: bool = True, use_rerank: bool = False,
               collections: list[str] | str | None = None):
        names = None
        if collections is not None:
            names = ([x.strip() for x in collections.split(",")]
                     if isinstance(collections, str)
                     else [str(x).strip() for x in collections])
            names = [x for x in names if x] or None
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "use_dense": use_dense,
            "use_sparse": use_sparse,
            "use_rerank": use_rerank,
            "collections": names,
        })
        hits = self.hits
        if names:
            wanted = set(names)
            hits = [hit for hit in hits if hit.collection in wanted]
        return hits[:top_k]


class FakeAnswerer:
    def __init__(self, answer: Answer):
        self.answer_obj = answer
        self.calls: list[dict] = []

    def answer(self, question: str, *, top_k: int = 5,
               use_rerank: bool = False) -> Answer:
        self.calls.append({"question": question, "top_k": top_k})
        return self.answer_obj


class TempCorpus(unittest.TestCase):
    """每个用例一个临时语料目录：raw/<repo>/<rel_path>。"""

    def setUp(self) -> None:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = TMP_ROOT / f"web-{uuid.uuid4().hex[:8]}"
        self.raw = self.tmp / "raw"
        (self.raw / "JavaGuide" / "docs").mkdir(parents=True)
        self.src = self.raw / "JavaGuide" / "docs" / "hashmap.md"
        self.src.write_text("# HashMap\n\n链表长度到 8 转红黑树。\n", encoding="utf-8")
        # raw_dir 之外的敏感文件：用它证明「越界返回 None」不是靠文件不存在蒙对的
        self.outside = self.tmp / "secret.md"
        self.outside.write_text("SECRET", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------- 编辑器


class ResolveSourcePathTests(TempCorpus):
    def test_resolves_file_inside_raw_dir(self) -> None:
        path = resolve_source_path(make_hit(), self.raw)
        self.assertIsNotNone(path)
        self.assertEqual(path, self.src.resolve())
        self.assertTrue(path.is_absolute())
        self.assertTrue(path.is_file())

    def test_accepts_plain_dict(self) -> None:
        chunk = {"repo": "JavaGuide", "rel_path": "docs/hashmap.md",
                 "line_start": 3, "line_end": 9}
        self.assertEqual(resolve_source_path(chunk, self.raw), self.src.resolve())

    def test_rejects_parent_escape(self) -> None:
        # raw/JavaGuide/docs/../../../secret.md -> tmp/secret.md，文件真的存在
        self.assertTrue(self.outside.is_file())
        hit = make_hit(rel_path="../../../secret.md")
        self.assertIsNone(resolve_source_path(hit, self.raw))

    def test_rejects_repo_escape(self) -> None:
        hit = make_hit(repo="..", rel_path="../secret.md")
        self.assertIsNone(resolve_source_path(hit, self.raw))

    def test_rejects_absolute_rel_path(self) -> None:
        hit = make_hit(rel_path=str(self.outside))
        self.assertIsNone(resolve_source_path(hit, self.raw))

    def test_rejects_missing_file(self) -> None:
        hit = make_hit(rel_path="docs/nope.md")
        self.assertIsNone(resolve_source_path(hit, self.raw))

    def test_rejects_empty_rel_path(self) -> None:
        self.assertIsNone(resolve_source_path(make_hit(rel_path=""), self.raw))

    def test_allows_dotdot_that_stays_inside(self) -> None:
        # docs/../hashmap.md 仍在 raw_dir 内，应当放行
        hit = make_hit(rel_path="docs/../docs/hashmap.md")
        self.assertEqual(resolve_source_path(hit, self.raw), self.src.resolve())

    # ---- 多集合之后新增的三种布局 ------------------------------------

    def test_resolves_markdown_inside_collection(self) -> None:
        """Markdown 集合：<raw>/<collection>/<repo>/<rel_path>。"""
        target = self.raw / "tech" / "JavaGuide" / "docs" / "hashmap.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# HashMap\n", encoding="utf-8")
        chunk = {"repo": "JavaGuide", "rel_path": "docs/hashmap.md",
                 "collection": "tech", "line_start": 1, "line_end": 2}
        self.assertEqual(resolve_source_path(chunk, self.raw), target.resolve())

    def test_resolves_flat_pdf_inside_collection(self) -> None:
        """PDF 集合是平铺的：<raw>/<collection>/<rel_path>，repo 只是文件名。

        PDF 目录里没有 `<repo>` 这个子目录，所以不能套 `<repo>/<rel_path>`。
        """
        target = self.raw / "profile" / "何毓-简历.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"%PDF-1.4\n")
        chunk = {"repo": "何毓-简历", "rel_path": "何毓-简历.pdf",
                 "collection": "profile", "locator": "P1",
                 "line_start": 1, "line_end": 1}
        self.assertEqual(resolve_source_path(chunk, self.raw), target.resolve())

    def test_falls_back_to_legacy_layout_without_collection(self) -> None:
        """没有 collection 字段（旧索引）时仍按 <raw>/<repo>/<rel_path> 解析。"""
        chunk = {"repo": "JavaGuide", "rel_path": "docs/hashmap.md"}
        self.assertEqual(resolve_source_path(chunk, self.raw), self.src.resolve())

    def test_collection_never_escapes_raw_dir(self) -> None:
        """collection 里塞 `..` 也不能打开 raw_dir 之外的文件。"""
        chunk = {"repo": "JavaGuide", "rel_path": "docs/hashmap.md",
                 "collection": "..", "line_start": 1, "line_end": 2}
        path = resolve_source_path(chunk, self.raw)
        if path is not None:
            self.assertTrue(path.is_relative_to(self.raw.resolve()))

    def test_legacy_meta_without_collection_finds_nested_corpus(self) -> None:
        """旧索引的 meta 没有 collection 字段，但语料已经挪进 <raw>/tech/。

        Web 的 /api/open 直接查原始 meta 字典，拿不到 collection，所以解析
        函数必须在缺字段时自己扫一级子目录，否则打开源码会失效。

        这里用 CS-Notes 而不是 TempCorpus 默认的 JavaGuide：raw/JavaGuide
        在夹具里本来就存在，两个候选都能命中会让用例测不出「扫子目录」这件事。
        """
        target = self.raw / "tech" / "CS-Notes" / "README.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# CS-Notes\n", encoding="utf-8")
        legacy = {"repo": "CS-Notes", "rel_path": "README.md",
                  "line_start": 1, "line_end": 2}   # 故意没有 collection
        self.assertEqual(resolve_source_path(legacy, self.raw), target.resolve())


class DetectEditorTests(unittest.TestCase):
    def test_prefers_code(self) -> None:
        def fake_which(name):
            return r"C:\fake\code.cmd" if name == "code" else r"C:\fake\notepad.exe"

        with mock.patch("kb.web.editor.shutil.which", side_effect=fake_which):
            self.assertEqual(detect_editor(), "code")

    def test_falls_back_to_notepad(self) -> None:
        def fake_which(name):
            return r"C:\fake\notepad.exe" if name == "notepad" else None

        with mock.patch("kb.web.editor.shutil.which", side_effect=fake_which):
            self.assertEqual(detect_editor(), "notepad")

    def test_none_when_nothing_available(self) -> None:
        with mock.patch("kb.web.editor.shutil.which", return_value=None):
            self.assertIsNone(detect_editor())


class OpenInEditorTests(TempCorpus):
    def setUp(self) -> None:
        super().setUp()
        self.code_exe = r"C:\fake\code.cmd"
        self.notepad_exe = r"C:\fake\notepad.exe"

    def test_code_uses_goto_argument(self) -> None:
        with mock.patch("kb.web.editor.shutil.which", return_value=self.code_exe), \
                mock.patch("kb.web.editor.subprocess.Popen") as popen:
            ok, command = open_in_editor(self.src, 42)

        self.assertTrue(ok)
        argv = popen.call_args[0][0]
        self.assertEqual(argv, [self.code_exe, "-g", f"{self.src}:42"])
        self.assertIn("-g", command)
        self.assertIn(f"{self.src}:42", command)

    def test_explicit_editor_path_is_honoured(self) -> None:
        with mock.patch("kb.web.editor.shutil.which", return_value=None), \
                mock.patch("kb.web.editor.subprocess.Popen") as popen:
            ok, _ = open_in_editor(self.src, 7, editor=self.code_exe)
        # which 找不到就报错，不会去猜路径
        self.assertFalse(ok)
        popen.assert_not_called()

    def test_notepad_opens_file_without_line(self) -> None:
        def fake_which(name):
            return self.notepad_exe if name == "notepad" else None

        with mock.patch("kb.web.editor.shutil.which", side_effect=fake_which), \
                mock.patch("kb.web.editor.subprocess.Popen") as popen:
            ok, command = open_in_editor(self.src, 42)

        self.assertTrue(ok)
        self.assertEqual(popen.call_args[0][0], [self.notepad_exe, str(self.src)])
        self.assertIn("不支持跳行", command)
        self.assertIn("42", command)

    def test_no_editor_returns_false(self) -> None:
        with mock.patch("kb.web.editor.shutil.which", return_value=None), \
                mock.patch("kb.web.editor.subprocess.Popen") as popen:
            ok, command = open_in_editor(self.src, 1)
        self.assertFalse(ok)
        self.assertIn("编辑器", command)
        popen.assert_not_called()

    def test_missing_file_returns_false(self) -> None:
        with mock.patch("kb.web.editor.subprocess.Popen") as popen:
            ok, command = open_in_editor(self.tmp / "nope.md", 1)
        self.assertFalse(ok)
        self.assertIn("不存在", command)
        popen.assert_not_called()

    def test_popen_failure_is_reported(self) -> None:
        with mock.patch("kb.web.editor.shutil.which", return_value=self.code_exe), \
                mock.patch("kb.web.editor.subprocess.Popen",
                           side_effect=OSError("拒绝访问")):
            ok, command = open_in_editor(self.src, 1)
        self.assertFalse(ok)
        self.assertIn("拒绝访问", command)


# ---------------------------------------------------------------- Web 应用


class WebAppTests(TempCorpus):
    def setUp(self) -> None:
        super().setUp()
        self.hits = [
            make_hit(1),
            make_hit(2, rel_path="docs/hashmap.md", line_start=40, line_end=80),
            make_hit(3, repo="CS-Notes", rel_path="notes/rag.md", line_start=1,
                     line_end=20, text="RAG " * 300),
        ]
        # CS-Notes 那份也真实存在，保证三条结果的 abs_path 都有值
        (self.raw / "CS-Notes" / "notes").mkdir(parents=True)
        (self.raw / "CS-Notes" / "notes" / "rag.md").write_text(
            "# RAG\n", encoding="utf-8")

    def build_app(self, answerer=None, editor: str | None = "code"):
        FakeRetriever.created = 0
        self.retriever = FakeRetriever(self.hits)
        self.app = create_app(self.retriever, answerer,
                              raw_dir=self.raw, editor=editor)
        return self.app

    # ---- /api/search ----

    def test_search_returns_expected_fields(self) -> None:
        app = self.build_app()
        with TestClient(app) as client:
            res = client.get("/api/search", params={"q": "HashMap", "top_k": 3})

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["count"], 3)
        self.assertEqual(len(body["results"]), 3)
        self.assertIn("elapsed_ms", body)

        required = {
            "chunk_id", "rank", "score", "breadcrumb", "repo", "rel_path",
            "line_start", "line_end", "text", "abs_path",
        }
        for item in body["results"]:
            self.assertTrue(required <= set(item), required - set(item))
            self.assertTrue(item["abs_path"])
            self.assertTrue(Path(item["abs_path"]).is_file())
            self.assertLessEqual(len(item["text"]), 401)

        self.assertEqual([r["rank"] for r in body["results"]], [1, 2, 3])
        self.assertEqual(body["results"][0]["rel_path"], "docs/hashmap.md")
        self.assertEqual(body["results"][0]["abs_path"], str(self.src.resolve()))

    def test_search_truncates_long_text_but_keeps_full(self) -> None:
        app = self.build_app()
        with TestClient(app) as client:
            body = client.get("/api/search", params={"q": "RAG"}).json()

        long_item = [r for r in body["results"] if r["chunk_id"] == "chunk-3"][0]
        self.assertTrue(long_item["truncated"])
        self.assertEqual(long_item["text_chars"], len("RAG " * 300))
        self.assertIsNotNone(long_item["text_full"])
        self.assertGreater(len(long_item["text_full"]), 400)
        self.assertLessEqual(len(long_item["text"]), 401)

    def test_search_passes_mode_flags(self) -> None:
        app = self.build_app()
        with TestClient(app) as client:
            client.get("/api/search", params={"q": "x", "mode": "sparse"})
            client.get("/api/search", params={"q": "x", "mode": "dense"})
            client.get("/api/search", params={"q": "x", "mode": "hybrid"})

        self.assertEqual(self.retriever.calls[0]["use_dense"], False)
        self.assertEqual(self.retriever.calls[0]["use_sparse"], True)
        self.assertEqual(self.retriever.calls[1]["use_dense"], True)
        self.assertEqual(self.retriever.calls[1]["use_sparse"], False)
        self.assertEqual(self.retriever.calls[2]["use_dense"], True)
        self.assertEqual(self.retriever.calls[2]["use_sparse"], True)

    def test_search_rejects_bad_mode_and_empty_query(self) -> None:
        app = self.build_app()
        with TestClient(app) as client:
            bad_mode = client.get("/api/search", params={"q": "x", "mode": "bogus"})
            empty = client.get("/api/search", params={"q": "   "})
        self.assertEqual(bad_mode.status_code, 400)
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(self.retriever.calls, [])

    # ---- /api/ask ----

    def test_ask_returns_answer(self) -> None:
        answer = make_answer(self.hits[0])
        answerer = FakeAnswerer(answer)
        app = self.build_app(answerer)
        with TestClient(app) as client:
            res = client.post("/api/ask", json={"q": "RAG 是什么", "top_k": 3})

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertFalse(body["refused"])
        self.assertIn("[1]", body["text"])
        self.assertEqual(len(body["citations"]), 1)
        self.assertEqual(body["citations"][0]["chunk_id"], "chunk-1")
        self.assertEqual(body["model"], "fake-model")
        self.assertIn("elapsed_ms", body)
        self.assertEqual(answerer.calls, [{"question": "RAG 是什么", "top_k": 3}])

    def test_ask_refused(self) -> None:
        answer = make_answer(self.hits[0], refused=True, reason="检索无结果")
        app = self.build_app(FakeAnswerer(answer))
        with TestClient(app) as client:
            res = client.post("/api/ask", json={"q": "无关问题"})

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["refused"])
        self.assertEqual(body["refuse_reason"], "检索无结果")
        self.assertEqual(body["citations"], [])
        self.assertEqual(body["text"], REFUSE_TEXT)

    def test_ask_without_answerer_is_503(self) -> None:
        app = self.build_app(answerer=None)
        with TestClient(app) as client:
            res = client.post("/api/ask", json={"q": "RAG 是什么"})
        self.assertEqual(res.status_code, 503)
        self.assertIn("问答未启用", res.json()["detail"])

    def test_ask_empty_question_is_400(self) -> None:
        app = self.build_app(FakeAnswerer(make_answer(self.hits[0])))
        with TestClient(app) as client:
            res = client.post("/api/ask", json={"q": "  "})
        self.assertEqual(res.status_code, 400)

    # ---- /api/open ----

    def test_open_unknown_chunk_is_404(self) -> None:
        app = self.build_app()
        with TestClient(app) as client:
            res = client.get("/api/open", params={"chunk_id": "does-not-exist"})
        self.assertEqual(res.status_code, 404)
        self.assertIn("does-not-exist", res.json()["detail"])

    def test_open_works_from_retriever_meta_without_prior_search(self) -> None:
        app = self.build_app()
        with TestClient(app) as client, \
                mock.patch.object(web_app, "open_in_editor",
                                  return_value=(True, 'code.cmd -g "x:12"')) as opened:
            res = client.get("/api/open", params={"chunk_id": "chunk-1"})

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["command"], 'code.cmd -g "x:12"')
        self.assertEqual(body["path"], str(self.src.resolve()))
        path, line = opened.call_args[0]
        self.assertEqual(path, self.src.resolve())
        self.assertEqual(line, 12)

    def test_open_works_for_search_results(self) -> None:
        app = self.build_app()
        with TestClient(app) as client, \
                mock.patch.object(web_app, "open_in_editor",
                                  return_value=(True, "ok")):
            client.get("/api/search", params={"q": "HashMap"})
            res = client.get("/api/open", params={"chunk_id": "chunk-2"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["ok"])

    def test_open_missing_source_file_reports_error(self) -> None:
        app = self.build_app()
        with TestClient(app) as client, \
                mock.patch.object(web_app, "open_in_editor",
                                  return_value=(True, "ok")):
            res = client.get("/api/open", params={"chunk_id": "chunk-3"})
            # chunk-3 指向 CS-Notes/notes/rag.md，存在；先删掉再试
            (self.raw / "CS-Notes" / "notes" / "rag.md").unlink()
            res2 = client.get("/api/open", params={"chunk_id": "chunk-3"})

        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["ok"])
        self.assertEqual(res2.status_code, 200)
        self.assertFalse(res2.json()["ok"])
        self.assertIsNone(res2.json()["path"])
        self.assertIn("源文件不存在", res2.json()["error"])

    def test_open_requires_chunk_id(self) -> None:
        app = self.build_app()
        with TestClient(app) as client:
            res = client.get("/api/open", params={"chunk_id": ""})
        self.assertEqual(res.status_code, 400)

    # ---- /api/health ----

    def test_health_fields(self) -> None:
        app = self.build_app()
        with TestClient(app) as client:
            res = client.get("/api/health")

        self.assertEqual(res.status_code, 200)
        body = res.json()
        expected = {
            "ok", "ready", "index_ready", "chunks", "embedder",
            "embedder_loaded", "model_dir", "model_dir_exists", "index_dir",
            "raw_dir", "raw_dir_exists", "answerer", "editor", "dim",
            # 多集合：集合名列表 + 数量
            "collections", "n_collections",
        }
        self.assertEqual(set(body), expected)
        self.assertTrue(body["ok"])
        self.assertTrue(body["ready"])
        self.assertTrue(body["index_ready"])
        self.assertEqual(body["chunks"], 3)
        # 单索引且没有集合名时集合名是空串（与 Retriever(Path) 的兼容路径一致）
        self.assertEqual(body["collections"], [""])
        self.assertEqual(body["n_collections"], 1)
        self.assertEqual(body["embedder"], "OnnxEmbedder")
        self.assertTrue(body["embedder_loaded"])
        self.assertEqual(body["dim"], 1024)
        self.assertEqual(body["editor"], "code")
        self.assertFalse(body["answerer"])
        self.assertEqual(body["raw_dir"], str(self.raw.resolve()))
        self.assertTrue(body["raw_dir_exists"])

    def test_health_reports_answerer(self) -> None:
        app = self.build_app(FakeAnswerer(make_answer(self.hits[0])))
        with TestClient(app) as client:
            self.assertTrue(client.get("/api/health").json()["answerer"])

    # ---- 前端 ----

    def test_index_html_has_key_elements(self) -> None:
        app = self.build_app()
        with TestClient(app) as client:
            res = client.get("/")

        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        html = res.text
        for needle in ('id="query-input"', 'id="search-btn"', 'id="mode-ask"',
                       'id="results"', 'id="answer-panel"', 'id="status-bar"',
                       "展开全文", "在 VS Code 打开", "/api/search", "/api/ask",
                       "/api/open", "/api/health"):
            self.assertIn(needle, html, f"index.html 缺少 {needle!r}")
        # 不许有外部依赖
        self.assertNotIn("<script src=", html)
        self.assertNotIn("cdn.", html.lower())

    def test_index_html_script_ids_all_exist(self) -> None:
        """JS 里 $('x') / getElementById('x') 引用的 id 必须在 HTML 里存在。

        前端没有构建工具，拼错一个 id 只会在浏览器里得到一个 null，
        页面白屏而单元测试仍然全绿 —— 所以这条得单独测。
        """
        html = web_app.INDEX_HTML.read_text(encoding="utf-8")
        script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
        ids = set(re.findall(r'id="([^"]+)"', html))
        used = set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", script))
        used |= set(re.findall(r"getElementById\('([A-Za-z0-9_-]+)'\)", script))
        # 引用卡片的 id 是 JS 里拼的 citation-<n>
        missing = sorted(used - ids - {"citation-"})
        self.assertEqual(missing, [], f"index.html 里没有这些 id：{missing}")

    # ---- 性能前提 ----

    def test_retriever_constructed_once(self) -> None:
        answerer = FakeAnswerer(make_answer(self.hits[0]))
        app = self.build_app(answerer)
        self.assertEqual(FakeRetriever.created, 1)
        self.assertEqual(self.retriever.calls, [])  # create_app 不预热、不查询

        with TestClient(app) as client:
            for _ in range(3):
                self.assertEqual(
                    client.get("/api/search", params={"q": "HashMap"}).status_code,
                    200)
            client.get("/api/health")
            client.post("/api/ask", json={"q": "HashMap"})

        self.assertEqual(FakeRetriever.created, 1,
                         "retriever 被重复构造了，Web 模式会退化成每次请求 70 秒")
        self.assertEqual(len(self.retriever.calls), 3)
        self.assertIs(app.state.retriever, self.retriever)
        self.assertIs(app.state.answerer, answerer)

    def test_app_keeps_passed_instances(self) -> None:
        answerer = FakeAnswerer(make_answer(self.hits[0]))
        app = self.build_app(answerer)
        self.assertIs(app.state.retriever, self.retriever)
        self.assertIs(app.state.answerer, answerer)
        self.assertEqual(app.state.editor, "code")
        self.assertEqual(app.state.raw_dir, self.raw.resolve())


# ---------------------------------------------------------------- 多集合


class MultiCollectionTests(TempCorpus):
    """多集合：两个真实索引目录 + 带 `index_dirs` 的假 retriever。

    这一层要验证的是「Web 层只做转发和校验」：集合列表、默认集合、按集合
    查、结果里的 collection 字段、以及统计的分组。真实的跨集合融合在
    `retrieve/` 里，这里不重复测。
    """

    TECH_CHUNKS = 3
    PROFILE_CHUNKS = 2

    def setUp(self) -> None:
        super().setUp()
        self.tech_dir = self.tmp / "index"
        self.profile_dir = self.tech_dir / "profile"
        self.profile_dir.mkdir(parents=True)

        # tech 索引：3 块，两个仓库
        self.write_index(self.tech_dir, n_chunks=self.TECH_CHUNKS, rows=[
            {"repo": "JavaGuide", "char_count": 100},
            {"repo": "CS-Notes", "char_count": 200},
            {"repo": "JavaGuide", "char_count": 300},
        ])
        # profile 索引：2 块，一个仓库
        self.write_index(self.profile_dir, n_chunks=self.PROFILE_CHUNKS, rows=[
            {"repo": "何毓-简历", "char_count": 50},
            {"repo": "何毓-简历", "char_count": 150},
        ])
        # 两个索引目录各放一个 1 MiB 的文件：profile 在 tech 目录下面，
        # 顶层汇总如果不去重就会变成 3 MiB 而不是 2 MiB。
        self.blob_tech = self.tech_dir / "blob.bin"
        self.blob_tech.write_bytes(b"x" * (1024 * 1024))
        self.blob_profile = self.profile_dir / "blob.bin"
        self.blob_profile.write_bytes(b"y" * (1024 * 1024))

        # 语料按集合分目录：raw/tech/<repo>/<rel_path>、raw/profile/<file>
        tech_src = self.raw / "tech" / "JavaGuide" / "docs" / "hashmap.md"
        tech_src.parent.mkdir(parents=True)
        tech_src.write_text("# HashMap\n\n链表长度到 8 转红黑树。\n", encoding="utf-8")
        self.tech_src = tech_src
        pdf = self.raw / "profile" / "何毓-简历.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b"%PDF-1.4\n")
        self.pdf = pdf

        self.hits = [
            make_hit(1, collection="tech"),
            make_hit(2, collection="tech", repo="CS-Notes",
                     rel_path="notes/rag.md", line_start=40, line_end=80),
            make_hit(3, collection="profile", repo="何毓-简历",
                     rel_path="何毓-简历.pdf", locator="P1",
                     line_start=1, line_end=1),
        ]

    # ---- 夹具 ----

    def write_index(self, directory: Path, *, n_chunks: int,
                    rows: list[dict]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index_meta.json").write_text(
            json.dumps({"n_chunks": n_chunks, "dim": 1024,
                        "embedder": "OnnxEmbedder"}, ensure_ascii=False),
            encoding="utf-8",
        )
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (directory / "chunks_meta.jsonl").write_text(text, encoding="utf-8")

    def build_app(self, *, default_collections=None, hits=None):
        retriever = FakeRetriever(
            self.hits if hits is None else hits,
            index_dirs={"tech": self.tech_dir, "profile": self.profile_dir},
        )
        app = create_app(retriever, None, raw_dir=self.raw, editor="code",
                         default_collections=default_collections)
        self.retriever = retriever
        return app

    # ---- /api/collections ----

    def test_collections_fields_and_counts(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            res = client.get("/api/collections")

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(set(body), {"collections", "default"})
        self.assertEqual(body["default"], ["tech"])
        entries = body["collections"]
        self.assertEqual([entry["name"] for entry in entries], ["tech", "profile"])
        for entry in entries:
            self.assertEqual(set(entry), {"name", "chunks", "has_index"})
            self.assertTrue(entry["has_index"])
        counts = {entry["name"]: entry["chunks"] for entry in entries}
        self.assertEqual(counts, {"tech": self.TECH_CHUNKS,
                                  "profile": self.PROFILE_CHUNKS})

    def test_collections_default_is_all_when_not_given(self) -> None:
        app = self.build_app()
        with TestClient(app) as client:
            body = client.get("/api/collections").json()
        self.assertEqual(body["default"], ["tech", "profile"])

    def test_collections_default_ignores_unknown_names(self) -> None:
        app = self.build_app(default_collections=["nope", "profile"])
        with TestClient(app) as client:
            body = client.get("/api/collections").json()
        self.assertEqual(body["default"], ["profile"])

    def test_collections_cached_at_create_app(self) -> None:
        """集合列表只在 create_app 时读一次 index_meta.json。"""
        with mock.patch.object(web_app, "_read_index_meta",
                               wraps=web_app._read_index_meta) as reader:
            app = self.build_app(default_collections=["tech"])
            after_create = reader.call_count
            with TestClient(app) as client:
                for _ in range(3):
                    self.assertEqual(
                        client.get("/api/collections").status_code, 200)

        self.assertEqual(after_create, 2, "create_app 应该每个集合读一次")
        self.assertEqual(reader.call_count, after_create,
                         "集合信息被重复读盘了，必须缓存在 app.state")

    def test_collections_entries_are_cached_values(self) -> None:
        """create_app 之后删掉索引元数据，接口仍返回启动时的结果。"""
        app = self.build_app(default_collections=["tech"])
        (self.profile_dir / "index_meta.json").unlink()
        with TestClient(app) as client:
            body = client.get("/api/collections").json()
        profile = [e for e in body["collections"] if e["name"] == "profile"][0]
        self.assertTrue(profile["has_index"])
        self.assertEqual(profile["chunks"], self.PROFILE_CHUNKS)

    # ---- /api/search?collections= ----

    def test_search_only_requested_collection(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            body = client.get("/api/search",
                              params={"q": "项目经历", "collections": "profile"}).json()

        self.assertEqual(body["collections"], ["profile"])
        self.assertEqual(body["count"], 1)
        self.assertEqual({item["collection"] for item in body["results"]},
                         {"profile"})
        self.assertEqual(self.retriever.calls[-1]["collections"], ["profile"])

    def test_search_collections_all(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            body = client.get("/api/search",
                              params={"q": "RAG", "collections": "all"}).json()

        self.assertEqual(body["collections"], ["tech", "profile"])
        self.assertEqual(body["count"], 3)
        self.assertEqual({item["collection"] for item in body["results"]},
                         {"tech", "profile"})

    def test_search_all_is_case_insensitive(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            body = client.get("/api/search",
                              params={"q": "RAG", "collections": "ALL"}).json()
        self.assertEqual(body["collections"], ["tech", "profile"])

    def test_search_unknown_collection_is_400_with_available_names(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            res = client.get("/api/search",
                             params={"q": "x", "collections": "nope"})

        self.assertEqual(res.status_code, 400)
        detail = res.json()["detail"]
        self.assertIn("nope", detail)
        self.assertIn("tech", detail)
        self.assertIn("profile", detail)
        self.assertEqual(self.retriever.calls, [])

    def test_search_without_collections_uses_default(self) -> None:
        app = self.build_app(default_collections=["profile"])
        with TestClient(app) as client:
            body = client.get("/api/search", params={"q": "HashMap"}).json()

        self.assertEqual(body["collections"], ["profile"])
        self.assertEqual(self.retriever.calls[-1]["collections"], ["profile"])
        self.assertEqual({item["collection"] for item in body["results"]},
                         {"profile"})

    def test_search_empty_collections_param_uses_default(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            body = client.get("/api/search",
                              params={"q": "x", "collections": ""}).json()
        self.assertEqual(body["collections"], ["tech"])

    def test_search_accepts_repeated_and_comma_separated(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            repeated = client.get("/api/search", params=[
                ("q", "RAG"), ("collections", "tech"), ("collections", "profile"),
            ]).json()
            comma = client.get("/api/search",
                               params={"q": "RAG",
                                       "collections": "tech,profile"}).json()

        for body in (repeated, comma):
            self.assertEqual(body["collections"], ["tech", "profile"])
            self.assertEqual(body["count"], 3)

    def test_search_results_carry_collection_and_locator(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            body = client.get("/api/search",
                              params={"q": "简历", "collections": "profile"}).json()

        item = body["results"][0]
        self.assertEqual(item["collection"], "profile")
        self.assertEqual(item["locator"], "P1")
        self.assertEqual(item["abs_path"], str(self.pdf.resolve()))

    # ---- /api/open 按集合解析路径 ----

    def test_open_uses_result_collection_for_path(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client, \
                mock.patch.object(web_app, "open_in_editor",
                                  return_value=(True, "ok")) as opened:
            client.get("/api/search",
                       params={"q": "简历", "collections": "profile"})
            res = client.get("/api/open", params={"chunk_id": "chunk-3"})

        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["ok"])
        self.assertEqual(res.json()["path"], str(self.pdf.resolve()))
        self.assertEqual(opened.call_args[0][0], self.pdf.resolve())

    # ---- /api/health ----

    def test_health_reports_collections(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            body = client.get("/api/health").json()

        self.assertEqual(body["collections"], ["tech", "profile"])
        self.assertEqual(body["n_collections"], 2)
        # 块数是所有集合之和（假 retriever 的 index_info 只有一份的块数）
        self.assertEqual(body["chunks"], self.TECH_CHUNKS + self.PROFILE_CHUNKS)

    # ---- /api/stats 按集合分组 ----

    def test_stats_groups_by_collection(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            body = client.get("/api/stats").json()

        groups = {entry["name"]: entry for entry in body["collections"]}
        self.assertEqual(sorted(groups), ["profile", "tech"])
        self.assertEqual(groups["tech"]["chunks"], self.TECH_CHUNKS)
        self.assertEqual(groups["profile"]["chunks"], self.PROFILE_CHUNKS)
        self.assertEqual(groups["tech"]["avg_chars"], 200)       # (100+200+300)/3
        self.assertEqual(groups["profile"]["avg_chars"], 100)    # (50+150)/2
        self.assertEqual(groups["tech"]["index_dir"], str(self.tech_dir))
        self.assertEqual(groups["profile"]["index_dir"], str(self.profile_dir))
        for entry in body["collections"]:
            self.assertEqual(
                set(entry),
                {"name", "chunks", "repos", "avg_chars",
                 "index_dir", "index_size_mb"},
            )

    def test_stats_index_size_does_not_double_count_nested_index(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            body = client.get("/api/stats").json()

        groups = {entry["name"]: entry for entry in body["collections"]}
        # tech 目录里还有 profile 索引：算 tech 的大小时要排掉
        self.assertAlmostEqual(groups["tech"]["index_size_mb"], 1.0, places=1)
        self.assertAlmostEqual(groups["profile"]["index_size_mb"], 1.0, places=1)
        self.assertAlmostEqual(body["index_size_mb"], 2.0, places=1)

    def test_stats_keeps_legacy_top_level_fields(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            body = client.get("/api/stats").json()

        # 旧字段只增不减
        for key in ("chunks", "repos", "avg_chars", "index_dir", "index_size_mb"):
            self.assertIn(key, body)
        self.assertEqual(body["chunks"], self.TECH_CHUNKS + self.PROFILE_CHUNKS)
        self.assertEqual(body["total_chunks"], body["chunks"])
        self.assertEqual(sum(entry["chunks"] for entry in body["collections"]),
                         body["chunks"])
        self.assertEqual(sum(repo["chunks"] for repo in body["repos"]),
                         body["chunks"])
        # (100+200+300+50+150) / 5 = 160
        self.assertEqual(body["avg_chars"], 160)
        self.assertEqual(body["index_dir"], str(self.tech_dir))
        self.assertEqual(
            {repo["name"]: repo["chunks"] for repo in body["repos"]},
            {"JavaGuide": 2, "何毓-简历": 2, "CS-Notes": 1},
        )

    # ---- 首页 ----

    def test_index_page_has_collection_ui_and_no_external_assets(self) -> None:
        app = self.build_app(default_collections=["tech"])
        with TestClient(app) as client:
            res = client.get("/")

        self.assertEqual(res.status_code, 200)
        html = res.text
        for needle in ('id="collection-chips"', 'id="collection-filters"',
                       'id="collection-badge-tpl"', "/api/collections"):
            self.assertIn(needle, html)
        lowered = html.lower()
        self.assertNotIn("<script src=", lowered)
        self.assertNotIn('href="http', lowered)
        self.assertNotIn("<link", lowered)

# ---------------------------------------------------------------- /api/stats


class StatsTests(TempCorpus):
    """`/api/stats`：数值来自 chunks_meta.jsonl，且只在 create_app 时扫一遍。"""

    ROWS = [
        {"chunk_id": "chunk-1", "repo": "JavaGuide", "char_count": 100},
        {"chunk_id": "chunk-2", "repo": "CS-Notes", "char_count": 200},
        {"chunk_id": "chunk-3", "repo": "JavaGuide", "char_count": 300},
    ]

    def setUp(self) -> None:
        super().setUp()
        self.hits = [
            make_hit(1),
            make_hit(2, repo="CS-Notes", rel_path="notes/rag.md"),
            make_hit(3),
        ]
        self.index = self.tmp / "index"
        self.index.mkdir()
        self.write_meta(self.ROWS)
        # 1 MiB + 2 KiB，方便断言 index_size_mb 真的是按目录累计的
        self.blob = self.index / "blob.bin"
        self.blob.write_bytes(b"x" * (1024 * 1024 + 2048))

    def write_meta(self, rows: list[dict]) -> None:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (self.index / "chunks_meta.jsonl").write_text(text, encoding="utf-8")

    def build_app(self, index_dir: Path | None = None):
        retriever = FakeRetriever(self.hits)
        retriever.index_dir = str(self.index if index_dir is None else index_dir)
        return create_app(retriever, None, raw_dir=self.raw, editor="code")

    def test_stats_fields_complete(self) -> None:
        app = self.build_app()
        with TestClient(app) as client:
            res = client.get("/api/stats")

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(
            set(body),
            # 旧字段一个都不能少（只增不减），collections / total_chunks 是新增的
            {"chunks", "repos", "avg_chars", "index_dir", "index_size_mb",
             "collections", "total_chunks"},
        )
        self.assertEqual(body["chunks"], 3)
        self.assertEqual(body["total_chunks"], 3)
        self.assertEqual(body["avg_chars"], 200)          # (100 + 200 + 300) / 3
        self.assertEqual(body["index_dir"], str(self.index))
        self.assertIsInstance(body["repos"], list)
        # 单集合时分组里也只有一项
        self.assertEqual(len(body["collections"]), 1)
        self.assertEqual(body["collections"][0]["name"], "")
        self.assertEqual(body["collections"][0]["chunks"], 3)

    def test_stats_repos_sorted_and_sum_to_chunks(self) -> None:
        app = self.build_app()
        with TestClient(app) as client:
            body = client.get("/api/stats").json()

        repos = body["repos"]
        self.assertEqual([repo["name"] for repo in repos], ["JavaGuide", "CS-Notes"])
        self.assertEqual([repo["chunks"] for repo in repos], [2, 1])
        self.assertEqual(sum(repo["chunks"] for repo in repos), body["chunks"])
        for repo in repos:
            self.assertEqual(set(repo), {"name", "chunks"})

    def test_stats_index_size_mb_counts_nested_files(self) -> None:
        nested = self.index / "sub"
        nested.mkdir()
        (nested / "extra.bin").write_bytes(b"y" * 4096)
        expected = sum(
            path.stat().st_size for path in self.index.rglob("*") if path.is_file()
        ) / (1024 * 1024)

        app = self.build_app()
        with TestClient(app) as client:
            body = client.get("/api/stats").json()

        self.assertGreater(body["index_size_mb"], 0.9)
        self.assertAlmostEqual(body["index_size_mb"], round(expected, 1), places=1)

    def test_stats_scans_index_only_once(self) -> None:
        with mock.patch.object(web_app, "_iter_chunk_meta",
                               wraps=web_app._iter_chunk_meta) as rows:
            app = self.build_app()
            with TestClient(app) as client:
                bodies = [client.get("/api/stats").json() for _ in range(4)]

        self.assertEqual(rows.call_count, 1,
                         "chunks_meta.jsonl 被重复扫描了，统计必须缓存在 create_app")
        self.assertEqual(bodies[0], bodies[-1])
        self.assertEqual(app.state.stats["chunks"], 3)

    def test_stats_is_cached_not_recomputed(self) -> None:
        app = self.build_app()
        # create_app 之后再改索引：接口应该仍然返回启动时的缓存
        self.write_meta(self.ROWS[:1])
        with TestClient(app) as client:
            body = client.get("/api/stats").json()

        self.assertEqual(body["chunks"], 3)
        self.assertEqual(app.state.stats["chunks"], 3)

    def test_stats_falls_back_to_retriever_meta(self) -> None:
        # 索引目录不存在（假 retriever / 只加载了一半）：退回内存里的 meta
        missing = self.tmp / "no-such-index"
        app = self.build_app(index_dir=missing)
        with TestClient(app) as client:
            body = client.get("/api/stats").json()

        self.assertEqual(body["chunks"], 3)
        self.assertEqual(body["avg_chars"], len(HIT_TEXT))
        self.assertEqual(body["index_size_mb"], 0.0)
        self.assertEqual(body["index_dir"], str(missing))
        counts = {repo["name"]: repo["chunks"] for repo in body["repos"]}
        self.assertEqual(counts, {"JavaGuide": 2, "CS-Notes": 1})
        self.assertEqual(sum(counts.values()), body["chunks"])

    def test_stats_empty_index_returns_zeros(self) -> None:
        retriever = FakeRetriever([])
        retriever.index_dir = str(self.tmp / "empty-index")
        app = create_app(retriever, None, raw_dir=self.raw, editor="code")
        with TestClient(app) as client:
            body = client.get("/api/stats").json()

        self.assertEqual(body["chunks"], 0)
        self.assertEqual(body["repos"], [])
        self.assertEqual(body["avg_chars"], 0)
        self.assertEqual(body["index_size_mb"], 0.0)

    def test_stats_groups_missing_repo_but_keeps_total(self) -> None:
        # repo 字段缺失的块归到「(未标注仓库)」，保证求和恒等于 chunks
        self.write_meta([
            {"chunk_id": "a", "repo": "", "char_count": 10},
            {"chunk_id": "b", "repo": "JavaGuide", "char_count": 30},
        ])
        app = self.build_app()
        with TestClient(app) as client:
            body = client.get("/api/stats").json()

        counts = {repo["name"]: repo["chunks"] for repo in body["repos"]}
        self.assertEqual(counts, {"(未标注仓库)": 1, "JavaGuide": 1})
        self.assertEqual(body["chunks"], 2)
        self.assertEqual(sum(counts.values()), body["chunks"])

    def test_stats_route_rejects_non_get(self) -> None:
        app = self.build_app()
        with TestClient(app) as client:
            res = client.post("/api/stats")
        self.assertEqual(res.status_code, 405)


# ---------------------------------------------------------------- cmd_web 发现索引


class WebIndexPlanTests(unittest.TestCase):
    """`cmd_web` 的索引发现逻辑（`_resolve_web_indexes`），不起服务、不加载模型。

    这里覆盖的是命令行到 `Retriever` 之间的那一段：哪些集合会被加载、没索引
    的集合是跳过还是报错、`--index-dir` 的单集合兼容模式、权重从哪来。
    """

    TEMPLATE = """\
[collections]
default_search = {default}

[collections.tech]
raw_dir = "{root}/raw/tech"
index_dir = "{root}/index"
parser = "markdown"

[collections.profile]
raw_dir = "{root}/raw/profile"
index_dir = "{root}/index/profile"
parser = "auto"
weight = {profile_weight}

[collections.papers]
raw_dir = "{root}/raw/papers"
index_dir = "{root}/index/papers"
parser = "auto"
"""

    def setUp(self) -> None:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = TMP_ROOT / f"plan-{uuid.uuid4().hex[:8]}"
        self.tmp.mkdir(parents=True)
        self.tech = self.tmp / "index"
        self.profile = self.tech / "profile"
        self.papers = self.tech / "papers"
        for directory, n_chunks in ((self.tech, 3), (self.profile, 2)):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "index_meta.json").write_text(
                json.dumps({"n_chunks": n_chunks, "embedder": "OnnxEmbedder"}),
                encoding="utf-8",
            )
        self.config = self.tmp / "config.toml"
        self.write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_config(self, *, default: str = '["tech"]',
                     profile_weight: str = "1.0") -> None:
        # TOML 基本字符串里反斜杠是转义符，路径统一用正斜杠，Windows 也认
        self.config.write_text(
            self.TEMPLATE.format(root=self.tmp.as_posix(), default=default,
                                 profile_weight=profile_weight),
            encoding="utf-8",
        )

    def plan(self, *extra: str) -> dict:
        args = build_parser().parse_args(
            ["web", "--config", str(self.config), *extra])
        cfg = _load_ingest_config(args)
        cfg.resolve(self.tmp)          # 配置里全是绝对路径，这一步不改任何东西
        return _resolve_web_indexes(args, cfg)

    def test_plan_loads_every_indexed_collection(self) -> None:
        plan = self.plan()
        self.assertFalse(plan["single"])
        self.assertEqual(list(plan["index_map"]), ["tech", "profile"])
        self.assertEqual(plan["chunks"], {"tech": 3, "profile": 2})
        self.assertEqual(plan["default"], ["tech"])

    def test_plan_skips_collections_without_index(self) -> None:
        plan = self.plan()
        # papers 的 index_dir 不存在：跳过、不报错
        self.assertEqual(plan["skipped"], ["papers"])
        self.assertNotIn("papers", plan["index_map"])

    def test_plan_weights_come_from_collection_config(self) -> None:
        self.write_config(profile_weight="0.8")
        plan = self.plan()
        self.assertEqual(plan["weights"], {"tech": 1.0, "profile": 0.8})

    def test_plan_raises_only_when_nothing_has_an_index(self) -> None:
        for directory in (self.tech, self.profile):
            (directory / "index_meta.json").unlink()
        with self.assertRaises(FileNotFoundError) as caught:
            self.plan()
        message = str(caught.exception)
        self.assertIn("所有集合都还没有索引", message)
        self.assertIn("python ingest.py index", message)

    def test_plan_default_falls_back_to_all_indexed(self) -> None:
        # default_search 指向没索引的 papers：默认集合退回全部有索引的
        self.write_config(default='["papers"]')
        plan = self.plan()
        self.assertEqual(plan["default"], ["tech", "profile"])

    def test_plan_index_dir_forces_single_collection_mode(self) -> None:
        plan = self.plan("--index-dir", str(self.tech))
        self.assertTrue(plan["single"])
        self.assertEqual(plan["index_map"], {"tech": self.tech})
        self.assertEqual(plan["chunks"], {"tech": 3})
        self.assertEqual(plan["default"], ["tech"])
        self.assertEqual(plan["skipped"], [])

    def test_plan_index_dir_without_meta_raises(self) -> None:
        empty = self.tmp / "empty-index"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError) as caught:
            self.plan("--index-dir", str(empty))
        self.assertIn("index_meta.json", str(caught.exception))


# ---------------------------------------------------------------- 前端静态契约


class FrontendHtmlTests(unittest.TestCase):
    """单文件前端不能靠构建工具兜底，这些契约只能静态检查。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = web_app.INDEX_HTML.read_text(encoding="utf-8")
        cls.scripts = re.findall(r"<script>(.*?)</script>", cls.html, re.S)
        cls.script = "\n".join(cls.scripts)
        cls.ids = set(re.findall(r'id="([^"]+)"', cls.html))

    def test_new_feature_ids_exist(self) -> None:
        expected = {
            # 历史面板
            "history-panel", "history-list", "history-clear",
            # 仓库筛选
            "filters", "filter-chips", "filter-summary",
            # 集合选择 + 集合徽标
            "collection-filters", "collection-chips", "collection-hint",
            "collection-badge-tpl",
            # 主题 / 统计 / 错误重试
            "theme-toggle", "stats-panel", "stats-body", "stats-btn", "stats-close",
            "retry-btn", "error-text", "status-chips",
            # 状态容器
            "loading", "empty-state", "toast",
        }
        self.assertEqual(sorted(expected - self.ids), [],
                         "index.html 缺少新增元素 id")
        legacy = {"query-input", "search-btn", "mode-ask", "mode-search",
                  "results", "answer-panel", "status-bar"}
        self.assertEqual(sorted(legacy - self.ids), [], "原有 id 被删了")

    def test_script_ids_are_all_defined(self) -> None:
        """JS 里 $('x') / getElementById('x') 引用的 id 必须都在 HTML 里。"""
        self.assertTrue(self.scripts, "index.html 里没有内联 <script>")
        used = set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", self.script))
        used |= set(re.findall(r"getElementById\('([A-Za-z0-9_-]+)'\)", self.script))
        # on('x', ...) 也是按 id 找元素，一并检查
        used |= set(re.findall(r"on\('([A-Za-z0-9_-]+)'", self.script))
        # 新增功能的关键 id 必须真的被脚本用到，而不是只写在 HTML 里
        exercised = {"query-input", "history-panel", "filter-chips",
                     "theme-toggle", "stats-panel", "retry-btn",
                     "collection-chips", "collection-filters",
                     "collection-badge-tpl"}
        self.assertEqual(sorted(exercised - used), [])
        self.assertEqual(sorted(used - self.ids - {"citation-"}), [])

    def test_no_external_resources(self) -> None:
        lowered = self.html.lower()
        for needle in ('src="http', 'href="http', "@import", "<script src=",
                       "cdn.", "unpkg", "jsdelivr", "googleapis"):
            self.assertNotIn(needle, lowered, f"index.html 引用了外部资源 {needle!r}")
        self.assertNotIn("<link", lowered)
        self.assertIn("-apple-system", self.html)      # 系统字体栈
        self.assertIn("Microsoft YaHei", self.html)

    def test_highlight_escapes_before_inserting_mark(self) -> None:
        self.assertIn("function escapeHtml", self.script)
        for entity in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
            self.assertIn(entity, self.script)
        # 允许写 innerHTML 的地方只有 highlightHtml() 的返回值
        assignments = re.findall(r"\.innerHTML\s*=\s*([^;]+);", self.script)
        self.assertEqual(len(assignments), 2, assignments)
        for expression in assignments:
            self.assertTrue(expression.strip().startswith("highlightHtml("), expression)
        self.assertIn(
            "'<mark class=\"hl\">' + escapeHtml(match[0]) + '</mark>'",
            self.script,
        )
        # 非高亮字段一律走 textContent
        self.assertIn("'crumb', item.breadcrumb", self.script)
        self.assertNotRegex(self.script, r"innerHTML\s*=\s*item\.")

    def test_history_storage_contract(self) -> None:
        self.assertIn("'kb.searchHistory'", self.script)
        self.assertIn("HISTORY_LIMIT = 20", self.script)
        self.assertIn("window.localStorage.setItem", self.script)
        self.assertRegex(self.script, r"ts:\s*Date\.now\(\)")
        # 同 q + mode 去重后移到最前
        self.assertIn("item.q === query && item.mode === mode", self.script)
        self.assertIn("rest.unshift({ q: query, mode: mode, ts: Date.now() })",
                      self.script)
        # 相对时间四档
        for needle in ("'刚刚'", "'分钟前'", "'小时前'"):
            self.assertIn(needle, self.script)
        self.assertIn("pad2(date.getMonth() + 1) + '-' + pad2(date.getDate())",
                      self.script)

    def test_url_state_contract(self) -> None:
        self.assertIn("window.history.replaceState", self.script)
        self.assertIn("url.searchParams.set('q', query)", self.script)
        self.assertIn("url.searchParams.set('mode', mode)", self.script)
        self.assertIn("new URLSearchParams(window.location.search)", self.script)
        self.assertIn("if (initial.q)", self.script)   # 载入时自动复现查询
        self.assertIn("applyMode(initial.mode)", self.script)

    def test_keyboard_shortcut_contract(self) -> None:
        self.assertIn("event.key === '/'", self.script)
        self.assertIn("!isTypingTarget(event.target)", self.script)
        self.assertIn("event.key === 'Escape'", self.script)
        self.assertIn("event.key === 'ArrowDown' || event.key === 'ArrowUp'",
                      self.script)
        self.assertIn("openSelected", self.script)
        self.assertIn("setAttribute('data-action', 'open')", self.script)
        self.assertIn("moveSelection", self.script)

    def test_theme_contract(self) -> None:
        self.assertIn("'(prefers-color-scheme: dark)'", self.script)
        self.assertIn("var THEMES = ['auto', 'light', 'dark'];", self.script)
        self.assertIn("'kb.theme'", self.script)
        self.assertIn("setAttribute('data-theme'", self.script)
        self.assertIn(':root[data-theme="dark"]', self.html)
        self.assertIn("☀", self.html)
        self.assertIn("🌙", self.script)

    def test_copy_citation_contract(self) -> None:
        self.assertIn("navigator.clipboard.writeText", self.script)
        self.assertIn("document.execCommand('copy')", self.script)
        self.assertIn("'已复制'", self.script)
        self.assertIn("1500", self.script)             # 1.5 秒后还原
        self.assertIn("'[' + label + '](' + target + ')'", self.script)
        self.assertIn("item.breadcrumb ? item.breadcrumb : path", self.script)
        self.assertIn("item.line_start ? ':' + item.line_start : ''", self.script)

    def test_loading_empty_error_and_refusal_states(self) -> None:
        self.assertEqual(self.html.count('class="skeleton"'), 3)
        self.assertIn("'搜索中…'", self.script)        # 按钮文案
        self.assertIn('id="retry-btn"', self.html)
        self.assertIn("showError", self.script)
        for needle in ("换个关键词", "去掉筛选条件", "试试问答模式"):
            self.assertIn(needle, self.script)
        # 拒答要有独立的警示样式，并保留检索结果入口
        self.assertIn("refuse-reason", self.html)
        self.assertIn(".answer-card.refused", self.html)
        self.assertIn("state.results = (data.hits || []).map(normalizeResult)",
                      self.script)

    def test_stats_panel_contract(self) -> None:
        self.assertIn("openStats", self.script)
        self.assertIn("api('/api/stats')", self.script)
        self.assertIn("state.stats", self.script)
        self.assertIn("index_size_mb", self.script)
        self.assertIn("avg_chars", self.script)
        self.assertIn("stats-table", self.html)

    # ---- 集合选择 / 集合徽标 ----

    def test_collection_chips_render_contract(self) -> None:
        self.assertIn("/api/collections", self.script)
        self.assertIn("function renderCollectionChips", self.script)
        self.assertIn("function toggleCollection", self.script)
        # chip 显示「名字 (块数)」，块数用 filter-count 那个数字样式
        self.assertIn("fmtInt(item.chunks)", self.script)
        self.assertIn("state.collections.forEach", self.script)
        # 至少留一个：全不选时提示并阻止
        self.assertIn("至少要保留一个集合", self.script)
        self.assertIn("if (active.length === 1)", self.script)

    def test_collection_badge_contract(self) -> None:
        # 徽标模板在 HTML 里，JS 克隆后插到 breadcrumb 那一行
        self.assertIn('<template id="collection-badge-tpl">', self.html)
        self.assertIn('class="collection-badge"', self.html)
        self.assertIn("function collectionBadge", self.script)
        self.assertIn("crumbRow.appendChild(collectionBadge(item.collection))",
                      self.script)
        # 颜色走 CSS 变量，深色模式重新定义同一批变量
        self.assertIn("--coll-tech:", self.html)
        self.assertIn("--coll-profile-bg:", self.html)
        dark = self.html.split(':root[data-theme="dark"]', 1)[1]
        self.assertIn("--coll-tech:", dark)
        self.assertIn("--coll-tone-1:", dark)
        # 未知集合按名字散列到 tone-N，至少有四档
        for index in range(4):
            self.assertIn(".tone-%d {" % index, self.html)

    def test_collection_storage_contract(self) -> None:
        self.assertIn("'kb.collections'", self.script)
        self.assertIn("JSON.stringify(state.activeCollections)", self.script)
        self.assertIn("JSON.parse(raw)", self.script)
        self.assertIn("Array.isArray(parsed)", self.script)
        # 存的集合已经不存在时忽略
        self.assertIn("!knownCollection(name)", self.script)
        self.assertIn("function defaultActiveCollections", self.script)

    def test_collection_request_and_url_contract(self) -> None:
        # 请求带 collections=，值来自选中的 chip
        self.assertIn("'&collections=' + encodeURIComponent(collectionQuery())",
                      self.script)
        self.assertIn("function collectionQuery", self.script)
        # URL 同步也带上集合，刷新能恢复
        self.assertIn("url.searchParams.set('collections'", self.script)
        self.assertIn("params.get('collections')", self.script)
        self.assertIn("applyCollections(initial.collections)", self.script)
        # 切集合要清空仓库筛选（结果集变了），两层过滤互不干扰
        self.assertIn("state.activeRepos = {};\n    renderCollectionChips();",
                      self.script)

    def test_stats_panel_groups_by_collection(self) -> None:
        self.assertIn("data.collections", self.script)
        self.assertIn("group.avg_chars", self.script)
        self.assertIn("group.index_size_mb", self.script)
        self.assertIn("renderRepoTable", self.script)
        self.assertIn("'集合 ' + name + ' · '", self.script)


class WebAppSecurityTests(TempCorpus):
    """接口里的敌意文本以 JSON 返回（不渲染），HTML 模板里没有数据拼接。"""

    def test_search_returns_hostile_text_as_json(self) -> None:
        hostile_text = "命中 <script>alert('xss')</script> & <img src=x onerror=alert(1)>"
        hostile_breadcrumb = "<img src=x onerror=alert(1)>"
        hit = SearchHit(
            chunk_id="chunk-x", doc_id="doc-x", repo="JavaGuide",
            rel_path="docs/hashmap.md", breadcrumb=hostile_breadcrumb,
            line_start=1, line_end=2, text=hostile_text, score=0.1,
            source="hybrid", rank=1,
        )
        retriever = FakeRetriever([hit])
        app = create_app(retriever, None, raw_dir=self.raw, editor="code")
        with TestClient(app) as client:
            res = client.get("/api/search", params={"q": "script"})
            index = client.get("/")

        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.headers["content-type"].startswith("application/json"),
                        "命中文本必须走 JSON，浏览器不会把它当 HTML 解析")
        body = res.json()
        self.assertEqual(body["results"][0]["text"], hostile_text)
        self.assertEqual(body["results"][0]["breadcrumb"], hostile_breadcrumb)
        self.assertIn("<script>", res.text)   # JSON 原样返回，靠 content-type 兜底
        # 首页是静态模板，不会把接口数据拼进去
        self.assertNotIn("<script>alert", index.text)
        self.assertNotIn("onerror=alert", index.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
