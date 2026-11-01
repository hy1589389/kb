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

TMP_ROOT = ROOT / ".tmp"

HIT_TEXT = (
    "HashMap 在链表长度达到 8 时转红黑树，低于 6 时退回链表。"
    "红黑树是为了把最坏情况下的查找从 O(n) 降到 O(log n)。"
)


def make_hit(index: int = 1, *, repo: str = "JavaGuide",
             rel_path: str = "docs/hashmap.md", line_start: int = 12,
             line_end: int = 40, text: str | None = None,
             score: float = 0.03) -> SearchHit:
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
    """

    created = 0

    def __init__(self, hits: list[SearchHit]):
        type(self).created += 1
        self.hits = list(hits)
        self.calls: list[dict] = []
        self.index_dir = str(TMP_ROOT / "index")
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
               use_sparse: bool = True, use_rerank: bool = False):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "use_dense": use_dense,
            "use_sparse": use_sparse,
            "use_rerank": use_rerank,
        })
        return self.hits[:top_k]


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
        }
        self.assertEqual(set(body), expected)
        self.assertTrue(body["ok"])
        self.assertTrue(body["ready"])
        self.assertTrue(body["index_ready"])
        self.assertEqual(body["chunks"], 3)
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
            {"chunks", "repos", "avg_chars", "index_dir", "index_size_mb"},
        )
        self.assertEqual(body["chunks"], 3)
        self.assertEqual(body["avg_chars"], 200)          # (100 + 200 + 300) / 3
        self.assertEqual(body["index_dir"], str(self.index))
        self.assertIsInstance(body["repos"], list)

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
                     "theme-toggle", "stats-panel", "retry-btn"}
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
