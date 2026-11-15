"""Phase 4 生成层测试（标准库 unittest，不联网，不需要索引和模型）。

运行：
    python tests/test_generate.py
    python -m unittest tests.test_generate -v

每个用例对应一条真实会出错的地方：

  · 密钥顺序         —— 环境变量、.env、注册表三层，顺序错了就会用错 key
  · 注册表回退       —— 本机 key 在系统作用域，进程里读不到；代码要在
  · 错误信息脱敏     —— HTTP 错误若把请求头带出来，Authorization 就漏了
  · 上下文预算       —— 不截断会撞模型窗口；乱截会把结论句切掉
  · 检索为空不调模型 —— 白白花钱，而且拿回来的还是拒答
  · INSUFFICIENT     —— 模型说资料不足时不能把标记原样当答案给用户
  · 越界引用         —— [9] 只有 3 条引用，这是幻觉的硬信号，必须留痕
  · 未引用句子计数   —— 评测指标，切句规则错了数字就不可用
  · 拒答时引用为空   —— 拒答正文是一句固定话，不该产生引用统计

测试用的临时目录放在项目内的 `.tmp`：本机系统 %TEMP% 不可写。
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock
from urllib import error as urllib_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows 控制台默认 cp936，中文断言消息会炸，这里强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from kb.config import GenerateConfig, IngestConfig  # noqa: E402
from kb.generate import cite, llm  # noqa: E402
from kb.generate.llm import (  # noqa: E402
    DeepSeekClient,
    FakeLLM,
    KeyNotFoundError,
    LLMError,
    load_api_key,
    parse_dotenv,
)
from kb.generate.pipeline import (  # noqa: E402
    REFUSE_INSUFFICIENT,
    REFUSE_LOW_SCORE,
    REFUSE_NO_HITS,
    Answerer,
)
from kb.generate.prompt import (  # noqa: E402
    INSUFFICIENT,
    build_messages,
    merge_same_location,
    prepare_blocks,
    select_hits,
)
from kb.retrieve.types import SearchHit  # noqa: E402

TMP_ROOT = ROOT / ".tmp"


@contextmanager
def temp_dir(prefix: str = "gen"):
    """工作区内的临时目录，退出时清理。"""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"{prefix}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def chdir(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@contextmanager
def clean_env(name: str):
    """确保测试用的变量在 os.environ 里不存在，退出时恢复原值。"""
    old = os.environ.pop(name, None)
    try:
        yield
    finally:
        if old is not None:
            os.environ[name] = old


def make_hit(index: int, *, text: str | None = None, score: float = 0.02,
             source: str = "hybrid") -> SearchHit:
    """构造一条检索结果。repo/rel_path/行号都是可核对的固定值。"""
    return SearchHit(
        chunk_id=f"chunk-{index}",
        doc_id=f"doc-{index}",
        repo="JavaGuide",
        rel_path=f"docs/java/collection/hashmap-{index}.md",
        breadcrumb=f"JavaGuide > Map（重要） > HashMap 的底层实现 {index}",
        line_start=100 * index,
        line_end=100 * index + 40,
        text="HashMap 在链表长度达到 8 时转为红黑树。" if text is None else text,
        score=score,
        source=source,
        rank=index,
    )


class FakeRetriever:
    """鸭子类型的检索器，记录调用参数并返回预设结果。"""

    def __init__(self, hits: list[SearchHit], *, source: str = "hybrid"):
        self.hits = list(hits)
        self.source = source
        self.calls: list[dict] = []

    def search(self, query: str, top_k: int = 10, *, use_dense: bool = True,
               use_sparse: bool = True, use_rerank: bool = False,
               collections: list[str] | str | None = None):
        call = {
            "query": query,
            "top_k": top_k,
            "use_dense": use_dense,
            "use_sparse": use_sparse,
            "use_rerank": use_rerank,
        }
        # 只有真的传了 collections 才记这一项，老用例的断言不受影响
        if collections:
            call["collections"] = collections
        self.calls.append(call)
        out = self.hits[:top_k]
        return [replace(h, rank=i, source=self.source)
                for i, h in enumerate(out, start=1)]


# --------------------------------------------------------------------------
# 密钥读取
# --------------------------------------------------------------------------

class TestLoadApiKey(unittest.TestCase):
    ENV_NAME = "KB_TEST_KEY"

    def setUp(self):
        self._tmp = temp_dir()
        self.tmp = self._tmp.__enter__()
        self.addCleanup(self._tmp.__exit__, None, None, None)

    def _write_env_file(self, text: str) -> Path:
        path = self.tmp / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_env_has_priority_over_dotenv_and_registry(self):
        dotenv = self._write_env_file(f"{self.ENV_NAME}=sk-from-dotenv\n")
        with clean_env(self.ENV_NAME), chdir(self.tmp):
            os.environ[self.ENV_NAME] = "sk-from-env"
            with mock.patch.object(llm, "DEFAULT_ENV_FILE", dotenv), \
                    mock.patch.object(llm, "_read_windows_env",
                                      return_value="sk-from-registry"):
                try:
                    self.assertEqual(load_api_key(self.ENV_NAME), "sk-from-env")
                finally:
                    os.environ.pop(self.ENV_NAME, None)

    def test_dotenv_used_when_env_missing(self):
        dotenv = self._write_env_file(
            "# 注释行\n"
            f"{self.ENV_NAME} = \"sk-quoted-444\"   # 行尾注释\n"
            "OTHER_KEY=1\n"
        )
        with clean_env(self.ENV_NAME), chdir(self.tmp):
            with mock.patch.object(llm, "DEFAULT_ENV_FILE", dotenv), \
                    mock.patch.object(llm, "_read_windows_env",
                                      return_value="sk-from-registry"):
                self.assertEqual(load_api_key(self.ENV_NAME), "sk-quoted-444")

    def test_registry_used_when_env_and_dotenv_missing(self):
        missing = self.tmp / "no-such.env"
        with clean_env(self.ENV_NAME), chdir(self.tmp):
            with mock.patch.object(llm, "DEFAULT_ENV_FILE", missing), \
                    mock.patch.object(llm, "_read_windows_env",
                                      return_value="sk-from-registry") as m:
                self.assertEqual(load_api_key(self.ENV_NAME), "sk-from-registry")
                m.assert_called_once_with(self.ENV_NAME)

    def test_dotenv_without_target_key_falls_through(self):
        dotenv = self._write_env_file("SOME_OTHER_KEY=whatever\n")
        with clean_env(self.ENV_NAME), chdir(self.tmp):
            with mock.patch.object(llm, "DEFAULT_ENV_FILE", dotenv), \
                    mock.patch.object(llm, "_read_windows_env",
                                      return_value="sk-from-registry"):
                self.assertEqual(load_api_key(self.ENV_NAME), "sk-from-registry")

    def test_key_not_found_message_has_no_key_fragment(self):
        missing = self.tmp / "no-such.env"
        with clean_env(self.ENV_NAME), chdir(self.tmp):
            with mock.patch.object(llm, "DEFAULT_ENV_FILE", missing), \
                    mock.patch.object(llm, "_read_windows_env", return_value=""):
                with self.assertRaises(KeyNotFoundError) as ctx:
                    load_api_key(self.ENV_NAME)
        msg = str(ctx.exception)
        self.assertIn(self.ENV_NAME, msg)          # 告诉用户设哪个变量
        self.assertIn(".env", msg)                 # 以及另一种设置方式
        self.assertNotIn("Bearer", msg)
        # 不能出现任何形似 key 的长串
        self.assertNotRegex(msg, r"sk-[A-Za-z0-9_\-]{8,}")
        self.assertNotRegex(msg, r"[A-Za-z0-9]{32,}")

    def test_registry_skipped_on_non_windows(self):
        with mock.patch.object(llm.sys, "platform", "linux"):
            self.assertEqual(llm._read_windows_env("ANY_KEY"), "")

    @unittest.skipUnless(sys.platform == "win32", "只在 Windows 上验证注册表")
    def test_registry_lookup_on_this_machine(self):
        """本机 key 在系统作用域：直接读注册表应该能拿到（值不打印）。"""
        value = llm._read_windows_env("DEEPSEEK_API_KEY")
        if not value:
            self.skipTest("本机注册表里没有 DEEPSEEK_API_KEY，跳过")
        self.assertGreater(len(value), 8)

    def test_parse_dotenv_forms(self):
        parsed = parse_dotenv(
            "# c\n"
            "A=1\n"
            "B = \"two words\"\n"
            "C='three'\n"
            "export D=4\n"
            "E=pa#ss\n"
            "F=value # trailing\n"
            "broken line\n"
        )
        self.assertEqual(parsed["A"], "1")
        self.assertEqual(parsed["B"], "two words")
        self.assertEqual(parsed["C"], "three")
        self.assertEqual(parsed["D"], "4")
        self.assertEqual(parsed["E"], "pa#ss")       # 紧贴的 # 不是注释
        self.assertEqual(parsed["F"], "value")       # 空白后的 # 是注释
        self.assertNotIn("broken line", parsed)


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------

class TestBuildMessages(unittest.TestCase):
    def test_system_prompt_states_the_rules(self):
        messages = build_messages("问题", [make_hit(1)], max_context_chars=6000)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        system = messages[0]["content"]
        self.assertIn("只能依据", system)
        self.assertIn(INSUFFICIENT, system)
        self.assertIn("[n]", system)
        self.assertIn("中文", system)

    def test_numbering_and_breadcrumb(self):
        hits = [make_hit(1), make_hit(2), make_hit(3)]
        user = build_messages("HashMap 怎么转红黑树", hits,
                              max_context_chars=6000)[1]["content"]
        self.assertIn("[1] 来源：JavaGuide/docs/java/collection/hashmap-1.md:100-140",
                      user)
        self.assertIn("[2] 来源：JavaGuide/docs/java/collection/hashmap-2.md:200-240",
                      user)
        self.assertIn("[3] 来源：", user)
        self.assertIn("标题：JavaGuide > Map（重要） > HashMap 的底层实现 1", user)
        self.assertIn("问题：HashMap 怎么转红黑树", user)
        # 编号顺序 = 传入顺序
        self.assertLess(user.index("[1]"), user.index("[2]"))
        self.assertLess(user.index("[2]"), user.index("[3]"))

    def test_context_truncation_keeps_top_ranked(self):
        hits = [make_hit(i, text="字" * 100) for i in range(1, 6)]
        user = build_messages("q", hits, max_context_chars=250,
                              max_chunk_chars=1500)[1]["content"]
        self.assertIn("资料已截断到前 2 条", user)
        self.assertIn("[2] 来源：", user)
        self.assertNotIn("[3] 来源：", user)

    def test_single_chunk_truncated_to_max_chunk_chars(self):
        hit = make_hit(1, text="甲" * 500)
        messages = build_messages("q", [hit], max_context_chars=6000,
                                  max_chunk_chars=100)
        user = messages[1]["content"]
        self.assertIn("本块已截断", user)
        self.assertNotIn("甲" * 101, user)

    def test_first_chunk_larger_than_budget_still_included(self):
        hit = make_hit(1, text="乙" * 400)
        blocks = prepare_blocks([hit], max_context_chars=100,
                                max_chunk_chars=1500)
        self.assertEqual(len(blocks), 1)
        self.assertLessEqual(len(blocks[0][1]), 100 + len("…（本块已截断）"))

    def test_empty_hits(self):
        user = build_messages("q", [], max_context_chars=6000)[1]["content"]
        self.assertIn("没有检索到任何资料", user)

    def test_zero_budget_gives_no_context(self):
        self.assertEqual(select_hits([make_hit(1)], max_context_chars=0), [])


# --------------------------------------------------------------------------
# 引用校验（纯函数）
# --------------------------------------------------------------------------

class TestCiteValidation(unittest.TestCase):
    def test_extract_refs_keeps_order_and_duplicates(self):
        self.assertEqual(cite.extract_refs("甲[2] 乙[1][2] 丙"),
                         [2, 1, 2])

    def test_extract_refs_ignores_non_numeric_brackets(self):
        self.assertEqual(cite.extract_refs("见 [文档](http://x) 和 [1]"), [1])

    def test_invalid_refs_out_of_range(self):
        self.assertEqual(cite.invalid_refs("a[1] b[9] c[0]", 3), [0, 9])

    def test_invalid_refs_empty_when_all_valid(self):
        self.assertEqual(cite.invalid_refs("a[1] b[3]", 3), [])

    def test_split_sentences_filters_short_and_structural(self):
        text = "第一句话在这里[1]。好。\n## 标题行\n| 表格 |\n```py\nx=1\n```\n第三句话也比较长。"
        sentences = cite.split_sentences(text)
        self.assertIn("第一句话在这里[1]。", sentences)
        self.assertIn("第三句话也比较长。", sentences)
        self.assertNotIn("好。", sentences)
        self.assertFalse(any(s.startswith("#") for s in sentences))
        self.assertFalse(any(s.startswith("|") for s in sentences))

    def test_count_uncited(self):
        text = "甲句有引用标记[1]。乙句没有引用标记。丙句也没有。好。"
        uncited, total = cite.count_uncited(text)
        self.assertEqual((uncited, total), (2, 3))


# --------------------------------------------------------------------------
# Answerer
# --------------------------------------------------------------------------

class TestAnswerer(unittest.TestCase):
    def _answerer(self, hits, responder="", *, source="hybrid", cfg=None):
        retriever = FakeRetriever(hits, source=source)
        fake = FakeLLM(responder)
        merged = {"max_context_chars": 6000, "max_chunk_chars": 1500}
        if cfg:
            merged.update(cfg)
        return Answerer(retriever, fake, cfg=merged), retriever, fake

    def test_normal_answer(self):
        hits = [make_hit(1), make_hit(2), make_hit(3)]
        responder = ("HashMap 在链表长度达到 8 时转为红黑树[1]。"
                     "红黑树的查询复杂度是 O(log n)[2]。")
        answerer, retriever, fake = self._answerer(hits, responder)
        ans = answerer.answer("HashMap 什么时候转红黑树", top_k=3)

        self.assertFalse(ans.refused)
        self.assertEqual(ans.refuse_reason, "")
        self.assertEqual(ans.text, responder)
        self.assertEqual(len(ans.citations), 3)
        for i, (cit, hit) in enumerate(zip(ans.citations, hits), start=1):
            self.assertEqual(cit.ref, i)
            self.assertEqual(cit.chunk_id, hit.chunk_id)
            self.assertEqual(cit.rel_path, hit.rel_path)
            self.assertEqual(cit.line_start, hit.line_start)
        self.assertEqual(ans.model, "fake-llm")
        self.assertIsInstance(ans.latency_ms, int)
        self.assertGreaterEqual(ans.latency_ms, 0)
        self.assertGreater(ans.usage.get("total_tokens", 0), 0)
        self.assertEqual(ans.invalid_refs, [])
        self.assertEqual(ans.uncited_sentences, 0)
        self.assertEqual(ans.sentence_count, 2)
        self.assertEqual(ans.context_chunks, 3)
        # 检索参数透传
        self.assertEqual(retriever.calls[0]["query"], "HashMap 什么时候转红黑树")
        self.assertEqual(retriever.calls[0]["top_k"], 3)
        # 提示词里确实带了资料
        user = fake.calls[0]["messages"][1]["content"]
        self.assertIn("[1] 来源：", user)

    def test_collections_passed_through_to_retriever(self):
        """传了集合就限制检索范围：问技术问题时简历不该进候选。"""
        hits = [make_hit(1), make_hit(2)]
        responder = "HashMap 在链表长度达到 8 时转为红黑树[1]。"
        answerer, retriever, _ = self._answerer(hits, responder)

        answerer.answer("HashMap 什么时候转红黑树", top_k=2,
                        collections=["tech"])

        self.assertEqual(retriever.calls[0]["collections"], ["tech"])

    def test_no_collections_means_omit_parameter(self):
        """不传集合时不能把 collections=None 塞给 retriever。

        单索引的 Retriever 和测试里的假对象都只认前三个关键字参数，
        塞一个 None 进去会让它们直接 TypeError。
        """
        hits = [make_hit(1)]
        responder = "HashMap 在链表长度达到 8 时转为红黑树[1]。"
        answerer, retriever, _ = self._answerer(hits, responder)

        answerer.answer("HashMap 什么时候转红黑树", top_k=1)

        self.assertNotIn("collections", retriever.calls[0])

    def test_insufficient_marker_refuses(self):
        # 默认会做「拒答复核」（见 tests/test_refusal_retry.py），这里要验的是
        # 「标记 -> 拒答」本身，所以显式关掉复核，保证只问一次模型。
        answerer, _, fake = self._answerer(
            [make_hit(1), make_hit(2)], "INSUFFICIENT",
            cfg={"refusal_retry": False})
        ans = answerer.answer("如何做法式红酒炖牛肉", top_k=2)
        self.assertTrue(ans.refused)
        self.assertEqual(ans.refuse_reason, REFUSE_INSUFFICIENT)
        self.assertNotIn("INSUFFICIENT", ans.text)      # 标记不能原样给用户
        self.assertIn("没有找到足够的信息", ans.text)
        self.assertEqual(ans.citations, [])
        self.assertEqual(len(fake.calls), 1)            # 模型确实被问了
        self.assertEqual(len(ans.hits), 2)              # 原始结果保留

    def test_no_hits_refuses_without_calling_llm(self):
        answerer, retriever, fake = self._answerer([], "不该被调用")
        ans = answerer.answer("如何做法式红酒炖牛肉", top_k=5)
        self.assertTrue(ans.refused)
        self.assertEqual(ans.refuse_reason, REFUSE_NO_HITS)
        self.assertEqual(fake.calls, [])
        self.assertEqual(len(retriever.calls), 1)

    def test_empty_question_does_not_hit_retriever(self):
        answerer, retriever, fake = self._answerer([make_hit(1)], "x")
        ans = answerer.answer("   ")
        self.assertTrue(ans.refused)
        self.assertEqual(ans.refuse_reason, REFUSE_NO_HITS)
        self.assertEqual(retriever.calls, [])
        self.assertEqual(fake.calls, [])

    def test_invalid_ref_detected(self):
        hits = [make_hit(1), make_hit(2), make_hit(3)]
        answerer, _, _ = self._answerer(
            hits, "结论是 8[1]。另外参见 [9] 的说法。")
        ans = answerer.answer("q", top_k=3)
        self.assertFalse(ans.refused)
        self.assertIn(9, ans.invalid_refs)
        self.assertEqual(ans.invalid_refs, [9])

    def test_uncited_sentences_counted(self):
        hits = [make_hit(1), make_hit(2)]
        responder = ("HashMap 在链表长度达到 8 时转为红黑树[1]。"
                     "红黑树查询复杂度为 O(log n)[2]。"
                     "这种方法减少了哈希冲突。"
                     "是的。")
        answerer, _, _ = self._answerer(hits, responder)
        ans = answerer.answer("q", top_k=2)
        self.assertEqual(ans.sentence_count, 3)   # 「是的」太短，不计入
        self.assertEqual(ans.uncited_sentences, 1)

    def test_context_truncation_limits_citations(self):
        hits = [make_hit(i, text="字" * 100) for i in range(1, 4)]
        answerer, _, _ = self._answerer(
            hits, "甲[1]。乙[2]。", cfg={"max_context_chars": 250})
        ans = answerer.answer("q", top_k=3)
        self.assertEqual(ans.context_chunks, 2)
        self.assertEqual(len(ans.citations), 2)
        self.assertEqual(len(ans.hits), 3)
        self.assertEqual(ans.citations[0].chunk_id, hits[0].chunk_id)
        self.assertEqual(ans.citations[1].chunk_id, hits[1].chunk_id)

    def test_min_top_score_skipped_for_hybrid(self):
        answerer, _, fake = self._answerer(
            [make_hit(1, score=0.02)], "答案[1]。",
            cfg={"min_top_score": 0.9})
        with self.assertLogs("kb.generate", level="WARNING") as logs:
            ans = answerer.answer("q", top_k=1)
        self.assertFalse(ans.refused)
        self.assertEqual(len(fake.calls), 1)
        self.assertTrue(any("min_top_score" in line for line in logs.output))

    def test_min_top_score_applies_to_single_path(self):
        answerer, _, fake = self._answerer(
            [make_hit(1, score=0.10, source="dense")], "答案[1]。",
            source="dense", cfg={"min_top_score": 0.5})
        ans = answerer.answer("q", top_k=1)
        self.assertTrue(ans.refused)
        self.assertEqual(ans.refuse_reason, REFUSE_LOW_SCORE)
        self.assertEqual(fake.calls, [])

    def test_min_top_score_zero_disables(self):
        answerer, _, fake = self._answerer(
            [make_hit(1, score=0.0, source="dense")], "答案[1]。",
            source="dense", cfg={"min_top_score": 0})
        self.assertFalse(answerer.answer("q", top_k=1).refused)
        self.assertEqual(len(fake.calls), 1)

    def test_use_rerank_passed_through(self):
        answerer, retriever, _ = self._answerer([make_hit(1)], "答案[1]。")
        answerer.answer("q", top_k=1, use_rerank=True)
        self.assertTrue(retriever.calls[0]["use_rerank"])

    def test_to_dict_is_json_serializable(self):
        answerer, _, _ = self._answerer([make_hit(1)], "答案[1]。")
        ans = answerer.answer("q", top_k=1)
        payload = json.loads(json.dumps(ans.to_dict(), ensure_ascii=False))
        self.assertEqual(payload["question"], "q")
        self.assertEqual(payload["citations"][0]["ref"], 1)
        self.assertEqual(payload["hits"][0]["chunk_id"], "chunk-1")
        self.assertIn("invalid_refs", payload)
        self.assertIn("uncited_sentences", payload)


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------

class TestRender(unittest.TestCase):
    def test_render_has_answer_and_every_citation(self):
        hits = [make_hit(1), make_hit(2)]
        answerer = Answerer(FakeRetriever(hits), FakeLLM("第一句[1]。第二句[2]。"),
                            cfg={"max_context_chars": 6000})
        ans = answerer.answer("HashMap 转红黑树", top_k=2)
        text = ans.render()
        self.assertIn("第一句[1]。第二句[2]。", text)
        self.assertIn("JavaGuide/docs/java/collection/hashmap-1.md:100-140", text)
        self.assertIn("JavaGuide/docs/java/collection/hashmap-2.md:200-240", text)
        self.assertIn("JavaGuide > Map（重要） > HashMap 的底层实现 1", text)
        self.assertIn("[1]", text)
        self.assertIn("[2]", text)

    def test_render_warns_about_invalid_refs(self):
        answerer = Answerer(FakeRetriever([make_hit(1)]), FakeLLM("引用了 [9]。"),
                            cfg={"max_context_chars": 6000})
        ans = answerer.answer("q", top_k=1)
        self.assertIn("引用了不存在的编号", ans.render())

    def test_render_refusal(self):
        answerer = Answerer(FakeRetriever([]), FakeLLM("不该调用"))
        text = answerer.answer("q").render()
        self.assertIn("拒答", text)
        self.assertIn(REFUSE_NO_HITS, text)
        self.assertIn("没有找到足够的信息", text)

    def test_render_marks_truncated_citation(self):
        hit = make_hit(1, text="字" * 300)
        answerer = Answerer(FakeRetriever([hit]), FakeLLM("答案[1]。"),
                            cfg={"max_chunk_chars": 100})
        ans = answerer.answer("q", top_k=1)
        self.assertTrue(ans.citations[0].truncated)
        self.assertEqual(ans.citations[0].full_length, 300)
        self.assertLess(len(ans.citations[0].text), 300)


# --------------------------------------------------------------------------
# DeepSeek 客户端（不联网：mock urlopen）
# --------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200):
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class TestDeepSeekClient(unittest.TestCase):
    KEY = "sk-unit-test-key-0123456789"

    def _client(self) -> DeepSeekClient:
        return DeepSeekClient(model="deepseek-chat", api_key=self.KEY,
                              timeout=1.0)

    def test_repr_and_str_do_not_contain_key(self):
        client = self._client()
        self.assertNotIn(self.KEY, repr(client))
        self.assertNotIn(self.KEY, str(client))

    def test_success_parses_content_and_usage(self):
        payload = {
            "choices": [{"message": {"role": "assistant",
                                     "content": "答案在这里[1]。"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10,
                      "total_tokens": 30},
        }
        with mock.patch.object(llm.urllib_request, "urlopen",
                               return_value=_FakeResponse(
                                   json.dumps(payload).encode("utf-8"))):
            text, usage = self._client().complete(
                [{"role": "user", "content": "hi"}])
        self.assertEqual(text, "答案在这里[1]。")
        self.assertEqual(usage["total_tokens"], 30)

    def test_http_error_message_has_no_key(self):
        err = urllib_error.HTTPError(
            "https://api.deepseek.com/chat/completions", 401, "Unauthorized",
            {}, io.BytesIO(b'{"error":{"message":"Authentication Fails"}}'))
        with mock.patch.object(llm.urllib_request, "urlopen", side_effect=err):
            with self.assertRaises(LLMError) as ctx:
                self._client().complete([{"role": "user", "content": "hi"}])
        msg = str(ctx.exception)
        self.assertNotIn(self.KEY, msg)
        self.assertNotIn("Bearer", msg)
        self.assertNotIn("Authorization", msg)
        self.assertIn("401", msg)

    def test_network_error_message_has_no_key(self):
        err = urllib_error.URLError("timed out")
        with mock.patch.object(llm.urllib_request, "urlopen", side_effect=err):
            with self.assertRaises(LLMError) as ctx:
                self._client().complete([{"role": "user", "content": "hi"}])
        msg = str(ctx.exception)
        self.assertNotIn(self.KEY, msg)
        self.assertNotIn("Bearer", msg)

    def test_key_echoed_by_some_error_is_sanitized(self):
        """防御性用例：万一异常文本里带了 key，也要被抹掉。"""
        err = urllib_error.HTTPError(
            "https://api.deepseek.com/chat/completions", 400, "Bad Request",
            {}, io.BytesIO(f'{{"error":"bad key {self.KEY}"}}'.encode("utf-8")))
        with mock.patch.object(llm.urllib_request, "urlopen", side_effect=err):
            with self.assertRaises(LLMError) as ctx:
                self._client().complete([{"role": "user", "content": "hi"}])
        self.assertNotIn(self.KEY, str(ctx.exception))
        self.assertIn("***", str(ctx.exception))

    def test_empty_api_key_raises_key_not_found_without_leak(self):
        with self.assertRaises(KeyNotFoundError) as ctx:
            DeepSeekClient(model="m", api_key="", api_key_env="KB_TEST_KEY")
        self.assertNotRegex(str(ctx.exception), r"sk-[A-Za-z0-9_\-]{8,}")

    def test_bad_response_shape_raises_llm_error(self):
        with mock.patch.object(llm.urllib_request, "urlopen",
                               return_value=_FakeResponse(b"{}")):
            with self.assertRaises(LLMError):
                self._client().complete([{"role": "user", "content": "hi"}])

    def test_invalid_json_raises_llm_error(self):
        with mock.patch.object(llm.urllib_request, "urlopen",
                               return_value=_FakeResponse(b"not json")):
            with self.assertRaises(LLMError):
                self._client().complete([{"role": "user", "content": "hi"}])


# --------------------------------------------------------------------------
# FakeLLM 与配置
# --------------------------------------------------------------------------

class TestFakeLLM(unittest.TestCase):
    def test_fixed_string(self):
        fake = FakeLLM("固定回答")
        text, usage = fake.complete([{"role": "user", "content": "x"}])
        self.assertEqual(text, "固定回答")
        self.assertIn("total_tokens", usage)
        self.assertEqual(len(fake.calls), 1)

    def test_callable_with_usage(self):
        def responder(messages):
            return f"看到 {len(messages)} 条消息", {"total_tokens": 7}

        fake = FakeLLM(responder)
        text, usage = fake.complete([{"role": "user", "content": "x"},
                                     {"role": "user", "content": "y"}])
        self.assertEqual(text, "看到 2 条消息")
        self.assertEqual(usage["total_tokens"], 7)

    def test_temperature_recorded(self):
        fake = FakeLLM("x")
        fake.complete([], temperature=0.3, max_tokens=64)
        self.assertEqual(fake.calls[0]["temperature"], 0.3)
        self.assertEqual(fake.calls[0]["max_tokens"], 64)

    def test_rejects_bad_responder(self):
        with self.assertRaises(TypeError):
            FakeLLM(123)


class TestGenerateConfig(unittest.TestCase):
    def test_project_config_parsed(self):
        cfg = IngestConfig.load(ROOT / "config.toml")
        gen = cfg.generate
        self.assertIsInstance(gen, GenerateConfig)
        self.assertEqual(gen.model, "deepseek-chat")
        self.assertEqual(gen.api_key_env, "DEEPSEEK_API_KEY")
        self.assertEqual(gen.max_context_chars, 6000)
        self.assertEqual(gen.max_chunk_chars, 1500)
        self.assertEqual(gen.top_k, 5)
        self.assertEqual(gen.min_top_score, 0.0)
        self.assertIsInstance(gen.min_top_score, float)
        # 没有 key 字面量混进配置
        self.assertNotIn("sk-", (ROOT / "config.toml").read_text(encoding="utf-8"))

    def test_retrieval_config_still_parsed(self):
        cfg = IngestConfig.load(ROOT / "config.toml")
        self.assertEqual(cfg.retrieval.final_top_k, 10)
        self.assertEqual(cfg.retrieval.rrf_k, 60)

    def test_defaults_without_file(self):
        gen = GenerateConfig.load(ROOT / "no-such-config.toml")
        self.assertEqual(gen.model, "deepseek-chat")
        self.assertEqual(gen.max_context_chars, 6000)

    def test_to_answerer_cfg(self):
        gen = GenerateConfig()
        cfg = gen.to_answerer_cfg()
        self.assertIn("max_context_chars", cfg)
        self.assertNotIn("model", cfg)
        self.assertNotIn("api_key_env", cfg)


# --------------------------------------------------------------------------
# CLI 不破坏原有子命令
# --------------------------------------------------------------------------

class TestCli(unittest.TestCase):
    def _help(self, *extra: str) -> str:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "ingest.py"), *extra, "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def test_all_subcommands_listed(self):
        out = self._help()
        for name in ("ingest", "index", "search", "ask"):
            self.assertIn(name, out)

    def test_ask_help_has_expected_flags(self):
        out = self._help("ask")
        for flag in ("--index-dir", "--top-k", "--json", "--show-hits",
                     "--no-rerank", "--model", "--temperature",
                     "--max-context-chars"):
            self.assertIn(flag, out)


# --------------------------------------------------------------------------
# 同一原文位置的块要归并成一条引用
#
# Phase 1 把一个超长 section 切成多个 part 时，每个 part 都拿到了整节的
# line_start/line_end，于是引用列表会出现「4 条引用只对应 2 个位置」。
# 这些 part 内容不同，所以是合并而不是去重。
# --------------------------------------------------------------------------

class TestMergeSameLocation(unittest.TestCase):
    def _hit(self, text: str, path="a/b.md", start=10, end=20, score=0.5):
        return SearchHit(
            chunk_id=f"c{abs(hash(text)) % 10000}", doc_id="d", repo="r",
            rel_path=path, breadcrumb="x > y", line_start=start, line_end=end,
            text=text, score=score, source="hybrid",
        )

    def test_same_location_merged_into_one(self):
        hits = [
            self._hit("第一段内容", start=222, end=297),
            self._hit("第二段内容", start=222, end=297),
            self._hit("别的文件", start=1, end=5),
        ]
        merged = merge_same_location(hits)
        self.assertEqual(len(merged), 2)
        self.assertEqual((merged[0].line_start, merged[0].line_end), (222, 297))
        self.assertIn("第一段内容", merged[0].text)
        self.assertIn("第二段内容", merged[0].text)

    def test_different_locations_are_kept(self):
        hits = [
            self._hit("同文件不同行", start=222, end=297),
            self._hit("同文件换一行", start=300, end=320),
        ]
        self.assertEqual(len(merge_same_location(hits)), 2)

    def test_order_follows_first_occurrence(self):
        hits = [
            self._hit("A", path="z.md", start=1, end=2),
            self._hit("B", path="z.md", start=1, end=2),
            self._hit("C", path="a.md", start=1, end=2),
        ]
        merged = merge_same_location(hits)
        self.assertEqual([h.rel_path for h in merged], ["z.md", "a.md"])
        self.assertEqual(merged[0].text, "A\n\nB")

    def test_empty_list(self):
        self.assertEqual(merge_same_location([]), [])

    def test_merges_before_budgeting(self):
        """合并要发生在算 context 预算之前，否则重复块会白白吃掉预算。"""
        dup1 = self._hit("重复块正文" * 30, start=1, end=9)
        dup2 = self._hit("重复块正文" * 30, start=1, end=9)
        other = self._hit("另一个位置" * 30, start=50, end=60)
        blocks = prepare_blocks([dup1, dup2, other],
                                max_context_chars=10000, max_chunk_chars=1500)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0][0].line_start, 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
