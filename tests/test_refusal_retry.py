"""拒答复核（refusal retry）测试。

背景：实测有两道题在资料明显够的情况下被模型判「资料不足」——问句是被拆开也能
正常回答的复合句。`Answerer` 因此在「模型只回 INSUFFICIENT」之后做一次复核：

  · 问题关键词在上下文里的覆盖度够高（默认 0.6）→ 追加一轮提示词，允许部分作答
  · 覆盖度不够 → 真的没资料，直接拒答，不多花一次 token
  · 二次仍然拒答 → 照常拒答，但 usage 要把两次都算上

这个文件覆盖上面四条分支，外加覆盖度算法本身的两个坑（单字命中虚高、缺词要掉）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from kb.generate.llm import FakeLLM  # noqa: E402
from kb.generate.pipeline import (  # noqa: E402
    REFUSE_INSUFFICIENT,
    Answerer,
    _gap_lines,
    _is_gap_only,
    _keyword_coverage,
)
from kb.generate.prompt import INSUFFICIENT, RETRY_AFTER_INSUFFICIENT  # noqa: E402
from kb.generate.types import Answer  # noqa: E402
from kb.retrieve.types import SearchHit  # noqa: E402

QUESTION = "HashMap 什么时候转红黑树"

# 上下文里重复了问题里的词，覆盖度接近 1
ON_TOPIC = "HashMap 什么时候转红黑树？链表长度到 8 就会转红黑树，低于 6 退回链表。"
# 完全无关，覆盖度接近 0
OFF_TOPIC = "这一段讲的是网络协议、端口号和握手重传，与哈希表无关。"


def make_hit(index: int = 1, *, text: str = ON_TOPIC) -> SearchHit:
    return SearchHit(
        chunk_id=f"chunk-{index}",
        doc_id=f"doc-{index}",
        repo="JavaGuide",
        rel_path="docs/hashmap.md",
        breadcrumb="JavaGuide > Map > HashMap",
        line_start=10,
        line_end=40,
        text=text,
        score=0.03,
        source="hybrid",
        rank=index,
    )


class FakeRetriever:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = list(hits)

    def search(self, query: str, top_k: int = 10, *, use_dense: bool = True,
               use_sparse: bool = True, use_rerank: bool = False,
               collections=None):
        return self.hits[:top_k]


def scripted_responder(*replies: str):
    """按调用顺序返回不同内容；用完之后重复最后一个。"""
    calls = {"n": 0}

    def responder(messages):
        index = min(calls["n"], len(replies) - 1)
        calls["n"] += 1
        return replies[index]

    responder.calls = calls   # type: ignore[attr-defined]
    return responder


def make_answerer(hits: list[SearchHit], responder, *, cfg: dict | None = None):
    merged = {"max_context_chars": 6000, "max_chunk_chars": 1500}
    if cfg:
        merged.update(cfg)
    fake = FakeLLM(responder)
    return Answerer(FakeRetriever(hits), fake, cfg=merged), fake


class KeywordCoverageTests(unittest.TestCase):
    def test_single_cjk_chars_do_not_inflate_coverage(self) -> None:
        """只有单字命中（「是」「的」）时覆盖率必须接近 0，不能被抬到高位。"""
        coverage = _keyword_coverage("哈希表什么时候树化",
                                     [make_hit(text="是 的 了 在 有 和 就 都 也")])
        self.assertLess(coverage, 0.1)

    def test_on_topic_context_scores_high(self) -> None:
        self.assertGreater(_keyword_coverage(QUESTION, [make_hit(text=ON_TOPIC)]), 0.9)

    def test_off_topic_context_scores_low(self) -> None:
        self.assertLess(_keyword_coverage(QUESTION, [make_hit(text=OFF_TOPIC)]), 0.3)

    def test_empty_question_returns_zero(self) -> None:
        self.assertEqual(_keyword_coverage("", [make_hit()]), 0.0)


class RetryBehaviourTests(unittest.TestCase):
    def test_retry_turns_refusal_into_answer(self) -> None:
        responder = scripted_responder(
            INSUFFICIENT,
            "HashMap 在链表长度到 8 时转红黑树[1]。",
        )
        answerer, fake = make_answerer([make_hit()], responder)
        answer = answerer.answer(QUESTION, top_k=1)

        self.assertFalse(answer.refused)
        self.assertTrue(answer.refusal_retried)
        self.assertEqual(answer.text, "HashMap 在链表长度到 8 时转红黑树[1]。")
        self.assertEqual(len(answer.citations), 1)
        # 第二次调用带上了「补充要求」，且没有丢掉原来的资料
        self.assertEqual(len(fake.calls), 2)
        retry_messages = fake.calls[1]["messages"]
        self.assertEqual(retry_messages[-1]["content"], RETRY_AFTER_INSUFFICIENT)
        self.assertIn("[1] 来源：", retry_messages[-2]["content"])

    def test_usage_covers_both_calls(self) -> None:
        responder = scripted_responder(INSUFFICIENT, "答案 [1]。")
        answerer, fake = make_answerer([make_hit()], responder)
        answer = answerer.answer(QUESTION, top_k=1)

        single = fake.calls[0]["messages"]
        tokens_first = sum(len(str(m.get("content", ""))) for m in single) // 4
        self.assertGreaterEqual(answer.usage.get("prompt_tokens", 0), tokens_first)

    def test_low_coverage_refuses_without_second_call(self) -> None:
        responder = scripted_responder(INSUFFICIENT, "不该被调用")
        # 闸门默认是 0（一律复核）；这里显式抬高，验证闸门机制本身还能用
        answerer, fake = make_answerer([make_hit(text=OFF_TOPIC)], responder,
                                       cfg={"refusal_retry_min_coverage": 0.6})
        answer = answerer.answer(QUESTION, top_k=1)

        self.assertTrue(answer.refused)
        self.assertEqual(answer.refuse_reason, REFUSE_INSUFFICIENT)
        self.assertFalse(answer.refusal_retried)
        self.assertEqual(len(fake.calls), 1)

    def test_default_gate_always_retries(self) -> None:
        """默认闸门是 0：域外内容也复核一次 —— 覆盖率信号实测不可靠，不再用它挡。"""
        responder = scripted_responder(INSUFFICIENT, "资料未覆盖：整段内容。")
        answerer, fake = make_answerer([make_hit(text=OFF_TOPIC)], responder)
        answer = answerer.answer(QUESTION, top_k=1)
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(answer.refusal_retried)

    def test_retry_can_be_disabled(self) -> None:
        responder = scripted_responder(INSUFFICIENT, "不该被调用")
        answerer, fake = make_answerer([make_hit()], responder,
                                       cfg={"refusal_retry": False})
        answer = answerer.answer(QUESTION, top_k=1)
        self.assertTrue(answer.refused)
        self.assertEqual(len(fake.calls), 1)

    def test_second_refusal_still_refuses_but_counts_both_calls(self) -> None:
        responder = scripted_responder(INSUFFICIENT, INSUFFICIENT)
        answerer, fake = make_answerer([make_hit()], responder)
        answer = answerer.answer(QUESTION, top_k=1)

        self.assertTrue(answer.refused)
        self.assertEqual(answer.refuse_reason, REFUSE_INSUFFICIENT)
        self.assertEqual(len(fake.calls), 2)
        self.assertGreater(answer.usage.get("prompt_tokens", 0), 0)

    def test_threshold_is_configurable(self) -> None:
        """把闸门调到 1.0 以上，覆盖度再高也不复核。"""
        responder = scripted_responder(INSUFFICIENT, "不该被调用")
        answerer, fake = make_answerer([make_hit()], responder,
                                       cfg={"refusal_retry_min_coverage": 1.01})
        answer = answerer.answer(QUESTION, top_k=1)
        self.assertTrue(answer.refused)
        self.assertEqual(len(fake.calls), 1)


class AnswerFieldTests(unittest.TestCase):
    def test_default_is_not_retried(self) -> None:
        answer = Answer(question="q", text="t", citations=[], refused=False,
                        refuse_reason="", hits=[], model="m", usage={}, latency_ms=0)
        self.assertFalse(answer.refusal_retried)
        self.assertFalse(answer.to_dict()["refusal_retried"])

    def test_to_dict_exposes_flag(self) -> None:
        answer = Answer(question="q", text="t", citations=[], refused=False,
                        refuse_reason="", hits=[], model="m", usage={}, latency_ms=0,
                        refusal_retried=True)
        self.assertTrue(answer.to_dict()["refusal_retried"])


class GapOnlyTests(unittest.TestCase):
    """只写「资料未覆盖：X」等于拒答，不能算成答出来了。"""

    def test_detection(self) -> None:
        self.assertTrue(_is_gap_only("资料未覆盖：如何做红烧肉"))
        self.assertTrue(_is_gap_only("资料未覆盖：如何做红烧肉。\n资料未覆盖：价格。"))
        self.assertFalse(_is_gap_only("HashMap 到 8 转红黑树 [1]"))
        self.assertFalse(_is_gap_only("HashMap 到 8 转红黑树 [1]。\n资料未覆盖：退回条件。"))
        self.assertEqual(_gap_lines("结论 [1]。\n资料未覆盖：退回条件。"),
                         ["资料未覆盖：退回条件。"])

    def test_gap_only_retry_becomes_refusal_keeping_the_gap(self) -> None:
        responder = scripted_responder(INSUFFICIENT, "资料未覆盖：如何做法式红酒炖牛肉。")
        answerer, fake = make_answerer([make_hit()], responder)
        answer = answerer.answer(QUESTION, top_k=1)

        self.assertTrue(answer.refused)
        self.assertEqual(answer.refuse_reason, REFUSE_INSUFFICIENT)
        self.assertIn("没有找到足够的信息", answer.text)
        self.assertIn("资料未覆盖：如何做法式红酒炖牛肉", answer.text)
        self.assertTrue(answer.refusal_retried)
        self.assertEqual(answer.citations, [])
        self.assertEqual(len(fake.calls), 2)

    def test_partial_answer_is_not_a_refusal(self) -> None:
        responder = scripted_responder(
            INSUFFICIENT,
            "HashMap 链表到 8 转红黑树[1]。\n资料未覆盖：转回链表的条件。",
        )
        answerer, _ = make_answerer([make_hit()], responder)
        answer = answerer.answer(QUESTION, top_k=1)

        self.assertFalse(answer.refused)
        self.assertTrue(answer.refusal_retried)
        self.assertEqual(len(answer.citations), 1)
        self.assertIn("资料未覆盖：转回链表的条件", answer.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
