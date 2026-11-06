"""LLM 名单式精排的测试：只测解析和兜底，不联网。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb.generate.llm import FakeLLM, LLMError  # noqa: E402
from kb.generate.rank import LLMReranker, RERANK_PROMPT  # noqa: E402


def hits(count: int = 5) -> list[dict]:
    return [
        {"chunk_id": f"c{index}", "breadcrumb": f"标题 {index}",
         "text": f"片段 {index}", "repo": "JavaGuide", "rel_path": "a.md",
         "line_start": index}
        for index in range(1, count + 1)
    ]


class LLMRerankerTests(unittest.TestCase):
    def test_reorders_by_model_answer(self) -> None:
        llm = FakeLLM("3,1")
        picked = LLMReranker(llm).rerank("token怎么控制", hits())
        self.assertEqual([item["chunk_id"] for item in picked], ["c3", "c1"])
        self.assertEqual(len(llm.calls), 1)
        call = llm.calls[0]
        self.assertEqual(call["messages"][0]["content"], RERANK_PROMPT)
        self.assertEqual(call["temperature"], 0.0)
        # 候选要带编号，模型才答得出「3,1」
        self.assertIn("[1] 标题 1", call["messages"][1]["content"])
        self.assertIn("问题：token怎么控制", call["messages"][1]["content"])

    def test_zero_means_nothing_relevant(self) -> None:
        picked = LLMReranker(FakeLLM("0")).rerank("token怎么控制", hits())
        self.assertEqual(picked, [])

    def test_drops_duplicate_and_out_of_range_indices(self) -> None:
        picked = LLMReranker(FakeLLM("2,2,9,1")).rerank("q", hits(5))
        self.assertEqual([item["chunk_id"] for item in picked], ["c2", "c1"])

    def test_returns_none_on_garbage(self) -> None:
        for answer in ("", "不知道", "a,b,c"):
            with self.subTest(answer=answer):
                self.assertIsNone(LLMReranker(FakeLLM(answer)).rerank("q", hits()))

    def test_returns_none_when_unavailable_or_failed(self) -> None:
        self.assertIsNone(LLMReranker(None).rerank("q", hits()))
        self.assertFalse(LLMReranker(None).available)

        def boom(_messages):
            raise LLMError("接口 500")

        self.assertIsNone(LLMReranker(FakeLLM(boom)).rerank("q", hits()))

    def test_caps_candidates(self) -> None:
        llm = FakeLLM("1")
        ranker = LLMReranker(llm, max_candidates=3)
        picked = ranker.rerank("q", hits(10))
        self.assertEqual(picked[0]["chunk_id"], "c1")
        prompt = llm.calls[0]["messages"][1]["content"]
        self.assertIn("[3]", prompt)
        self.assertNotIn("[4]", prompt)

    def test_empty_hits_or_question(self) -> None:
        llm = FakeLLM("1")
        ranker = LLMReranker(llm)
        self.assertIsNone(ranker.rerank("q", []))
        self.assertIsNone(ranker.rerank("   ", hits()))
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
