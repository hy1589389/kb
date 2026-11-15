"""答案侧评测测试：判定解析、指标口径、报告。

跑法：
    python tests/test_answer_eval.py
    python -m unittest tests.test_answer_eval -v

不联网：生成用 `FakeLLM`（固定返回一段 JSON），answerer 是鸭子类型的假对象。
重点覆盖「判错了会静默给出好看数字」的几处：

  · checked 和句子数不一致 —— 必须记成 judge_error，不能算成 100% 支持
  · 拒答的题不送判定     —— 拒答正文是固定话术，判定它等于白花 token
  · 假拒答口径           —— 只在「上下文里确实有 gold」时才算假拒答
  · 没有 gold 标注的题   —— 不进假拒答分母，否则比率会虚高
  · 分类型抽样           —— 空样本一律给 0，不给 NaN
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from kb.eval.answer import (  # noqa: E402
    AnswerOutcome,
    build_judge_messages,
    evaluate_one,
    format_answer_report,
    judge_answer,
    parse_judgement,
    run,
    save_answer_run,
    summarize,
)
from kb.eval.questions import Question  # noqa: E402
from kb.generate.llm import FakeLLM  # noqa: E402
from kb.generate.types import REFUSE_TEXT, Answer, Citation  # noqa: E402

TMP_ROOT = ROOT / ".tmp"


def make_citation(ref: int, chunk_id: str, *, text: str = "HashMap 链表长度到 8 转红黑树。") -> Citation:
    return Citation(
        ref=ref,
        chunk_id=chunk_id,
        repo="JavaGuide",
        rel_path="docs/hashmap.md",
        breadcrumb="JavaGuide > Map > HashMap",
        line_start=12,
        line_end=40,
        text=text,
        score=0.03,
    )


def make_answer(*, text: str = "链表长度到 8 转红黑树 [1]。低于 6 退回链表 [1]。",
                citations: list[Citation] | None = None, refused: bool = False,
                hits: list | None = None, usage: dict | None = None) -> Answer:
    cites = citations if citations is not None else [make_citation(1, "chunk-1")]
    return Answer(
        question="HashMap 什么时候转红黑树",
        text=REFUSE_TEXT if refused else text,
        citations=[] if refused else cites,
        refused=refused,
        refuse_reason="没有召回到任何段落" if refused else "",
        hits=hits if hits is not None else cites,
        model="fake",
        usage=usage or {"prompt_tokens": 1000, "completion_tokens": 50,
                        "total_tokens": 1050},
        latency_ms=1200,
    )


def make_question(qid: str = "q001", gold: list[str] | None = None,
                  category: str = "事实", difficulty: str = "easy") -> Question:
    return Question(qid=qid, question="HashMap 什么时候转红黑树",
                    gold=gold if gold is not None else ["chunk-1"],
                    category=category, difficulty=difficulty)


def make_hit(chunk_id: str, rank: int):
    """检索结果的最小替身：评测层只读 chunk_id 和 rank。"""
    return SimpleNamespace(chunk_id=chunk_id, rank=rank)


class JudgeParsingTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        result = parse_judgement('{"checked": 3, "unsupported": [{"id": 2, "why": "段落没提"}]}', 3)
        self.assertEqual(result["error"], "")
        self.assertEqual(result["supported"], 2)
        self.assertEqual(result["ratio"], 0.6667)
        self.assertEqual(result["unsupported"][0]["id"], 2)

    def test_fenced_json(self) -> None:
        text = '```json\n{"checked": 2, "unsupported": []}\n```'
        self.assertEqual(parse_judgement(text, 2)["ratio"], 1.0)

    def test_prose_around_json(self) -> None:
        text = '判断如下：\n{"checked": 2, "unsupported": [{"id": 1, "why": "无关"}]}\n以上。'
        result = parse_judgement(text, 2)
        self.assertEqual(result["error"], "")
        self.assertEqual(result["ratio"], 0.5)

    def test_checked_mismatch_is_error(self) -> None:
        """模型数错了句数 → 比例不可信，必须报错而不是算成 100%。"""
        result = parse_judgement('{"checked": 2, "unsupported": []}', 5)
        self.assertIn("checked=2", result["error"])
        self.assertNotIn("ratio", result)

    def test_missing_checked_is_error(self) -> None:
        result = parse_judgement('{"unsupported": []}', 3)
        self.assertIn("checked", result["error"])

    def test_no_json_is_error(self) -> None:
        self.assertIn("JSON", parse_judgement("全部都有依据。", 3)["error"])

    def test_illegal_ids_are_dropped(self) -> None:
        text = '{"checked": 3, "unsupported": [{"id": 9}, {"id": 0}, {"id": "x"}, {"id": 2}, {"id": 2}]}'
        result = parse_judgement(text, 3)
        self.assertEqual([item["id"] for item in result["unsupported"]], [2])

    def test_empty_answer_claims(self) -> None:
        result = parse_judgement('{"checked": 0, "unsupported": []}', 0)
        self.assertEqual(result["ratio"], 0.0)


class JudgeCallTests(unittest.TestCase):
    def test_prompt_contains_context_and_numbered_sentences(self) -> None:
        citations = [make_citation(1, "chunk-1", text="红黑树为了把查询降到 O(log n)")]
        messages = build_judge_messages("问题？", citations, ["第一句。", "第二句。"])
        user = messages[-1]["content"]
        self.assertIn("[1] 红黑树为了把查询降到 O(log n)", user)
        self.assertIn("1. 第一句。", user)
        self.assertIn("2. 第二句。", user)
        self.assertIn("必须是 2", user)
        self.assertEqual(messages[0]["role"], "system")

    def test_judge_answer_fills_sentence_text(self) -> None:
        llm = FakeLLM('{"checked": 2, "unsupported": [{"id": 2, "why": "段落没提"}]}')
        answer = make_answer()
        result = judge_answer(llm, "HashMap 什么时候转红黑树", answer)
        self.assertEqual(result["ratio"], 0.5)
        self.assertTrue(result["unsupported"][0]["sentence"].startswith("低于 6"))
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["temperature"], 0.0)

    def test_judge_failure_does_not_raise(self) -> None:
        def boom(messages):
            raise RuntimeError("网络断了")

        result = judge_answer(FakeLLM(boom), "问题", make_answer())
        self.assertIn("网络断了", result["error"])
        self.assertNotIn("ratio", result)


class EvaluateOneTests(unittest.TestCase):
    def test_gold_in_context_when_cited(self) -> None:
        outcome = evaluate_one(make_question(),
                               make_answer(hits=[make_hit("chunk-1", 1)]))
        self.assertTrue(outcome.gold_in_context)
        self.assertEqual(outcome.gold_hit_rank, 1)
        self.assertFalse(outcome.refused)

    def test_gold_not_cited_is_not_in_context(self) -> None:
        """gold 块召回了但答案引的是别的块 → 上下文里没有依据。"""
        answer = make_answer(citations=[make_citation(1, "chunk-9")],
                             hits=[make_hit("chunk-9", 1)])
        outcome = evaluate_one(make_question(), answer)
        self.assertFalse(outcome.gold_in_context)
        self.assertEqual(outcome.gold_hit_rank, 0)

    def test_refusal_skips_judge(self) -> None:
        llm = FakeLLM('{"checked": 1, "unsupported": []}')
        outcome = evaluate_one(make_question(), make_answer(refused=True), judge_llm=llm)
        self.assertTrue(outcome.refused)
        self.assertEqual(outcome.judge, {})
        self.assertEqual(llm.calls, [])   # 拒答不送判定，省 token

    def test_uncited_and_invalid_refs_are_counted(self) -> None:
        """第三句引了不存在的 [7]：算「有标记」，但仍要进 invalid_refs。

        两件事口径不同，别混：`uncited` 只数没有 [n] 的句子，`invalid_refs`
        数编号越界的引用 —— 引了错的号不等于没引。
        """
        answer = make_answer(text="第一句有引用 [1]。第二句没有标记。还引了不存在的 [7]。")
        outcome = evaluate_one(make_question(), answer)
        self.assertEqual(outcome.uncited_sentences, 1)
        self.assertEqual(outcome.invalid_refs, [7])

    def test_gap_declaration_is_not_counted_as_uncited(self) -> None:
        """「资料未覆盖：X」是缺口声明，不是没标引用的论断，两边统计都要排除。"""
        answer = make_answer(
            text="链表长度到 8 转红黑树 [1]。\n资料未覆盖：转回链表的条件。")
        outcome = evaluate_one(make_question(), answer)
        self.assertEqual(outcome.uncited_sentences, 0)
        self.assertEqual(outcome.sentence_count, 1)

    def test_gap_declaration_is_not_sent_to_judge(self) -> None:
        llm = FakeLLM('{"checked": 1, "unsupported": []}')
        answer = make_answer(
            text="链表长度到 8 转红黑树 [1]。\n资料未覆盖：转回链表的条件。")
        outcome = evaluate_one(make_question(), answer, judge_llm=llm)
        self.assertEqual(outcome.judge["total"], 1)
        self.assertEqual(outcome.judge["ratio"], 1.0)


class SummarizeTests(unittest.TestCase):
    def test_rates_use_the_right_denominators(self) -> None:
        outcomes = [
            # gold 召回、被引用、判定 100%
            AnswerOutcome(qid="q1", question="a", gold_in_context=True,
                          gold_hit_rank=1,
                          judge={"ratio": 1.0, "unsupported": []}, sentence_count=2),
            # gold 进了提示词（第 2 名）却拒答 → 假拒答
            AnswerOutcome(qid="q2", question="b", refused=True, gold_hit_rank=2),
            # 根本没召回到 gold、拒答 → 不算假拒答
            AnswerOutcome(qid="q3", question="c", refused=True, gold_hit_rank=0),
        ]
        summary = summarize(outcomes)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["n_refused"], 2)
        self.assertEqual(summary["refusal_rate"], 0.6667)
        self.assertEqual(summary["n_retrieved"], 2)
        self.assertEqual(summary["gold_retrieved_rate"], 0.6667)
        self.assertEqual(summary["n_cited"], 1)
        self.assertEqual(summary["gold_cited_rate"], 0.3333)
        self.assertEqual(summary["n_false_refusal"], 1)
        self.assertEqual(summary["false_refusal_rate"], 0.5)
        self.assertEqual(summary["judged"], 1)
        self.assertEqual(summary["mean_support_ratio"], 1.0)

    def test_false_refusal_counts_retrieved_but_uncited_gold(self) -> None:
        """召回到了却没引用、还拒答 —— 这是模型层拒答过严，不是检索问题。"""
        outcome = AnswerOutcome(qid="q023", question="链表交换", refused=True,
                                gold_hit_rank=1, gold_in_context=False)
        summary = summarize([outcome])
        self.assertEqual(summary["n_false_refusal"], 1)
        self.assertEqual(summary["false_refusal_rate"], 1.0)
        self.assertEqual(summary["gold_cited_rate"], 0.0)

    def test_empty_sample_gives_zeros(self) -> None:
        summary = summarize([])
        self.assertEqual(summary["n"], 0)
        self.assertEqual(summary["refusal_rate"], 0.0)
        self.assertEqual(summary["mean_support_ratio"], 0.0)
        self.assertEqual(summary["by_category"], {})

    def test_judge_errors_are_reported_not_counted(self) -> None:
        outcomes = [
            AnswerOutcome(qid="q1", question="a", gold_in_context=True,
                          judge={"error": "checked 不一致"}),
            AnswerOutcome(qid="q2", question="b", gold_in_context=True,
                          judge={"ratio": 0.5, "unsupported": [{"id": 1, "why": "-"}]}),
        ]
        summary = summarize(outcomes)
        self.assertEqual(summary["judge_error"], 1)
        self.assertEqual(summary["judged"], 1)
        self.assertEqual(summary["mean_support_ratio"], 0.5)
        self.assertEqual(summary["unsupported_sentences"], 1)


class FakeAnswerer:
    """只实现 answer()，足够 `run()` 用。"""

    def __init__(self, answers: dict[str, Answer]):
        self.answers = answers
        self.calls: list[dict] = []

    def answer(self, question: str, top_k: int = 5, collections=None) -> Answer:
        self.calls.append({"question": question, "top_k": top_k,
                           "collections": collections})
        return self.answers[question]


class RunTests(unittest.TestCase):
    def test_run_wires_questions_to_answers(self) -> None:
        questions = [make_question("q001"), make_question("q002")]
        answerer = FakeAnswerer({
            "HashMap 什么时候转红黑树": make_answer(),
        })
        judge = FakeLLM('{"checked": 2, "unsupported": []}')
        outcomes, summary, meta = run(questions, answerer, top_k=3, judge_llm=judge)

        self.assertEqual(len(outcomes), 2)
        self.assertEqual(summary["judged"], 2)
        self.assertEqual(answerer.calls[0]["top_k"], 3)
        self.assertTrue(meta["judge"])
        self.assertEqual(meta["questions"], 2)

    def test_run_records_refusal_without_judging(self) -> None:
        answerer = FakeAnswerer({"HashMap 什么时候转红黑树": make_answer(
            refused=True, hits=[make_hit("chunk-1", 1)])})
        judge = FakeLLM('{"checked": 1, "unsupported": []}')
        outcomes, summary, _ = run([make_question()], answerer, judge_llm=judge)
        self.assertEqual(judge.calls, [])
        self.assertEqual(summary["n_refused"], 1)
        # gold 在提示词里却被拒答 —— 口径按 gold_hit_rank 算，不是按「被引用」
        self.assertEqual(summary["n_false_refusal"], 1)


class ArtifactTests(unittest.TestCase):
    def test_save_keeps_three_top_level_keys(self) -> None:
        outcome = AnswerOutcome(qid="q001", question="a", gold_in_context=True,
                                judge={"ratio": 0.5, "unsupported": []})
        summary = summarize([outcome])
        with tempfile.TemporaryDirectory(dir=TMP_ROOT if TMP_ROOT.exists() else None) as tmp:
            path = save_answer_run(Path(tmp) / "run_answer.json", [outcome], summary,
                                   {"top_k": 5, "judge": True})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(payload), {"config", "summary", "per_query"})
        self.assertEqual(payload["per_query"][0]["qid"], "q001")

    def test_report_shows_rates_and_bad_sentences(self) -> None:
        outcome = AnswerOutcome(
            qid="q001", question="a", gold_in_context=True,
            judge={"ratio": 0.5, "total": 2,
                   "unsupported": [{"id": 2, "why": "段落没提",
                                    "sentence": "第二句是编的。"}]},
        )
        summary = summarize([outcome])
        report = format_answer_report([outcome], summary,
                                      {"top_k": 5, "judge": True, "elapsed_sec": 1.0})
        self.assertIn("假拒答", report)
        self.assertIn("平均支持率", report)
        self.assertIn("50.0%", report)
        self.assertIn("第二句是编的", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
