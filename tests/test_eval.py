"""评测逻辑回归测试（仅标准库，无需 pytest，也不需要真实索引）。

运行：
    python tests/test_eval.py
    python -m unittest discover -s tests -v

这里不碰任何真的向量库：Retriever 用鸭子类型的假对象（FakeRetriever）替代。
要守住的几件事：
  · Recall@k / MRR / Hit@1 的算法本身算对（用能手工核对的小例子）
  · 一条问题有多个 gold 时，命中任意一个就算命中
  · 一条都没命中时 RR 必须是 0，不能是 None 也不能崩
  · 分层指标归组正确，题数加起来等于总数
  · 失败案例清单要给出 gold 的 breadcrumb 和实际 top-3，这是调优的输入
  · 两次运行的差值和方向要对（含 mean_rank 越小越好这种反向指标）
  · 评测集格式错误必须报错，不能静默跳过
"""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows 控制台默认 cp936，中文断言消息会炸，这里强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from kb.eval import (  # noqa: E402
    EvalDataError,
    QueryOutcome,
    Question,
    build_outcomes,
    diff_runs,
    first_hit_rank,
    format_diff,
    format_report,
    gold_coverage_at_k,
    hit_at_k,
    load_questions,
    mean_reciprocal_rank,
    parse_question,
    recall_at_k,
    reciprocal_rank,
    summarize,
    to_serializable,
)


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------

class FakeHit:
    """冒充 kb.retrieve.SearchHit，只带评测用得到的字段。"""

    def __init__(self, chunk_id, breadcrumb, score=1.0, repo="JavaGuide", rank=0):
        self.chunk_id = chunk_id
        self.breadcrumb = breadcrumb
        self.score = score
        self.repo = repo
        self.rank = rank
        self.doc_id = "doc-" + chunk_id
        self.rel_path = "docs/x.md"
        self.line_start = 1
        self.line_end = 2
        self.text = "text"
        self.source = "dense"


class FakeRetriever:
    """鸭子类型检索器：按预先写死的剧本返回结果，不依赖索引。"""

    def __init__(self, script):
        self.script = script            # query -> list[chunk_id]
        self.calls = []                 # 记录调用参数，用来断言开关被正确透传

    def search(self, query, top_k=10, *, use_dense=True, use_sparse=True, use_rerank=False):
        self.calls.append({"query": query, "top_k": top_k, "use_dense": use_dense,
                           "use_sparse": use_sparse, "use_rerank": use_rerank})
        ids = self.script.get(query, [])
        return [FakeHit(cid, f"BC:{cid}", rank=i) for i, cid in enumerate(ids[:top_k], start=1)]


def make_question(qid, question, gold, difficulty="easy", category="术语精确匹配", note=""):
    return Question(qid=qid, question=question, gold=list(gold),
                    difficulty=difficulty, category=category, note=note)


# 临时目录放在项目内的 tests/.tmp 下，不写系统 temp（沙箱里系统 temp 可能不可写）
TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


def make_tmpdir(name: str) -> Path:
    d = TMP_ROOT / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# 1. Recall@k
# --------------------------------------------------------------------------

class TestRecallAtK(unittest.TestCase):
    def test_first_hit_rank_basic(self):
        self.assertEqual(first_hit_rank(["b"], ["a", "b", "c"]), 2)
        self.assertEqual(first_hit_rank(["a"], ["a", "b"]), 1)
        self.assertIsNone(first_hit_rank(["z"], ["a", "b", "c"]))

    def test_recall_is_binary_per_question(self):
        """手工核对：predicted = [a, b, c]，gold = c，只有 k>=3 才命中。"""
        gold, predicted = ["c"], ["a", "b", "c"]
        self.assertEqual(recall_at_k(gold, predicted, 1), 0.0)
        self.assertEqual(recall_at_k(gold, predicted, 2), 0.0)
        self.assertEqual(recall_at_k(gold, predicted, 3), 1.0)
        self.assertEqual(recall_at_k(gold, predicted, 10), 1.0)
        self.assertFalse(hit_at_k(gold, predicted, 2))
        self.assertTrue(hit_at_k(gold, predicted, 3))

    def test_recall_at_k_matches_hand_computed_average(self):
        """两条题、手工算出来的 Recall@1 = 1/2，Recall@3 = 2/2。"""
        outcomes = [
            QueryOutcome("q1", "问题一", ["a"], ["a", "x", "y"]),
            QueryOutcome("q2", "问题二", ["b"], ["x", "y", "b"]),
        ]
        summary = summarize(outcomes, ks=(1, 3, 5))
        self.assertEqual(summary["recall"][1], 0.5)
        self.assertEqual(summary["recall"][3], 1.0)
        self.assertEqual(summary["recall"][5], 1.0)
        self.assertEqual(summary["hit@1"], 0.5)
        self.assertEqual(summary["n"], 2)

    def test_multiple_gold_chunks_any_hit_counts(self):
        """gold 有两个 chunk，top-1 命中的是第二个，也必须算命中。"""
        gold = ["chunk_a", "chunk_b"]
        predicted = ["chunk_b", "other"]
        self.assertEqual(recall_at_k(gold, predicted, 1), 1.0)
        self.assertEqual(first_hit_rank(gold, predicted), 1)
        self.assertEqual(reciprocal_rank(gold, predicted, 10), 1.0)

    def test_gold_coverage_distinguishes_partial_hits(self):
        """Recall@k 是二值的，gold_coverage 才能看出"只召回了一半"。"""
        gold = ["a", "b"]
        predicted = ["a", "x", "y"]
        self.assertEqual(recall_at_k(gold, predicted, 3), 1.0)
        self.assertAlmostEqual(gold_coverage_at_k(gold, predicted, 3), 0.5)

    def test_empty_predicted_never_hits(self):
        self.assertEqual(recall_at_k(["a"], [], 10), 0.0)
        self.assertFalse(hit_at_k(["a"], [], 1))


# --------------------------------------------------------------------------
# 2. MRR
# --------------------------------------------------------------------------

class TestMRR(unittest.TestCase):
    def test_reciprocal_rank_values(self):
        self.assertEqual(reciprocal_rank(["a"], ["a", "b"], 10), 1.0)
        self.assertEqual(reciprocal_rank(["b"], ["a", "b"], 10), 0.5)
        self.assertAlmostEqual(reciprocal_rank(["c"], ["a", "b", "c"], 10), 1 / 3)

    def test_no_hit_returns_zero(self):
        """一条都没命中必须是 0.0，不是 None 也不是异常。"""
        self.assertEqual(reciprocal_rank(["z"], ["a", "b", "c"], 10), 0.0)
        self.assertEqual(reciprocal_rank(["z"], [], 10), 0.0)

    def test_mrr_cutoff_at_k(self):
        """命中排在第 5，但 k=3 时截断，RR@3 记 0。"""
        self.assertEqual(reciprocal_rank(["e"], ["a", "b", "c", "d", "e"], 3), 0.0)
        self.assertEqual(reciprocal_rank(["e"], ["a", "b", "c", "d", "e"], 5), 0.2)

    def test_mean_reciprocal_rank_hand_computed(self):
        """(1 + 0.5 + 0) / 3 = 0.5"""
        pairs = [
            (["a"], ["a", "b"]),
            (["b"], ["a", "b"]),
            (["z"], ["a", "b"]),
        ]
        self.assertAlmostEqual(mean_reciprocal_rank(pairs, 10), 0.5)
        self.assertEqual(mean_reciprocal_rank([], 10), 0.0)

    def test_summary_mrr_and_hit1(self):
        outcomes = [
            QueryOutcome("q1", "一", ["a"], ["a"]),          # rank1 → rr 1.0
            QueryOutcome("q2", "二", ["b"], ["x", "b"]),     # rank2 → rr 0.5
            QueryOutcome("q3", "三", ["c"], ["x", "y"]),     # 未命中 → rr 0
        ]
        summary = summarize(outcomes, ks=(1, 3, 5, 10))
        self.assertAlmostEqual(summary["mrr@10"], 0.5)
        self.assertAlmostEqual(summary["hit@1"], 1 / 3)
        self.assertEqual(summary["n_failures"], 1)
        self.assertEqual(summary["mean_rank"], 1.5)          # (1 + 2) / 2


# --------------------------------------------------------------------------
# 3. 分层指标
# --------------------------------------------------------------------------

class TestStratifiedMetrics(unittest.TestCase):
    def _outcomes(self):
        # easy: 2 条，全命中 → R@5 = 1.0
        # medium: 2 条，1 条命中 → R@5 = 0.5
        # hard: 2 条，全未命中 → R@5 = 0.0
        return [
            QueryOutcome("e1", "题", ["a"], ["a"], difficulty="easy", category="术语精确匹配"),
            QueryOutcome("e2", "题", ["b"], ["b"], difficulty="easy", category="术语精确匹配"),
            QueryOutcome("m1", "题", ["c"], ["x", "c"], difficulty="medium", category="同义改写"),
            QueryOutcome("m2", "题", ["d"], ["x", "y"], difficulty="medium", category="同义改写"),
            QueryOutcome("h1", "题", ["e"], ["x"], difficulty="hard", category="跨节综合"),
            QueryOutcome("h2", "题", ["f"], ["y"], difficulty="hard", category="跨节综合"),
        ]

    def test_grouping_counts_and_recall(self):
        summary = summarize(self._outcomes(), ks=(1, 5, 10))
        by_d = summary["by_difficulty"]

        self.assertEqual(set(by_d), {"easy", "medium", "hard"})
        self.assertEqual(by_d["easy"]["n"], 2)
        self.assertEqual(by_d["medium"]["n"], 2)
        self.assertEqual(by_d["hard"]["n"], 2)
        self.assertEqual(sum(by_d[k]["n"] for k in by_d), summary["n"])

        self.assertEqual(by_d["easy"]["recall"][5], 1.0)
        self.assertEqual(by_d["easy"]["hit@1"], 1.0)
        self.assertEqual(by_d["medium"]["recall"][5], 0.5)
        self.assertEqual(by_d["medium"]["hit@1"], 0.0)
        self.assertEqual(by_d["hard"]["recall"][5], 0.0)
        self.assertEqual(by_d["hard"]["mrr@10"], 0.0)

    def test_grouping_by_category(self):
        summary = summarize(self._outcomes(), ks=(5,))
        by_c = summary["by_category"]
        self.assertEqual(by_c["术语精确匹配"]["n"], 2)
        self.assertEqual(by_c["术语精确匹配"]["recall"][5], 1.0)
        self.assertEqual(by_c["同义改写"]["recall"][5], 0.5)
        self.assertEqual(by_c["跨节综合"]["recall"][5], 0.0)

    def test_difficulty_order_is_stable(self):
        summary = summarize(self._outcomes(), ks=(5,))
        self.assertEqual(list(summary["by_difficulty"]), ["easy", "medium", "hard"])

    def test_errors_do_not_silently_count_as_hits(self):
        """检索报错的题目要单独列出来，且不能算命中。"""
        outcomes = [
            QueryOutcome("q1", "一", ["a"], [], error="检索超时"),
            QueryOutcome("q2", "二", ["b"], ["b"]),
        ]
        summary = summarize(outcomes, ks=(1,))
        self.assertEqual(summary["recall"][1], 0.5)
        self.assertEqual(len(summary["errors"]), 1)
        self.assertEqual(summary["errors"][0]["qid"], "q1")


# --------------------------------------------------------------------------
# 4. 失败案例清单
# --------------------------------------------------------------------------

class TestFailureList(unittest.TestCase):
    def _summary(self):
        outcomes = [
            QueryOutcome("q1", "为什么扩容要重新计算哈希",
                         ["gold-1"], ["gold-1", "hit-2"],
                         difficulty="medium", category="同义改写",
                         note="[medium] 同义改写：正文用 rehash",
                         breadcrumbs={"gold-1": "负载因子和扩容", "hit-2": "扩容 rehash"}),
            QueryOutcome("q2", "G1 和 CMS 的区别",
                         ["gold-2a", "gold-2b"], ["hit-2", "hit-3", "hit-4"],
                         difficulty="hard", category="干扰项",
                         breadcrumbs={"gold-2a": "垃圾收集器 > G1 收集器",
                                      "gold-2b": "垃圾收集器 > CMS 收集器",
                                      "hit-2": "内存分配和回收原则",
                                      "hit-3": "垃圾收集器 > ParNew 收集器",
                                      "hit-4": "Full GC 的触发条件"}),
        ]
        return summarize(outcomes, ks=(1, 3, 5, 10))

    def test_failure_contains_question_gold_and_top3(self):
        summary = self._summary()
        self.assertEqual(summary["n_failures"], 1)
        f = summary["failures"][0]
        self.assertEqual(f["qid"], "q2")
        self.assertEqual(f["question"], "G1 和 CMS 的区别")
        self.assertEqual(f["gold"], ["gold-2a", "gold-2b"])
        self.assertEqual(f["gold_breadcrumbs"],
                         ["垃圾收集器 > G1 收集器  [gold-2a]",
                          "垃圾收集器 > CMS 收集器  [gold-2b]"])
        self.assertEqual(f["top_breadcrumbs"],
                         ["内存分配和回收原则  [hit-2]",
                          "垃圾收集器 > ParNew 收集器  [hit-3]",
                          "Full GC 的触发条件  [hit-4]"])
        self.assertEqual(len(f["top_breadcrumbs"]), 3)
        self.assertEqual(f["top_chunk_ids"], ["hit-2", "hit-3", "hit-4"])
        self.assertEqual(f["difficulty"], "hard")

    def test_hit_question_not_in_failure_list(self):
        summary = self._summary()
        self.assertEqual(summary["n_failures"], 1)
        self.assertNotIn("q1", [f["qid"] for f in summary["failures"]])
        self.assertEqual(summary["recall"][5], 0.5)  # q1 命中、q2 未命中

    def test_report_renders_failure_block(self):
        summary = self._summary()
        report = format_report([], summary, {"questions": "x", "n_questions": 2})
        self.assertIn("失败案例清单（1 条未召回）", report)
        self.assertIn("垃圾收集器 > G1 收集器", report)
        self.assertIn("按难度分层", report)
        self.assertIn("总体指标", report)

    def test_missing_breadcrumb_falls_back_to_chunk_id(self):
        outcomes = [QueryOutcome("q1", "题", ["gold-x"], ["hit-x"])]
        summary = summarize(outcomes)
        self.assertEqual(summary["failures"][0]["gold_breadcrumbs"], ["[gold-x]"])
        self.assertEqual(summary["failures"][0]["top_breadcrumbs"], ["[hit-x]"])

    def test_same_breadcrumb_different_chunks_stay_distinguishable(self):
        """同一小节的多个 chunk breadcrumb 一样，必须靠 chunk_id 区分。"""
        outcomes = [
            QueryOutcome("q1", "题", ["g1"], ["h1"],
                         breadcrumbs={"g1": "put 方法", "h1": "put 方法"}),
        ]
        f = summarize(outcomes)["failures"][0]
        self.assertNotEqual(f["gold_breadcrumbs"][0], f["top_breadcrumbs"][0])
        self.assertIn("g1", f["gold_breadcrumbs"][0])
        self.assertIn("h1", f["top_breadcrumbs"][0])

    def test_all_hit_reports_empty_list(self):
        outcomes = [QueryOutcome("q1", "题", ["a"], ["a"])]
        summary = summarize(outcomes)
        self.assertEqual(summary["failures"], [])
        self.assertIn("无（全部召回）", format_report([], summary, {}))


# --------------------------------------------------------------------------
# 5. 两次运行对比
# --------------------------------------------------------------------------

class TestRunComparison(unittest.TestCase):
    def _runs(self):
        def mk(rr_pairs):
            outcomes = [
                QueryOutcome(qid, f"问题{qid}", ["g-" + qid], pred)
                for qid, pred in rr_pairs
            ]
            summary = summarize(outcomes, ks=(1, 3, 5, 10))
            return to_serializable(outcomes, summary, {"use_dense": True, "use_sparse": True})

        # A：q1 命中 rank1，q2 未命中            → R@1 = 0.5, MRR = 0.5
        run_a = mk([("q1", ["g-q1"]), ("q2", ["x", "y"])])
        # B：q1 降到 rank2，q2 变成命中 rank1    → R@1 = 0.5, MRR = 0.75
        run_b = mk([("q1", ["x", "g-q1"]), ("q2", ["g-q2", "y"])])
        return run_a, run_b

    def test_delta_and_direction(self):
        run_a, run_b = self._runs()
        diff = diff_runs(run_a, run_b)
        rows = {r["metric"]: r for r in diff["overall"]}

        self.assertAlmostEqual(rows["recall@1"]["a"], 0.5)
        self.assertAlmostEqual(rows["recall@1"]["b"], 0.5)
        self.assertAlmostEqual(rows["recall@1"]["delta"], 0.0)
        self.assertEqual(rows["recall@1"]["direction"], "same")

        self.assertAlmostEqual(rows["mrr@10"]["a"], 0.5)
        self.assertAlmostEqual(rows["mrr@10"]["b"], 0.75)
        self.assertAlmostEqual(rows["mrr@10"]["delta"], 0.25)
        self.assertEqual(rows["mrr@10"]["direction"], "up")

    def test_mean_rank_is_lower_is_better(self):
        """A: q1 排 1、q2 未命中 → mean_rank 1.0；B: q1 掉到 2、q2 排 1 → 1.5。"""
        run_a, run_b = self._runs()
        rows = {r["metric"]: r for r in diff_runs(run_a, run_b)["overall"]}
        self.assertAlmostEqual(rows["mean_rank"]["a"], 1.0)
        self.assertAlmostEqual(rows["mean_rank"]["b"], 1.5)
        self.assertAlmostEqual(rows["mean_rank"]["delta"], 0.5)
        self.assertEqual(rows["mean_rank"]["direction"], "down")   # 名次变大 = 变差

        # 反过来：B 变 A 是名次从 1.5 降到 1.0，direction 必须是 up
        back = {r["metric"]: r for r in diff_runs(run_b, run_a)["overall"]}
        self.assertAlmostEqual(back["mean_rank"]["delta"], -0.5)
        self.assertEqual(back["mean_rank"]["direction"], "up")

    def test_improved_and_regressed_lists(self):
        run_a, run_b = self._runs()
        diff = diff_runs(run_a, run_b)
        self.assertEqual([i["qid"] for i in diff["improved"]], ["q2"])
        self.assertEqual([i["qid"] for i in diff["regressed"]], [])
        self.assertEqual(diff["improved"][0]["rank_a"], None)
        self.assertEqual(diff["improved"][0]["rank_b"], 1)

    def test_regression_direction(self):
        """反向：B 把原本命中的题弄丢了，必须报 regressed。"""
        run_a, run_b = self._runs()
        diff = diff_runs(run_b, run_a)
        self.assertEqual([i["qid"] for i in diff["regressed"]], ["q2"])
        rows = {r["metric"]: r for r in diff["overall"]}
        self.assertAlmostEqual(rows["mrr@10"]["delta"], -0.25)
        self.assertEqual(rows["mrr@10"]["direction"], "down")

    def test_both_failed_list(self):
        run_a, run_b = self._runs()
        diff = diff_runs(run_a, run_b)
        self.assertEqual(diff["both_failed"], [])
        empty = {"summary": {"recall": {1: 0.0}, "mrr@10": 0.0, "hit@1": 0.0, "n": 1},
                 "per_query": [{"qid": "q1", "question": "题", "rank": None}]}
        diff2 = diff_runs(empty, empty)
        self.assertEqual([i["qid"] for i in diff2["both_failed"]], ["q1"])

    def test_by_difficulty_diff_grouped(self):
        def mk(diff_name, pred):
            o = [QueryOutcome("q1", "题", ["g"], pred, difficulty=diff_name)]
            return to_serializable(o, summarize(o, ks=(1,)), {})

        diff = diff_runs(mk("hard", ["x"]), mk("hard", ["g"]))
        self.assertIn("hard", diff["by_difficulty"])
        rows = {r["metric"]: r for r in diff["by_difficulty"]["hard"]}
        self.assertEqual(rows["recall@1"]["direction"], "up")

    def test_format_diff_is_renderable(self):
        run_a, run_b = self._runs()
        text = format_diff(diff_runs(run_a, run_b))
        self.assertIn("总体指标对比", text)
        self.assertIn("由未召回变为召回 1 条", text)
        self.assertIn("mrr@10", text)
        # 变好的一行方向标记是 ^，变差是 v（纯 ASCII，终端宽度稳定）
        self.assertIn("^", text)
        self.assertNotIn("↑", text)

    def test_diff_table_columns_are_aligned(self):
        """表格每一行的显示宽度必须一致，否则贴进 README 会歪。"""
        from kb.eval.report import _dw

        run_a, run_b = self._runs()
        text = format_diff(diff_runs(run_a, run_b), "A", "B")
        table_lines = [ln for ln in text.splitlines()
                       if ln.startswith("  ") and ("recall@" in ln or "mrr@" in ln
                                                   or ln.lstrip().startswith("指标")
                                                   or ln.lstrip().startswith("---"))]
        self.assertGreater(len(table_lines), 3)
        widths = {_dw(ln) for ln in table_lines}
        self.assertEqual(len(widths), 1, f"列宽不一致：{sorted(widths)}；{table_lines}")

    def test_long_labels_do_not_break_alignment(self):
        from kb.eval.report import _dw

        run_a, run_b = self._runs()
        text = format_diff(diff_runs(run_a, run_b), "baseline-with-long-name", "candidate")
        lines = [ln for ln in text.splitlines() if ln.startswith("  ") and "recall@1 " in ln]
        self.assertEqual({_dw(ln) for ln in lines}, {_dw(lines[0])})

    def test_char_width_handles_cjk(self):
        from kb.eval.report import _dw

        self.assertEqual(_dw("abc"), 3)
        self.assertEqual(_dw("中文"), 4)
        self.assertEqual(_dw("中文abc"), 7)


# --------------------------------------------------------------------------
# 6. 评测集格式校验
# --------------------------------------------------------------------------

class TestQuestionValidation(unittest.TestCase):
    def _write(self, lines) -> Path:
        tmp = make_tmpdir("questions") / "q.jsonl"
        tmp.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines),
                       encoding="utf-8")
        return tmp

    def test_valid_file_loads(self):
        p = self._write([
            {"qid": "q001", "question": "树化阈值为什么是 8",
             "gold": ["c1", "c2"], "note": "[easy] 术语精确匹配：正文原样出现"},
            {"qid": "q002", "question": "rehash 成本",
             "gold": ["c3"], "note": "[hard] 同义改写：类别：同义改写"},
        ])
        qs = load_questions(p)
        self.assertEqual(len(qs), 2)
        self.assertEqual(qs[0].qid, "q001")
        self.assertEqual(qs[0].gold, ["c1", "c2"])
        self.assertEqual(qs[0].difficulty, "easy")
        # note 里没写"类别："时归到未分类，不靠猜
        self.assertEqual(qs[0].category, "未分类")
        self.assertEqual(qs[1].difficulty, "hard")
        self.assertEqual(qs[1].category, "同义改写")

    def test_explicit_category_field_wins(self):
        q = parse_question({"qid": "q1", "question": "题", "gold": ["c1"],
                            "category": "代码类", "difficulty": "hard",
                            "note": "[easy] 类别：术语精确匹配"})
        self.assertEqual(q.category, "代码类")
        self.assertEqual(q.difficulty, "hard")

    def test_missing_qid_raises_with_line_number(self):
        p = self._write([{"question": "题", "gold": ["c1"]}])
        with self.assertRaises(EvalDataError) as ctx:
            load_questions(p)
        self.assertIn("第 1 行", str(ctx.exception))
        self.assertIn("qid", str(ctx.exception))

    def test_missing_question_raises(self):
        p = self._write([{"qid": "q1", "gold": ["c1"]}])
        with self.assertRaises(EvalDataError) as ctx:
            load_questions(p)
        self.assertIn("question", str(ctx.exception))

    def test_missing_gold_field_raises(self):
        p = self._write([{"qid": "q1", "question": "题"}])
        with self.assertRaises(EvalDataError) as ctx:
            load_questions(p)
        self.assertIn("gold", str(ctx.exception))

    def test_empty_gold_raises_instead_of_silent_skip(self):
        p = self._write([{"qid": "q1", "question": "题", "gold": []}])
        with self.assertRaises(EvalDataError) as ctx:
            load_questions(p)
        msg = str(ctx.exception)
        self.assertIn("gold", msg)
        self.assertIn("为空", msg)

    def test_gold_must_be_list_of_strings(self):
        with self.assertRaises(EvalDataError):
            parse_question({"qid": "q1", "question": "题", "gold": "c1"}, line_no=3)
        with self.assertRaises(EvalDataError):
            parse_question({"qid": "q1", "question": "题", "gold": [""]}, line_no=3)
        with self.assertRaises(EvalDataError):
            parse_question({"qid": "q1", "question": "题", "gold": ["a", "a"]}, line_no=3)

    def test_duplicate_qid_raises(self):
        p = self._write([
            {"qid": "q1", "question": "题一", "gold": ["c1"]},
            {"qid": "q1", "question": "题二", "gold": ["c2"]},
        ])
        with self.assertRaises(EvalDataError) as ctx:
            load_questions(p)
        self.assertIn("重复", str(ctx.exception))

    def test_bad_json_reports_line(self):
        p = make_tmpdir("bad_json") / "bad.jsonl"
        p.write_text('{"qid": "q1", "question": "题", "gold": ["c1"]}\n{not json}\n',
                     encoding="utf-8")
        with self.assertRaises(EvalDataError) as ctx:
            load_questions(p)
        self.assertIn("第 2 行", str(ctx.exception))
        self.assertIn("JSON", str(ctx.exception))

    def test_bad_difficulty_raises(self):
        with self.assertRaises(EvalDataError) as ctx:
            parse_question({"qid": "q1", "question": "题", "gold": ["c1"],
                            "difficulty": "very-hard"}, line_no=7)
        self.assertIn("difficulty", str(ctx.exception))

    def test_missing_file_raises_clear_error(self):
        with self.assertRaises(EvalDataError) as ctx:
            load_questions(Path("no/such/questions.jsonl"))
        self.assertIn("不存在", str(ctx.exception))

    def test_empty_file_raises(self):
        p = make_tmpdir("empty") / "empty.jsonl"
        p.write_text("\n\n# 只有注释\n", encoding="utf-8")
        with self.assertRaises(EvalDataError) as ctx:
            load_questions(p)
        self.assertIn("没有读到任何题目", str(ctx.exception))

    def test_note_only_difficulty_tag_defaults(self):
        q = parse_question({"qid": "q1", "question": "题", "gold": ["c1"],
                            "note": "随手写的备注"})
        self.assertEqual(q.difficulty, "unknown")
        self.assertEqual(q.category, "未分类")


# --------------------------------------------------------------------------
# 7. 端到端：假 Retriever 跑一遍（不需要真实索引）
# --------------------------------------------------------------------------

class TestEndToEndWithFakeRetriever(unittest.TestCase):
    def _questions(self):
        return [
            make_question("q1", "树化阈值为什么是 8", ["g1"], "easy"),
            make_question("q2", "为什么扩容要重新计算哈希", ["g2"], "medium"),
            make_question("q3", "G1 和 CMS 的区别", ["g3"], "hard"),
        ]

    def test_run_pipeline_and_switch_passthrough(self):
        from kb.eval import run

        script = {
            "树化阈值为什么是 8": ["g1", "x", "y"],
            "为什么扩容要重新计算哈希": ["x", "g2"],
            "G1 和 CMS 的区别": ["x", "y", "z"],
        }
        fake = FakeRetriever(script)
        outcomes, summary, config = run(
            self._questions(), fake, top_k=3, use_dense=True, use_sparse=False,
            use_rerank=True, config={"questions": "fake"},
        )
        self.assertEqual(len(outcomes), 3)
        self.assertEqual(summary["recall"][1], 1 / 3)     # 只有 q1 排第 1
        self.assertEqual(summary["recall"][3], 2 / 3)     # q1、q2 在前 3
        self.assertEqual(summary["n_failures"], 1)
        self.assertEqual(summary["failures"][0]["qid"], "q3")
        self.assertEqual(config["use_sparse"], False)
        self.assertTrue(config["use_rerank"])
        # 开关必须原样透传给检索层
        self.assertTrue(all(c["use_dense"] is True for c in fake.calls))
        self.assertTrue(all(c["use_sparse"] is False for c in fake.calls))
        self.assertTrue(all(c["top_k"] == 3 for c in fake.calls))

    def test_build_outcomes_rejects_non_hit_objects(self):
        """检索层返回了没有 chunk_id 的东西要报错，不能当成空字符串混过去。"""
        qs = self._questions()[:1]
        with self.assertRaises(TypeError):
            build_outcomes(qs, {"q1": [object()]})

    def test_build_outcomes_accepts_dicts(self):
        qs = self._questions()[:1]
        outcomes = build_outcomes(qs, {"q1": [{"chunk_id": "g1", "breadcrumb": "BC"}]})
        self.assertTrue(outcomes[0].hit)
        self.assertEqual(outcomes[0].gold_breadcrumbs(), ["BC  [g1]"])

    def test_missing_qid_is_recorded_as_error(self):
        qs = self._questions()[:1]
        outcomes = build_outcomes(qs, {})
        self.assertFalse(outcomes[0].hit)
        self.assertIn("没有返回结果", outcomes[0].error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
