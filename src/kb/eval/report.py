"""把评测结果渲染成能直接贴进 README 的文本。

单独放一个模块是为了让渲染逻辑也能被单元测试覆盖（tests/test_eval.py 里
对失败案例清单和对比表格的断言就打在 Formatter 上），并且让 run_eval.py
只管流程、不管排版。
"""

from __future__ import annotations

import unicodedata

from .metrics import DEFAULT_KS, DIFFICULTY_ORDER

DIFFICULTY_CN = {"easy": "简单", "medium": "中等", "hard": "困难", "unknown": "未标注"}
_CATEGORY_ORDER_PREFIX = ("术语", "同义", "中英", "代码", "跨节", "干扰")


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def _num(x: float | None, width: int = 5) -> str:
    return "  n/a" if x is None else f"{x:{width}.3f}"


def char_width(ch: str) -> int:
    """单个字符的显示宽度。East Asian Width 为 W/F 的占 2 列。

    AMBIGUOUS（`Δ`、`↑`、`±` 这类）按 1 列算。它们是"半个字宽"字符，
    不同终端渲染不一致，但表里只有最后一列的箭头可能是 AMBIGUOUS，
    差 1 列也不会影响前面各列的对齐。
    """
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _dw(text: str) -> int:
    return sum(char_width(ch) for ch in text)


def _rpad(text: str, width: int) -> str:
    """按显示宽度补齐，避免中文列在等宽字体下错位。"""
    return text + " " * max(0, width - _dw(text))


def format_config(config: dict) -> str:
    n = config.get("n_questions", "-")
    total = config.get("n_questions_total")
    n_text = str(n)
    if total and n != total:
        n_text += f"（--limit 截断自 {total} 条）"
    lines = [
        "运行配置",
        f"  评测集    : {config.get('questions', '-')}",
        f"  索引目录  : {config.get('index_dir', '-')}",
        f"  检索开关  : dense={config.get('use_dense')} sparse={config.get('use_sparse')} "
        f"rerank={config.get('use_rerank')}",
        f"  题目数量  : {n_text}",
        f"  top_k     : {config.get('top_k', '-')}",
    ]
    if config.get("elapsed_sec") is not None:
        lines.append(f"  耗时      : {config['elapsed_sec']}s")
    return "\n".join(lines)


def format_overall(summary: dict, ks=DEFAULT_KS) -> str:
    mrr_k = summary.get("mrr_k", 10)
    head = "  ".join(f"R@{k:<2}" for k in ks)
    vals = "  ".join(f"{summary['recall'][k] * 100:5.1f}%" for k in ks
                     if k in summary.get("recall", {}))
    lines = [
        f"总体指标（{summary['n']} 条）",
        f"  {head}   MRR@{mrr_k}   Hit@1",
        f"  {vals}   {summary[f'mrr@{mrr_k}']:.3f}    {summary['hit@1'] * 100:5.1f}%",
    ]
    if summary.get("mean_rank"):
        lines.append(f"  命中项平均名次 : {summary['mean_rank']:.2f}")
    lines.append(f"  未召回         : {summary.get('n_failures', 0)} 条")
    return "\n".join(lines)


def format_strata(summary: dict, ks=DEFAULT_KS) -> str:
    lines = ["按难度分层", f"  {_rpad('难度', 8)}{_rpad('题数', 6)}{_rpad('R@1', 8)}"
                           f"{_rpad('R@3', 8)}{_rpad('R@5', 8)}{_rpad('R@10', 8)}"
                           f"{_rpad('MRR@10', 9)}Hit@1"]
    for name, block in summary.get("by_difficulty", {}).items():
        lines.append(
            f"  {_rpad(DIFFICULTY_CN.get(name, name), 8)}{_rpad(str(block['n']), 6)}"
            f"{_rpad(_pct(block['recall'].get(1)), 8)}"
            f"{_rpad(_pct(block['recall'].get(3)), 8)}"
            f"{_rpad(_pct(block['recall'].get(5)), 8)}"
            f"{_rpad(_pct(block['recall'].get(10)), 8)}"
            f"{_rpad(_num(block.get('mrr@10')), 9)}{_pct(block['hit@1'])}"
        )
    return "\n".join(lines)


def format_categories(summary: dict) -> str:
    cats = summary.get("by_category", {})
    if not cats:
        return ""
    lines = ["按题型分层", f"  {_rpad('题型', 16)}{_rpad('题数', 6)}{_rpad('R@5', 8)}MRR@10"]
    for name, block in cats.items():
        lines.append(
            f"  {_rpad(name, 16)}{_rpad(str(block['n']), 6)}"
            f"{_rpad(_pct(block['recall'].get(5)), 8)}{_num(block.get('mrr@10'))}"
        )
    return "\n".join(lines)


def format_failures(summary: dict, limit: int | None = None, show_top: int = 3) -> str:
    failures = summary.get("failures", [])
    if not failures:
        return "失败案例清单：无（全部召回）"
    shown = failures if limit is None else failures[:limit]
    lines = [f"失败案例清单（{len(failures)} 条未召回"
             + (f"，展示前 {len(shown)} 条）" if len(shown) < len(failures) else "）")]
    for i, f in enumerate(shown, start=1):
        lines.append("")
        lines.append(f"  [{i}] {f['qid']}  {f['question']}")
        lines.append(f"      难度/题型 : {DIFFICULTY_CN.get(f['difficulty'], f['difficulty'])} / {f['category']}")
        if f.get("note"):
            lines.append(f"      备注      : {f['note']}")
        if f.get("error"):
            lines.append(f"      错误      : {f['error']}")
            continue
        lines.append("      gold      :")
        for bc in f["gold_breadcrumbs"]:
            lines.append(f"        - {bc}")
        lines.append(f"      实际 top{show_top} :")
        for rank, bc in enumerate(f["top_breadcrumbs"], start=1):
            lines.append(f"        {rank}. {bc}")
    if len(shown) < len(failures):
        lines.append("")
        lines.append(f"  （还有 {len(failures) - len(shown)} 条，用 --json 或 --out 看完整清单）")
    return "\n".join(lines)


def format_report(outcomes, summary: dict, config: dict, *, failure_limit: int | None = None,
                  ks=DEFAULT_KS) -> str:
    blocks = [
        format_config(config),
        format_overall(summary, ks),
        format_strata(summary, ks),
    ]
    cats = format_categories(summary)
    if cats:
        blocks.append(cats)
    if summary.get("errors"):
        blocks.append("检索期错误\n" + "\n".join(
            f"  {e['qid']}: {e['error']}" for e in summary["errors"]))
    blocks.append(format_failures(summary, limit=failure_limit))
    return "\n\n".join(b for b in blocks if b)


def format_diff(diff: dict, label_a: str = "A", label_b: str = "B") -> str:
    """渲染对比表。

    列宽固定，并且**每一行都以 `_rpad(..., LAST_COL)` 结束** —— 包括表头。
    最后一列里是 `↑` / `↓` 这种 East Asian Width 为 AMBIGUOUS 的字符，
    不同终端宽度不一致，只有把整列按同一宽度补齐、不由行尾补空格，
    行宽才可能真正对齐。
    """
    LAST_COL = 5
    col = max(9, _dw(label_a), _dw(label_b)) + 2
    total = 2 + 14 + col * 3 + LAST_COL

    def table(rows: list[dict], title: str) -> str:
        header = (f"  {_rpad('指标', 14)}{_rpad('A', col)}{_rpad('B', col)}"
                  f"{_rpad('Delta', col)}{_rpad('dir', LAST_COL)}")
        lines = [title, header, "  " + "-" * (total - 2)]
        for row in rows:
            va, vb, d = row["a"], row["b"], row["delta"]
            if d is None:
                delta_s, arrow = "n/a", "-"
            elif abs(d) < 1e-12:
                delta_s, arrow = "0", "="
            else:
                delta_s = f"{d:+.4f}"
                arrow = "^" if row["direction"] == "up" else "v"
            a_s = f"{va:.4f}" if isinstance(va, float) else str(va)
            b_s = f"{vb:.4f}" if isinstance(vb, float) else str(vb)
            lines.append(f"  {_rpad(row['metric'], 14)}{_rpad(a_s, col)}"
                         f"{_rpad(b_s, col)}{_rpad(delta_s, col)}{_rpad(arrow, LAST_COL)}")
        return "\n".join(lines)

    blocks = [table(diff["overall"], f"总体指标对比（Delta = {label_b} - {label_a}）")]
    for name in DIFFICULTY_ORDER:
        if name in diff.get("by_difficulty", {}):
            blocks.append(table(diff["by_difficulty"][name],
                                f"难度 {DIFFICULTY_CN.get(name, name)}"
                                f"（Delta = {label_b} - {label_a}）"))

    def qlist(items, title, fmt):
        if not items:
            return f"{title}：无"
        return title + "\n" + "\n".join("  " + fmt(i) for i in items)

    blocks.append(qlist(
        diff["improved"], f"由未召回变为召回 {len(diff['improved'])} 条",
        lambda i: f"{i['qid']} (rank {i['rank_a']} → {i['rank_b']})  {i['question']}"))
    blocks.append(qlist(
        diff["regressed"], f"由召回退化为未召回 {len(diff['regressed'])} 条",
        lambda i: f"{i['qid']} (rank {i['rank_a']} → {i['rank_b']})  {i['question']}"))
    blocks.append(f"两次都没召回 {len(diff['both_failed'])} 条："
                  + (", ".join(i["qid"] for i in diff["both_failed"]) or "无"))
    return "\n\n".join(blocks)
