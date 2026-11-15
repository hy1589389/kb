"""答案侧评测：不只看「检索到没检索到」，而是看「答得有没有依据」。

检索评测（`runner.py`）回答的是「gold 块进没进前 k」。这一层回答另外三件事：

    1. 该答的答了吗   —— 拒答率、假拒答率（gold 块就在上下文里却拒答）
    2. 答的有依据吗   —— 逐句让模型判断「这句能否由给定段落支持」，得到支持率
                         和没有依据的句子清单
    3. 引用站得住吗   —— 无效引用编号、没标引用的句子占比（纯计算，不再调模型）

为什么要单独一层：检索全对但答案编了一段，Recall@5 照样是 85%。RAG 项目被追问
穿的地方基本都在这一层。

成本：每题一次生成 + 一次判定（`--no-judge` 可只跑第 1、3 类）。

判定结果的可信度靠自己报告的事实：提示词要求模型回 `checked`（它数了几句话），
和实际句数对不上就记成 `judge_error` 而不是猜一个比例 —— 判定失败比判定错更常见，
静默算成 1.0 是最坏的结果。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..generate.cite import count_uncited, invalid_refs, split_sentences
from .questions import Question

__all__ = [
    "AnswerOutcome",
    "build_judge_messages",
    "evaluate_one",
    "format_answer_report",
    "judge_answer",
    "parse_judgement",
    "run",
    "save_answer_run",
    "summarize",
]

# 支持率看的是「每句有没有依据」，句数太多时提示词会膨胀到几千 token。
# 30 句已经覆盖正常答案（实测一次问答 3~8 句）。
MAX_JUDGE_SENTENCES = 30

JUDGE_SYSTEM = (
    "你是检索增强问答的评审。你的唯一任务：判断答案里的每一句话，是否能由给定的"
    "参考段落推出。不要判断答案对不对、好不好，只看「有没有依据」。"
)

JUDGE_TEMPLATE = """问题：{question}

参考段落（答案里的 [n] 对应这些编号）：
{context}

答案（按顺序编号，每句一行）：
{sentences}

请逐句判断「这句能否由参考段落支持」。判据：
· 能直接从某段读出，或由某段直接推出 —— 算支持
· 参考段落里没有相关信息、或与段落矛盾 —— 算不支持
· 只是复述问题、寒暄、免责声明 —— 算支持（不涉及事实主张）

只输出一个 JSON 对象，不要任何解释文字：
{{"checked": <你判断了多少句，必须是 {total}>,
  "unsupported": [{{"id": <句子编号>, "why": "<不超过 30 字的原因>"}}]}}

没有被列进 unsupported 的一律视为支持。"""


@dataclass
class AnswerOutcome:
    """一道题的答案侧结果。字段都是「事后能复盘」用的，不做二次加工。"""

    qid: str
    question: str
    category: str = "未分类"
    difficulty: str = "unknown"
    refused: bool = False
    refuse_reason: str = ""
    # 这次答案是「拒答复核」之后重答的（第一次只回了 INSUFFICIENT）
    refusal_retried: bool = False
    answer_chars: int = 0
    n_citations: int = 0
    # 上下文里有没有 gold：区分「检索没召回」和「召回了但生成没用/拒答」
    gold_in_context: bool = False
    gold_hit_rank: int = 0
    invalid_refs: list[int] = field(default_factory=list)
    uncited_sentences: int = 0
    sentence_count: int = 0
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0
    # 判定结果：{"supported": n, "total": n, "ratio": 0..1, "unsupported": [...], "error": ""}
    judge: dict = field(default_factory=dict)

    # ---- 派生量（报告里用，不进原始记录） ----
    def uncited_ratio(self) -> float:
        if not self.sentence_count:
            return 0.0
        return self.uncited_sentences / self.sentence_count

    def support_ratio(self) -> float | None:
        value = self.judge.get("ratio")
        return None if value is None else float(value)

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "question": self.question,
            "category": self.category,
            "difficulty": self.difficulty,
            "refused": self.refused,
            "refuse_reason": self.refuse_reason,
            "refusal_retried": self.refusal_retried,
            "answer_chars": self.answer_chars,
            "n_citations": self.n_citations,
            "gold_in_context": self.gold_in_context,
            "gold_hit_rank": self.gold_hit_rank,
            "invalid_refs": list(self.invalid_refs),
            "uncited_sentences": self.uncited_sentences,
            "sentence_count": self.sentence_count,
            "usage": dict(self.usage or {}),
            "latency_ms": int(self.latency_ms),
            "judge": dict(self.judge or {}),
        }


def build_judge_messages(question: str, citations: list[Any],
                         sentences: list[str]) -> list[dict]:
    """拼判定提示词。`citations` 是答案实际用到的段落（`Answer.citations`）。"""
    blocks: list[str] = []
    for item in citations:
        ref = int(getattr(item, "ref", 0) or 0)
        body = str(getattr(item, "text", "") or "").strip()
        if not body:
            continue
        blocks.append(f"[{ref}] {body}")
    context = "\n\n".join(blocks) or "（无）"
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(sentences, start=1))
    user = JUDGE_TEMPLATE.format(
        question=question, context=context, sentences=numbered, total=len(sentences)
    )
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user}]


def _extract_json_object(text: str) -> dict | None:
    """从可能带代码围栏 / 前后废话的输出里抠出第一个 JSON 对象。"""
    body = (text or "").strip()
    if not body:
        return None
    if body.startswith("```"):
        # ```json\n{...}\n```
        body = body.split("\n", 1)[-1] if "\n" in body else body
        body = body.rsplit("```", 1)[0]
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_judgement(text: str, total: int) -> dict:
    """判定输出 -> 结构化结果。任何不可信的地方都记成 error，不猜。"""
    parsed = _extract_json_object(text)
    if parsed is None:
        return {"error": "判定输出里没有 JSON 对象", "raw": (text or "")[:200]}

    checked = parsed.get("checked")
    if not isinstance(checked, int) or isinstance(checked, bool):
        return {"error": f"缺少 checked 字段或不是整数：{checked!r}",
                "raw": (text or "")[:200]}
    if checked != total:
        # 模型漏判/多判说明它没有按编号逐句处理，这时候的比例没有意义
        return {"error": f"checked={checked} 与句子数 {total} 不一致",
                "raw": (text or "")[:200]}

    unsupported: list[dict] = []
    seen: set[int] = set()
    for item in parsed.get("unsupported") or []:
        if isinstance(item, dict):
            raw_id = item.get("id")
            why = str(item.get("why") or "")[:60]
        else:
            raw_id, why = item, ""
        try:
            number = int(raw_id)
        except (TypeError, ValueError):
            continue
        if number < 1 or number > total or number in seen:
            continue
        seen.add(number)
        unsupported.append({"id": number, "why": why})

    supported = total - len(unsupported)
    return {
        "supported": supported,
        "total": total,
        "ratio": round(supported / total, 4) if total else 0.0,
        "unsupported": unsupported,
        "error": "",
    }


def judge_answer(llm: Any, question: str, answer: Any) -> dict:
    """调一次模型判定忠实度。`llm` 只要有 `complete(messages, ...)` 即可。"""
    text = str(getattr(answer, "text", "") or "")
    sentences = split_sentences(text)[:MAX_JUDGE_SENTENCES]
    if not sentences:
        return {"error": "答案里没有可判定的句子"}
    messages = build_judge_messages(question, list(getattr(answer, "citations", []) or []),
                                    sentences)
    try:
        raw, usage = llm.complete(messages, temperature=0.0, max_tokens=600)
    except Exception as exc:  # noqa: BLE001 —— 判定失败不该让整轮评测崩
        return {"error": f"{type(exc).__name__}: {exc}"}
    result = parse_judgement(str(raw or ""), len(sentences))
    result["usage"] = dict(usage or {})
    if result.get("unsupported"):
        # 报告里要能直接看到原句，不然只有编号没法复盘
        by_id = {i: text for i, text in enumerate(sentences, start=1)}
        for item in result["unsupported"]:
            item["sentence"] = by_id.get(item["id"], "")
    return result


def evaluate_one(question: Question, answer: Any, *,
                 judge_llm: Any = None) -> AnswerOutcome:
    """一道题的结果整理。`answer` 只要求是 `Answer` 那样的鸭子类型。"""
    text = str(getattr(answer, "text", "") or "")
    citations = list(getattr(answer, "citations", []) or [])
    hits = list(getattr(answer, "hits", []) or [])
    cited_ids = {str(getattr(c, "chunk_id", "")) for c in citations}
    gold = set(question.gold or ())

    gold_in_context = bool(gold & cited_ids)
    gold_hit_rank = 0
    for item in hits:
        if str(getattr(item, "chunk_id", "")) in gold:
            try:
                gold_hit_rank = int(getattr(item, "rank", 0) or 0)
            except (TypeError, ValueError):
                gold_hit_rank = 0
            break

    uncited, sentences = count_uncited(text)
    outcome = AnswerOutcome(
        qid=question.qid,
        question=question.question,
        category=question.category,
        difficulty=question.difficulty,
        refused=bool(getattr(answer, "refused", False)),
        refuse_reason=str(getattr(answer, "refuse_reason", "") or ""),
        refusal_retried=bool(getattr(answer, "refusal_retried", False)),
        answer_chars=len(text),
        n_citations=len(citations),
        gold_in_context=gold_in_context,
        gold_hit_rank=gold_hit_rank,
        invalid_refs=list(invalid_refs(text, len(citations))),
        uncited_sentences=uncited,
        sentence_count=sentences,
        usage=dict(getattr(answer, "usage", {}) or {}),
        latency_ms=int(getattr(answer, "latency_ms", 0) or 0),
    )
    if judge_llm is not None and not outcome.refused:
        # 拒答的答案是一句固定话术，没内容可判，也就不花这次 token
        outcome.judge = judge_answer(judge_llm, question.question, answer)
    return outcome


def summarize(outcomes: list[AnswerOutcome]) -> dict:
    """聚合。没有样本时一律给 0 / None，不给 NaN —— 报告里要能读。"""
    n = len(outcomes)
    refused = [o for o in outcomes if o.refused]
    # 「本可以答」的口径是 gold 进了**送进提示词的检索结果**（gold_hit_rank>0），
    # 不是「被答案引用」。用后者的口径会把「召回到了却没引、还拒答」的题漏掉 ——
    # 那恰恰是模型层拒答过严的证据。实测 40 题里有 2 道（gold 排第 1、第 2）被判成
    # 「资料不足」，用旧口径这两条会被算成「本来就没召回」。
    retrieved = [o for o in outcomes if o.gold_hit_rank > 0]
    cited = [o for o in outcomes if o.gold_in_context]
    false_refusal = [o for o in refused if o.gold_hit_rank > 0]
    answered = [o for o in outcomes if not o.refused]
    judged = [o for o in outcomes if o.support_ratio() is not None]
    failed = [o for o in outcomes if (o.judge or {}).get("error")]

    def ratio(part: int, whole: int) -> float:
        return round(part / whole, 4) if whole else 0.0

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    by_category: dict[str, dict] = {}
    for outcome in outcomes:
        row = by_category.setdefault(outcome.category, {"n": 0, "refused": 0,
                                                        "ratios": []})
        row["n"] += 1
        row["refused"] += 1 if outcome.refused else 0
        if outcome.support_ratio() is not None:
            row["ratios"].append(outcome.support_ratio())

    return {
        "n": n,
        "n_refused": len(refused),
        "refusal_rate": ratio(len(refused), n),
        # 复核救回来的题数：第一次只回 INSUFFICIENT，重问后**答出来了**。
        # 复核后仍拒答（包括只写「资料未覆盖」的）单独计数，别混进「救回」。
        "n_retried": len([o for o in outcomes
                          if o.refusal_retried and not o.refused]),
        "n_retry_refused": len([o for o in outcomes
                                if o.refusal_retried and o.refused]),
        # 漏斗：检索命中 -> 被答案引用 -> 逐句有依据
        "n_retrieved": len(retrieved),
        "gold_retrieved_rate": ratio(len(retrieved), n),
        "n_cited": len(cited),
        "gold_cited_rate": ratio(len(cited), n),
        "n_false_refusal": len(false_refusal),
        "false_refusal_rate": ratio(len(false_refusal), len(retrieved)),
        "mean_uncited_ratio": mean([o.uncited_ratio() for o in answered]),
        "invalid_refs_total": sum(len(o.invalid_refs) for o in outcomes),
        "judged": len(judged),
        "judge_error": len(failed),
        "mean_support_ratio": mean([o.support_ratio() for o in judged
                                    if o.support_ratio() is not None]),
        "unsupported_sentences": sum(len((o.judge or {}).get("unsupported") or [])
                                     for o in judged),
        "by_category": {
            name: {
                "n": row["n"],
                "refused": row["refused"],
                "mean_support_ratio": mean(row["ratios"]),
            }
            for name, row in sorted(by_category.items())
        },
        "mean_latency_ms": int(sum(o.latency_ms for o in outcomes) / n) if n else 0,
        "total_tokens": sum(int((o.usage or {}).get("total_tokens") or 0)
                            for o in outcomes),
    }


def run(questions: list[Question], answerer: Any, *,
        top_k: int = 5, judge_llm: Any = None, collections: list[str] | None = None,
        config: dict | None = None,
        progress: Callable[[int, int, Question], None] | None = None
        ) -> tuple[list[AnswerOutcome], dict, dict]:
    """跑一批题：检索 + 生成 +（可选）判定。返回 `(记录, 汇总, 配置)`。"""
    started = time.time()
    outcomes: list[AnswerOutcome] = []
    for index, question in enumerate(questions, start=1):
        answer = answerer.answer(question.question, top_k=top_k,
                                 collections=collections)
        outcomes.append(evaluate_one(question, answer, judge_llm=judge_llm))
        if progress is not None:
            progress(index, len(questions), question)

    meta = {
        "questions": len(questions),
        "top_k": int(top_k),
        "judge": judge_llm is not None,
        "collections": list(collections or []),
        "elapsed_sec": round(time.time() - started, 1),
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta.update(config or {})
    return outcomes, summarize(outcomes), meta


def save_answer_run(path: Path | str, outcomes: list[AnswerOutcome],
                    summary: dict, config: dict) -> Path:
    """落盘。顶层键和检索评测保持一致（config / summary / per_query）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": dict(config or {}),
        "summary": summary,
        "per_query": [o.to_dict() for o in outcomes],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return target


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_answer_report(outcomes: list[AnswerOutcome], summary: dict,
                         config: dict, *, failure_limit: int | None = None) -> str:
    """纯文本报告。数字都带分母，避免「85%」这种查不到样本量的写法。"""
    lines: list[str] = []
    lines.append(f"  题数        : {summary['n']}")
    lines.append(f"  配置        : top_k={config.get('top_k')} "
                 f"judge={config.get('judge')} 耗时={config.get('elapsed_sec')}s")
    lines.append("")
    lines.append("  · 拒答")
    lines.append(f"    拒答          : {summary['n_refused']}/{summary['n']} "
                 f"({_pct(summary['refusal_rate'])})")
    lines.append(f"    复核救回      : {summary['n_retried']} 题"
                 f"（第一次拒答、重问后作答）")
    if summary.get("n_retry_refused"):
        lines.append(f"    复核后仍拒答  : {summary['n_retry_refused']} 题"
                     f"（重问后只声明了资料未覆盖）")
    lines.append(f"    假拒答        : {summary['n_false_refusal']}/"
                 f"{summary['n_retrieved']} ({_pct(summary['false_refusal_rate'])})"
                 f"   ← gold 进了提示词却拒答")
    lines.append("")
    lines.append("  · 漏斗（gold 块走到哪一步）")
    lines.append(f"    检索前 {config.get('top_k')} 条 : {summary['n_retrieved']}/"
                 f"{summary['n']} ({_pct(summary['gold_retrieved_rate'])})")
    lines.append(f"    被答案引用    : {summary['n_cited']}/{summary['n']} "
                 f"({_pct(summary['gold_cited_rate'])})")
    lines.append("")
    lines.append("  · 引用")
    lines.append(f"    无引用句占比  : {_pct(summary['mean_uncited_ratio'])}（已答的题平均）")
    lines.append(f"    无效引用编号  : {summary['invalid_refs_total']} 处")
    lines.append("")
    lines.append("  · 忠实度（逐句判定）")
    if summary["judged"]:
        lines.append(f"    平均支持率    : {_pct(summary['mean_support_ratio'])}"
                     f"（{summary['judged']} 题判定成功）")
        lines.append(f"    无依据句      : {summary['unsupported_sentences']} 句")
    else:
        lines.append("    平均支持率    : 未判定（--no-judge，或这一轮全部失败）")
    if summary["judge_error"]:
        lines.append(f"    [!] 判定失败  : {summary['judge_error']} 题"
                     f"（不算进支持率，见 per_query 的 judge.error）")
    lines.append(f"    平均耗时      : {summary['mean_latency_ms']} ms/题，"
                 f"生成 token 合计 {summary['total_tokens']}")

    categories = summary.get("by_category") or {}
    if len(categories) > 1:
        lines.append("")
        lines.append("  · 按题型")
        for name, row in categories.items():
            lines.append(f"    {name:<12} n={row['n']:<3} 拒答={row['refused']:<2} "
                         f"支持率={_pct(row['mean_support_ratio'])}")

    # 复盘清单：无依据的句子最值得看，其次假拒答
    bad = [o for o in outcomes if (o.judge or {}).get("unsupported")]
    if bad:
        limit = len(bad) if failure_limit is None else failure_limit
        lines.append("")
        lines.append("  · 没有依据的句子")
        for outcome in bad[:limit]:
            for item in (outcome.judge.get("unsupported") or [])[:3]:
                lines.append(f"    [{outcome.qid}] 第 {item['id']} 句："
                             f"{item.get('sentence', '')[:60]}")
                if item.get("why"):
                    lines.append(f"        原因：{item['why']}")
    return "\n".join(lines)
