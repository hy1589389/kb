"""评测集加载与校验。

`questions.jsonl` 每行一个 JSON 对象：

    {"qid": "q001", "question": "HashMap 的树化阈值为什么是 8",
     "gold": ["chunk_id_1", "chunk_id_2"], "note": "[easy] 术语精确匹配：……"}

只有 `qid` / `question` / `gold` 是必填字段，其余可选：
  · `category` —— 题型，缺省时从 note 里解析 `类别：xxx`
  · `difficulty` —— easy / medium / hard，缺省时从 note 里解析 `[easy]` 这类标记
  · `note` —— 备注

校验的原则是"报错要指名道姓"：格式错误一律抛 EvalDataError，带上
行号和 qid，不允许静默跳过。评测集被静默丢几条，指标就没意义了。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

VALID_DIFFICULTIES = ("easy", "medium", "hard")

_RE_DIFFICULTY = re.compile(r"\[(easy|medium|hard)\]", re.IGNORECASE)
_RE_CATEGORY = re.compile(
    r"(?:类别|category)\s*[:：]\s*([A-Za-z\u4e00-\u9fff/]+)", re.IGNORECASE)


class EvalDataError(ValueError):
    """评测集格式错误。消息里必须能看出是哪一行、哪个字段。"""


@dataclass
class Question:
    qid: str
    question: str
    gold: list[str]
    note: str = ""
    category: str = "未分类"
    difficulty: str = "unknown"
    line_no: int = 0
    raw: dict = field(default_factory=dict)

    def has_gold(self, chunk_ids) -> bool:
        """给定一批检索结果的 chunk_id，是否有任意一个命中 gold。"""
        return any(cid in self.gold for cid in chunk_ids)


def _require_str(obj: dict, key: str, where: str) -> str:
    if key not in obj:
        raise EvalDataError(f"{where}: 缺字段 {key!r}")
    val = obj[key]
    if not isinstance(val, str) or not val.strip():
        raise EvalDataError(f"{where}: 字段 {key!r} 必须是非空字符串，实际是 {val!r}")
    return val


def parse_question(obj: object, line_no: int = 0, source: str = "") -> Question:
    """把一行 JSON 解析成 Question，格式不对就抛 EvalDataError。"""
    head = f"{source or 'questions.jsonl'} 第 {line_no} 行" if line_no else (source or "question")
    if not isinstance(obj, dict):
        raise EvalDataError(f"{head}: 每行必须是一个 JSON 对象，实际是 {type(obj).__name__}")

    qid = _require_str(obj, "qid", head)
    where = f"{head}（qid={qid}）"
    question = _require_str(obj, "question", where)

    if "gold" not in obj:
        raise EvalDataError(f"{where}: 缺字段 'gold'")
    gold = obj["gold"]
    if not isinstance(gold, list):
        raise EvalDataError(f"{where}: 'gold' 必须是 chunk_id 列表，实际是 {type(gold).__name__}")
    if not gold:
        raise EvalDataError(f"{where}: 'gold' 为空。没有标准答案的题不要放进评测集")
    for i, cid in enumerate(gold):
        if not isinstance(cid, str) or not cid.strip():
            raise EvalDataError(f"{where}: gold[{i}] 不是非空字符串：{cid!r}")
    if len(set(gold)) != len(gold):
        dup = sorted({c for c in gold if gold.count(c) > 1})
        raise EvalDataError(f"{where}: gold 里有重复 chunk_id：{dup}")

    note = obj.get("note", "")
    if note is None:
        note = ""
    if not isinstance(note, str):
        raise EvalDataError(f"{where}: 'note' 必须是字符串，实际是 {type(note).__name__}")

    category = obj.get("category") or ""
    if not isinstance(category, str):
        raise EvalDataError(f"{where}: 'category' 必须是字符串")
    difficulty = obj.get("difficulty") or ""
    if not isinstance(difficulty, str):
        raise EvalDataError(f"{where}: 'difficulty' 必须是字符串")

    if not difficulty:
        # 没写 difficulty 字段就从 note 里找 [easy] 这类标记；找不到归到 unknown 层
        m = _RE_DIFFICULTY.search(note)
        difficulty = m.group(1).lower() if m else "unknown"
    else:
        difficulty = difficulty.strip().lower()
        if difficulty not in VALID_DIFFICULTIES:
            raise EvalDataError(
                f"{where}: difficulty={difficulty!r} 不是 {'/'.join(VALID_DIFFICULTIES)} 之一"
            )

    if not category:
        m = _RE_CATEGORY.search(note)
        category = m.group(1) if m else "未分类"

    return Question(
        qid=qid,
        question=question,
        gold=list(gold),
        note=note,
        category=category,
        difficulty=difficulty,
        line_no=line_no,
        raw=obj,
    )


def load_questions(path: str | Path) -> list[Question]:
    """读取 questions.jsonl。任何一行格式不对都直接抛错，不静默跳过。"""
    p = Path(path)
    if not p.is_file():
        raise EvalDataError(f"评测集文件不存在：{p}")

    questions: list[Question] = []
    seen: dict[str, int] = {}
    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue  # 允许空行和 # 注释行
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvalDataError(f"{p} 第 {line_no} 行不是合法 JSON：{exc.msg}") from exc
            q = parse_question(obj, line_no=line_no, source=str(p))
            if q.qid in seen:
                raise EvalDataError(
                    f"{p} 第 {line_no} 行 qid={q.qid!r} 与第 {seen[q.qid]} 行重复"
                )
            seen[q.qid] = line_no
            questions.append(q)

    if not questions:
        raise EvalDataError(f"{p} 里没有读到任何题目")
    return questions
