"""「今日复习」的题库：从评测结果的失败案例里出题。

设计上只有一个前提：**题库来自已经在仓库里的评测结果**（`results/run_hybrid.json`），
不引入用户系统、定时任务或任何额外存储。

- 哪道题：`date.toordinal() % 题库大小`，同一天永远是同一道，隔天换一道。
  不需要 cron，也不需要「今天有没有复习过」的状态：进页面时算一次就有答案。
  「今天」按东八区算（见 `review_today`），不直接用 `date.today()`。
- 出什么题：`per_query` 里 `rank` 缺失或大于 `REVIEW_TOP_K` 的条目，也就是
  前 k 条没命中的题。默认 k=5，与 README 里 Recall@5 的口径一致，**不能**用
  结果文件里的 `config.top_k`（那是 10，用来算 Recall@10 的），否则一条都选不出来。
- 题目里带 `gold`（评测标注的正确块 id），Web 层用它去 `retriever.meta` 里取
  原文，比重新检索更准，也省一次查询向量编码。

这个模块只做数据整形，不 import FastAPI，也不碰 retriever：解析失败一律返回
空题库 + `reason`，让服务照常起来、界面显示一句人话，而不是让 `/api/review/today`
把整个页面拖成 500。
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "REVIEW_TOP_K",
    "REVIEW_MAX_CHUNKS",
    "REVIEW_TZ",
    "load_review_bank",
    "review_failures",
    "review_today",
    "parse_review_day",
]

# 「今日复习」按东八区算日期。用固定偏移而不是 Asia/Shanghai 是因为：
# 中国没有夏令时，固定 +08:00 与它等价；而 Windows 上的 zoneinfo 需要额外的
# tzdata 包，这台机器 pip 装不了。这样本地（+08）和 HF Space（容器默认 UTC）
# 算出来的「今天」是同一个，不会出现北京时间早上 8 点才换题。
REVIEW_TZ = timezone(timedelta(hours=8))

# 「前 k 条没命中」的 k。题库 = rank 缺失或 > k 的题。
REVIEW_TOP_K = 5

# 复习卡片里最多放几段参考原文。gold 有时是 3 个块（跨节综合题），
# 全放出来会把卡片撑得比屏幕还长。
REVIEW_MAX_CHUNKS = 3

# `day` 查询参数的格式。见 parse_review_day 里对 3.11+ fromisoformat 的说明。
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _short_path(path: Path) -> str:
    """给界面显示的短路径：只保留最后两级。

    结果文件在真实项目里是 `G:\\kb\\results\\run_hybrid.json`，整串放进卡片
    只是噪音；两级足够看出是哪个文件。
    """
    parts = [part for part in Path(path).parts if part not in ("", "/", "\\")]
    return "/".join(parts[-2:]) if parts else str(path)


def _rank_of(value: Any) -> int | None:
    """`rank` 字段 -> int；缺失 / null / 非数字一律当「没命中」。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def review_failures(run: Mapping[str, Any] | None, *,
                    top_k: int = REVIEW_TOP_K) -> list[dict]:
    """从评测结果里挑出失败题，保持原顺序。

    每条只保留复习用得上的字段。`question` 为空的条目直接跳过 —— 界面上
    没有题干什么都做不了；`gold` 为空也保留，Web 层会退回现检索。
    """
    if not isinstance(run, Mapping):
        return []
    rows = run.get("per_query")
    if not isinstance(rows, list):
        return []

    items: list[dict] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        rank = _rank_of(row.get("rank"))
        if rank is not None and rank <= top_k:
            continue
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        gold = [str(cid).strip() for cid in (row.get("gold") or [])
                if cid is not None]
        items.append({
            "qid": str(row.get("qid") or ""),
            "question": question,
            "category": str(row.get("category") or ""),
            "difficulty": str(row.get("difficulty") or ""),
            "rank": rank,
            "gold": [cid for cid in gold if cid],
        })
    return items


def load_review_bank(path: Path | str | None, *,
                     top_k: int = REVIEW_TOP_K) -> dict:
    """读评测结果文件，算出复习题库。

    返回的 dict 会原样缓存在 `app.state.review`，字段：

    ============  ========================================================
    items         失败题列表，每条 {qid, question, category, difficulty, rank, gold}
    file          结果文件路径（原样，调试用）
    source        界面显示的短路径，如 ``results/run_hybrid.json``
    top_k         判定「没命中」的 k（前 k 条没命中算失败）
    eval_top_k    评测时实际取了多少条（界面用来说明「没进前 N」）
    n_evaluated   评测集条数
    reason        题库为空时的原因；可用时是空串
    ============  ========================================================

    读文件失败**不抛异常**：复习入口是页面上的一小块，不该让整个服务起不来。
    """
    info: dict = {
        "items": [],
        "file": str(path) if path else "",
        "source": _short_path(Path(path)) if path else "",
        "top_k": int(top_k),
        "eval_top_k": 0,
        "n_evaluated": 0,
        "reason": "",
    }
    if not path:
        info["reason"] = "没有配置复习题库（[web].review_file 留空？）"
        return info

    target = Path(path)
    try:
        run = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        info["reason"] = f"复习题库读不出来：{exc}"
        return info
    except ValueError as exc:
        # JSON 坏了 / 文件为空。消息里带上文件名，别让人猜是哪个文件
        info["reason"] = f"复习题库不是合法 JSON：{target.name}（{exc}）"
        return info
    if not isinstance(run, dict) or "summary" not in run:
        info["reason"] = (f"复习题库格式不对：{target.name} 缺 summary / per_query，"
                          "不是评测结果文件")
        return info

    rows = run.get("per_query")
    if isinstance(rows, list):
        info["n_evaluated"] = len(rows)
    config = run.get("config")
    if isinstance(config, Mapping):
        info["eval_top_k"] = _rank_of(config.get("top_k")) or 0

    info["items"] = review_failures(run, top_k=top_k)
    if not info["items"]:
        info["reason"] = (f"{info['source']} 里没有失败题："
                          f"{info['n_evaluated']} 条全部命中前 {info['top_k']}")
    return info


def review_today(now: datetime | None = None) -> date:
    """今天的日期，按东八区算。

    不直接用 `date.today()`：那取的是进程本地时区。本地跑（+08）没问题，
    但 HF Space 的容器默认 UTC，同一个「今天」会相差 8 小时，题目在北京时间
    早上 8 点才换。固定成 +08 两个环境就一致了。

    `now` 只给测试用：传一个 tz-aware 的 datetime 就能固定住时间；
    传 naive 的会被当成 UTC。
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(REVIEW_TZ).date()


def parse_review_day(text: str | None, *, today: date | None = None) -> date:
    """`day` 查询参数 -> date。空 / 不传 = 今天（东八区）；格式不对抛 ValueError。

    加这个参数只是为了「看别天的题」（界面上的换一题、测试里的固定日期），
    正常使用不需要传。先按 `YYYY-MM-DD` 卡一道正则再交给 fromisoformat：
    3.11 之后的 fromisoformat 也吃 `20261001` 这种省掉连字符的写法，
    接口上不该把两种都当成合法输入。
    """
    raw = str(text or "").strip()
    if not raw:
        return today or review_today()
    if not _DAY_RE.match(raw):
        raise ValueError(f"day 必须是 YYYY-MM-DD，收到 {raw!r}")
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"day 必须是 YYYY-MM-DD，收到 {raw!r}") from exc
