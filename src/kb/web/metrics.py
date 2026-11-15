"""Web 层的进程内指标：`/metrics`（Prometheus 文本）和 `/api/metrics`（JSON）。

为什么自己写而不是装 `prometheus_client`：这台机器 pip 装不了包（解包阶段被拒），
部署用的 `requirements-deploy.txt` 也只列了 8 个运行时依赖。指标这件事的关键是
「记什么」，不是「用什么记」，两百行标准库够用，容器里也不用多装东西。

三条约束，每条都对应一个真会踩的坑：

1. **`def` 路由跑在线程池里。** FastAPI 的同步处理器不是协程，uvicorn 会把每个
   请求丢进 anyio 的线程池，所以 `incr()` / `observe()` 会被多个线程同时调用。
   所有可变状态都在 `threading.Lock` 里改。
2. **标签只能是有限枚举。** Prometheus 里每个标签组合是一条独立时间序列，把查询词、
   chunk_id、集合名列表写进标签，内存和抓取体积都会失控。这里只允许
   endpoint / status / mode / rerank / stage / kind 这几组固定值。
3. **不落盘、不持久化。** 魔搭免费档的容器随时重启，写文件只会给人「有历史数据」
   的错觉。延迟样本是环形缓冲（默认 2048 条），请求摘要 64 条，重启即空。
   要长期趋势就得真接一个后端，这件事不在本模块的范围内。

和 `/api/health` 的分工：health 回答「现在能不能用」（就绪、块数、模型），
这里回答「用起来是什么样」（延迟分位、空召回率、拒答率、token 数）。
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["Metrics", "PERCENTILES", "MAX_SAMPLES", "MAX_TRACES"]

# 延迟分位。只取三个：再多也没有人看，而每个分位都是抓取时现算的排序成本。
PERCENTILES: tuple[float, ...] = (50.0, 95.0, 99.0)

# 每个时间序列保留的延迟样本数。2048 条 float 约 16 KB，一条序列一个 deque。
MAX_SAMPLES = 2048

# 请求摘要（`/api/metrics` 里的 recent）。单条几百字节，64 条约几十 KB。
MAX_TRACES = 64

# 已知指标的类型。没登记的按 counter 处理 —— 允许调用方先加指标后补登记，
# 但 Prometheus 文本里没有 TYPE 行会被当成 untyped，所以新指标记得登记。
_KINDS: dict[str, str] = {
    "kb_http_requests_total": "counter",
    "kb_http_latency_ms": "summary",
    "kb_search_total": "counter",
    "kb_search_empty_total": "counter",
    "kb_search_latency_ms": "summary",
    "kb_search_top_score": "summary",
    "kb_answer_total": "counter",
    "kb_answer_invalid_refs_total": "counter",
    "kb_answer_uncited_ratio": "summary",
    "kb_llm_tokens_total": "counter",
    "kb_llm_cost_micros_total": "counter",
    "kb_index_chunks": "gauge",
    "kb_collections": "gauge",
    "kb_answerer_enabled": "gauge",
}

_Labels = tuple[tuple[str, str], ...]


def _key(name: str, labels: Mapping[str, Any] | None) -> tuple[str, _Labels]:
    """`(名字, 排好序的标签)`。排序是为了让同一组标签无论传参顺序都一样。"""
    if not labels:
        return name, ()
    items = [(str(k), str(v)) for k, v in labels.items() if v is not None]
    items.sort()
    return name, tuple(items)


def _group(entries: Mapping[tuple[str, _Labels], Any]) -> dict[str, list[tuple[_Labels, Any]]]:
    """按指标名分组，供渲染时一个指标写一条 TYPE 行。"""
    grouped: dict[str, list[tuple[_Labels, Any]]] = {}
    for (name, labels), value in sorted(entries.items()):
        grouped.setdefault(name, []).append((labels, value))
    return grouped


def _escape(value: str) -> str:
    """Prometheus 标签值转义：反斜杠、双引号、换行。"""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels_text(labels: _Labels, extra: Sequence[tuple[str, str]] = ()) -> str:
    pairs = [f'{name}="{_escape(value)}"' for name, value in labels]
    pairs.extend(f'{name}="{_escape(value)}"' for name, value in extra)
    return "{" + ",".join(pairs) + "}" if pairs else ""


def _number(value: Any) -> str:
    """数字写成 Prometheus 能吃的样子：整数不带小数点，小数最多 6 位有效数字。"""
    number = float(value)
    if math.isfinite(number) and number.is_integer() and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.6g}"


def _quantiles(values: Sequence[float]) -> dict[float, float]:
    """最近秩法（nearest-rank）：floor 插值在小样本上会编出不存在的精度。

    n=100 时 p50 取第 50 个、p95 取第 95 个；n=1 时三个分位都是那一个值。
    """
    data = sorted(values)
    count = len(data)
    if not count:
        return {p: 0.0 for p in PERCENTILES}
    out: dict[float, float] = {}
    for p in PERCENTILES:
        index = math.ceil(p / 100.0 * count) - 1
        out[p] = data[min(count - 1, max(0, index))]
    return out


class Metrics:
    """线程安全的进程内指标容器。

    用法只有四个方法：`incr`（计数）、`observe`（观测延迟/比例）、`set_gauge`
    （瞬时值，抓取前刷新）、`record_http`（中间件用，一次请求一行）。

    读写成本：写是一次字典查找加一次浮点加法，在锁内几十纳秒；读（`render` /
    `snapshot`）要把样本复制出来排序，2048 条约 1 ms —— 只发生在抓取时。
    """

    def __init__(self, *, max_samples: int = MAX_SAMPLES,
                 max_traces: int = MAX_TRACES) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, _Labels], float] = {}
        self._samples: dict[tuple[str, _Labels], deque[float]] = {}
        self._sums: dict[tuple[str, _Labels], float] = {}
        self._counts: dict[tuple[str, _Labels], int] = {}
        self._gauges: dict[tuple[str, _Labels], float] = {}
        self._traces: deque[dict] = deque(maxlen=max(1, int(max_traces)))
        self._max_samples = max(1, int(max_samples))
        self._started = time.time()

    # ---------------------------------------------------------------- 写入

    def incr(self, name: str, labels: Mapping[str, Any] | None = None,
             value: float = 1.0) -> None:
        key = _key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + float(value)

    def set_gauge(self, name: str, value: float,
                  labels: Mapping[str, Any] | None = None) -> None:
        key = _key(name, labels)
        with self._lock:
            self._gauges[key] = float(value)

    def observe(self, name: str, value: float,
                labels: Mapping[str, Any] | None = None) -> None:
        """记一次观测。`_sum` / `_count` 统计全部观测，分位数只统计最近
        `max_samples` 条 —— 这是 summary 的正常语义，不是丢数据。"""
        key = _key(name, labels)
        number = float(value)
        with self._lock:
            samples = self._samples.get(key)
            if samples is None:
                samples = deque(maxlen=self._max_samples)
                self._samples[key] = samples
            samples.append(number)
            self._sums[key] = self._sums.get(key, 0.0) + number
            self._counts[key] = self._counts.get(key, 0) + 1

    def record_http(self, *, endpoint: str, status: int, latency_ms: float) -> None:
        """一次 HTTP 请求：计数（带状态码）+ 端点级延迟。中间件调用。"""
        self.incr("kb_http_requests_total", {"endpoint": endpoint, "status": str(status)})
        self.observe("kb_http_latency_ms", latency_ms, {"endpoint": endpoint})

    def add_trace(self, record: Mapping[str, Any]) -> None:
        """一条请求摘要。环形缓冲满了自动丢最老的，不报错、不阻塞。"""
        with self._lock:
            self._traces.append(dict(record))

    # ---------------------------------------------------------------- 读取

    def counter_values(self, name: str) -> list[tuple[dict[str, str], float]]:
        """某个计数器每种标签组合的当前值。算「总成本」这类聚合时用。"""
        with self._lock:
            return [
                (dict(labels), value)
                for (metric, labels), value in sorted(self._counters.items())
                if metric == name
            ]

    def snapshot(self) -> dict[str, Any]:
        """`/api/metrics` 的响应体。给人看和给脚本用都够。"""
        with self._lock:
            counters = [
                {"name": name, "labels": dict(labels), "value": value}
                for (name, labels), value in sorted(self._counters.items())
            ]
            gauges = [
                {"name": name, "labels": dict(labels), "value": value}
                for (name, labels), value in sorted(self._gauges.items())
            ]
            latencies = []
            for (name, labels), samples in sorted(self._samples.items()):
                latencies.append({
                    "name": name,
                    "labels": dict(labels),
                    "count": self._counts.get((name, labels), 0),
                    "sum": round(self._sums.get((name, labels), 0.0), 3),
                    "quantiles": {
                        f"p{int(p)}": round(q, 3)
                        for p, q in _quantiles(list(samples)).items()
                    },
                })
            recent = [dict(item) for item in self._traces]
        return {
            "uptime_seconds": round(time.time() - self._started, 1),
            "counters": counters,
            "gauges": gauges,
            "latencies": latencies,
            "recent": recent,
        }

    def render(self) -> str:
        """Prometheus 文本格式（text/plain; version=0.0.4）。

        只写有价值的三种：counter、gauge、summary。summary 的分位数取自环形
        缓冲的最近样本，`_count` / `_sum` 取自全量观测，两者口径不同是 summary
        本来的语义。
        """
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            samples = {key: list(value) for key, value in self._samples.items()}
            sums = dict(self._sums)
            counts = dict(self._counts)

        lines: list[str] = []
        for kind, entries in (("counter", counters), ("gauge", gauges)):
            for name, rows in _group(entries).items():
                lines.append(f"# TYPE {name} {kind}")
                for labels, value in rows:
                    lines.append(f"{name}{_labels_text(labels)} {_number(value)}")

        for name, rows in _group(samples).items():
            lines.append(f"# TYPE {name} summary")
            for labels, values in rows:
                for p, q in _quantiles(values).items():
                    quantile = ("quantile", _number(p / 100.0))
                    lines.append(
                        f"{name}{_labels_text(labels, extra=(quantile,))} {_number(q)}"
                    )
                key = (name, labels)
                lines.append(f"{name}_sum{_labels_text(labels)} {_number(sums.get(key, 0.0))}")
                lines.append(f"{name}_count{_labels_text(labels)} {_number(counts.get(key, 0))}")
        return "\n".join(lines) + "\n"

    def kind_of(self, name: str) -> str:
        """指标类型；没登记过的当 counter（Prometheus 里就是 untyped）。"""
        return _KINDS.get(name, "counter")
