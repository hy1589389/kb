"""指标层测试：`Metrics` 容器 + `/metrics`、`/api/metrics` 两个路由。

跑法：
    python tests/test_metrics.py
    python -m unittest tests.test_metrics -v

不加载模型、不碰真实索引：retriever / answerer 都是鸭子类型的假对象。
覆盖的都是「一定会出错」的几处：

  · 分位数算法   —— 最近秩法在小样本上不能编出不存在的精度（n=10 时 p99 就是最大值）
  · 环形缓冲     —— 样本和请求摘要都有上限，否则长跑的进程内存一直涨
  · 线程安全     —— 同步路由跑在线程池里，计数是并发写
  · 标签基数     —— 标签只能是路由模板这类有限枚举，不能出现查询词
  · 标签转义     —— 值里有引号 / 反斜杠时 Prometheus 文本不能被写坏
  · 自己不计自己 —— 抓取 `/metrics` 不该进延迟统计
"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows 控制台默认 cp936，中文断言消息会炸，这里强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from kb.generate.types import REFUSE_TEXT, Answer, Citation  # noqa: E402
from kb.retrieve.types import SearchHit  # noqa: E402
from kb.web.app import create_app  # noqa: E402
from kb.web.metrics import PERCENTILES, Metrics  # noqa: E402

TMP_ROOT = ROOT / ".tmp"

HIT_TEXT = "HashMap 在链表长度达到 8 时转红黑树，低于 6 时退回链表。"


def make_hit(index: int = 1, *, score: float = 0.03) -> SearchHit:
    return SearchHit(
        chunk_id=f"chunk-{index}",
        doc_id=f"doc-{index}",
        repo="JavaGuide",
        rel_path="docs/hashmap.md",
        breadcrumb="JavaGuide > Map > HashMap 的底层实现",
        line_start=12,
        line_end=40,
        text=HIT_TEXT,
        score=score,
        source="hybrid",
        rank=index,
    )


class MetricsContainerTests(unittest.TestCase):
    """`Metrics` 本身：计数、分位、上限、并发、渲染。"""

    def test_counter_accumulates_by_label(self) -> None:
        metrics = Metrics()
        metrics.incr("kb_search_total", {"mode": "hybrid", "rerank": "0"})
        metrics.incr("kb_search_total", {"mode": "hybrid", "rerank": "0"}, 2.0)
        metrics.incr("kb_search_total", {"mode": "dense", "rerank": "0"})

        counters = metrics.snapshot()["counters"]
        by_labels = {tuple(sorted(item["labels"].items())): item["value"]
                     for item in counters}
        self.assertEqual(by_labels[(("mode", "hybrid"), ("rerank", "0"))], 3.0)
        self.assertEqual(by_labels[(("mode", "dense"), ("rerank", "0"))], 1.0)

    def test_label_order_does_not_split_series(self) -> None:
        """同一组标签换个传参顺序必须落到同一条序列上。"""
        metrics = Metrics()
        metrics.incr("kb_http_requests_total", {"endpoint": "/api/search", "status": "200"})
        metrics.incr("kb_http_requests_total", {"status": "200", "endpoint": "/api/search"})
        self.assertEqual(len(metrics.snapshot()["counters"]), 1)

    def test_quantiles_nearest_rank(self) -> None:
        """1..100 的 p50/p95/p99 正好是 50/95/99，不上浮不乱插值。"""
        metrics = Metrics()
        for value in range(1, 101):
            metrics.observe("kb_http_latency_ms", value)
        quantiles = metrics.snapshot()["latencies"][0]["quantiles"]
        self.assertEqual(quantiles["p50"], 50)
        self.assertEqual(quantiles["p95"], 95)
        self.assertEqual(quantiles["p99"], 99)

    def test_quantiles_small_sample_do_not_invent_precision(self) -> None:
        """n=3 时三个分位都应当落在真实样本上，且 p99 是最大值。"""
        metrics = Metrics()
        for value in (10, 20, 30):
            metrics.observe("kb_http_latency_ms", value)
        quantiles = metrics.snapshot()["latencies"][0]["quantiles"]
        self.assertEqual(sorted(set(quantiles.values())), [20, 30])
        self.assertEqual(quantiles["p99"], 30)

    def test_samples_are_bounded_but_count_is_not(self) -> None:
        """环形缓冲只影响分位数，`_count` / `_sum` 统计全量观测。"""
        metrics = Metrics(max_samples=100)
        for value in range(1, 501):
            metrics.observe("kb_http_latency_ms", value)
        row = metrics.snapshot()["latencies"][0]
        self.assertEqual(row["count"], 500)
        self.assertEqual(row["sum"], float(sum(range(1, 501))))
        # 分位数只看最近 100 条（401..500）
        self.assertEqual(row["quantiles"]["p50"], 450)

    def test_traces_are_bounded(self) -> None:
        metrics = Metrics(max_traces=5)
        for index in range(20):
            metrics.add_trace({"q": f"q{index}"})
        recent = metrics.snapshot()["recent"]
        self.assertEqual(len(recent), 5)
        self.assertEqual([item["q"] for item in recent], ["q15", "q16", "q17", "q18", "q19"])

    def test_concurrent_writes_do_not_lose_counts(self) -> None:
        """8 个线程各写 500 次，总数必须一条不丢。"""
        metrics = Metrics()

        def worker() -> None:
            for _ in range(500):
                metrics.incr("kb_search_total", {"mode": "hybrid", "rerank": "0"})
                metrics.observe("kb_search_latency_ms", 1.0, {"stage": "retrieve"})

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["counters"][0]["value"], 4000)
        self.assertEqual(snapshot["latencies"][0]["count"], 4000)
        self.assertEqual(snapshot["latencies"][0]["sum"], 4000.0)

    def test_render_escapes_label_values(self) -> None:
        metrics = Metrics()
        metrics.incr("kb_http_requests_total",
                     {"endpoint": '/api/x"y\\z', "status": "200"})
        text = metrics.render()
        self.assertIn('kb_http_requests_total{endpoint="/api/x\\"y\\\\z",status="200"} 1', text)
        self.assertIn("# TYPE kb_http_requests_total counter", text)

    def test_render_summary_shape(self) -> None:
        metrics = Metrics()
        metrics.record_http(endpoint="/api/search", status=200, latency_ms=40)
        metrics.record_http(endpoint="/api/search", status=200, latency_ms=60)
        text = metrics.render()

        self.assertIn("# TYPE kb_http_latency_ms summary", text)
        self.assertIn('kb_http_latency_ms{endpoint="/api/search",quantile="0.5"} 40', text)
        self.assertIn('kb_http_latency_ms_sum{endpoint="/api/search"} 100', text)
        self.assertIn('kb_http_latency_ms_count{endpoint="/api/search"} 2', text)
        for percentile in PERCENTILES:
            self.assertIn(f'quantile="{percentile / 100:g}"', text)

    def test_render_ends_with_newline(self) -> None:
        """Prometheus 的文本解析器要求最后一行以换行结束。"""
        metrics = Metrics()
        metrics.incr("kb_search_total", {"mode": "hybrid", "rerank": "0"})
        self.assertTrue(metrics.render().endswith("\n"))

    def test_gauge_is_replaced_not_accumulated(self) -> None:
        metrics = Metrics()
        metrics.set_gauge("kb_index_chunks", 100)
        metrics.set_gauge("kb_index_chunks", 120)
        self.assertEqual(metrics.snapshot()["gauges"][0]["value"], 120.0)

    def test_counter_values_filters_by_name(self) -> None:
        metrics = Metrics()
        metrics.incr("kb_llm_cost_micros_total", {"kind": "prompt"}, 2000)
        metrics.incr("kb_llm_cost_micros_total", {"kind": "completion"}, 4000)
        metrics.incr("kb_search_total", {"mode": "hybrid", "rerank": "0"})
        rows = metrics.counter_values("kb_llm_cost_micros_total")
        self.assertEqual({labels["kind"]: value for labels, value in rows},
                         {"prompt": 2000.0, "completion": 4000.0})


class FakeRetriever:
    """鸭子类型检索器：只要 `search()` 和几个属性，`create_app` 就能用。"""

    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = list(hits)
        self.calls: list[dict] = []
        self.index_dir = str(TMP_ROOT / "index")
        self.index_info = {"n_chunks": len(self.hits), "dim": 1024,
                           "embedder": "OnnxEmbedder"}
        self.embedder = object()
        self.model_dir = str(TMP_ROOT / "models" / "bge-m3")
        self.meta = [hit.to_dict() for hit in self.hits]

    def search(self, query: str, top_k: int = 10, *, use_dense: bool = True,
               use_sparse: bool = True, use_rerank: bool = False,
               collections=None):
        self.calls.append({"query": query, "top_k": top_k,
                           "use_dense": use_dense, "use_sparse": use_sparse})
        return list(self.hits[:top_k])


class FakeAnswerer:
    """固定返回一个 `Answer`，用来触发问答侧的计数。"""

    def __init__(self, *, refused: bool = False, hits: list[SearchHit] | None = None,
                 usage: dict | None = None, invalid_refs: list[int] | None = None,
                 uncited: int = 0, sentences: int = 3) -> None:
        self.refused = refused
        self.hits = list(hits or [])
        self.usage = dict(usage or {})
        self.invalid_refs = list(invalid_refs or [])
        self.uncited = uncited
        self.sentences = sentences
        self.llm = None

    def answer(self, question: str, top_k: int = 5, collections=None) -> Answer:
        citations = [] if self.refused else [Citation.from_hit(self.hits[0], ref=1)]
        return Answer(
            question=question,
            text=REFUSE_TEXT if self.refused else "先检索再生成 [1]。",
            citations=citations,
            refused=self.refused,
            refuse_reason="没有召回到任何段落" if self.refused else "",
            hits=list(self.hits),
            model="fake-model",
            usage=self.usage,
            latency_ms=12,
            invalid_refs=self.invalid_refs,
            uncited_sentences=self.uncited,
            sentence_count=self.sentences,
            context_chunks=0 if self.refused else len(self.hits),
        )


class WebMetricsTests(unittest.TestCase):
    """/metrics 与 /api/metrics 端到端：走真实路由和中间件。"""

    def setUp(self) -> None:
        self.raw = TMP_ROOT / "metrics-corpus"
        (self.raw / "JavaGuide" / "docs").mkdir(parents=True, exist_ok=True)
        (self.raw / "JavaGuide" / "docs" / "hashmap.md").write_text(
            "# HashMap\n", encoding="utf-8")
        self.hits = [make_hit(1), make_hit(2, score=0.02)]

    def build_client(self, answerer=None) -> TestClient:
        app = create_app(FakeRetriever(self.hits), answerer,
                         raw_dir=self.raw, editor="code")
        return TestClient(app)

    def counters(self, client: TestClient) -> dict:
        payload = client.get("/api/metrics").json()
        return {(item["name"], tuple(sorted(item["labels"].items()))): item["value"]
                for item in payload["counters"]}

    def test_search_is_counted_with_route_template_label(self) -> None:
        with self.build_client() as client:
            self.assertEqual(client.get("/api/search",
                                        params={"q": "HashMap", "top_k": 2}).status_code, 200)
            counters = self.counters(client)

        self.assertEqual(counters.get(
            ("kb_search_total", (("mode", "hybrid"), ("rerank", "0")))), 1.0)
        # 标签必须是路由模板，不能带查询词
        self.assertEqual(counters.get(
            ("kb_http_requests_total", (("endpoint", "/api/search"), ("status", "200")))), 1.0)

    def test_metrics_scrape_does_not_count_itself(self) -> None:
        with self.build_client() as client:
            client.get("/api/search", params={"q": "HashMap"})
            self.counters(client)
            client.get("/metrics")
            counters = self.counters(client)

        endpoints = {labels[0][1] for (name, labels) in counters
                     if name == "kb_http_requests_total"}
        self.assertNotIn("/metrics", endpoints)
        self.assertIn("/api/search", endpoints)

    def test_empty_result_is_counted(self) -> None:
        app = create_app(FakeRetriever([]), None, raw_dir=self.raw, editor="code")
        with TestClient(app) as client:
            body = client.get("/api/search", params={"q": "没有这种东西"}).json()
            counters = self.counters(client)

        self.assertEqual(body["count"], 0)
        self.assertEqual(counters.get(("kb_search_empty_total", (("mode", "hybrid"),))), 1.0)

    def test_tokens_and_refusal_are_counted(self) -> None:
        answerer = FakeAnswerer(hits=self.hits,
                                usage={"prompt_tokens": 1200, "completion_tokens": 180},
                                uncited=1)
        with self.build_client(answerer) as client:
            self.assertEqual(client.post("/api/ask", json={"q": "HashMap"}).status_code, 200)
            counters = self.counters(client)

        self.assertEqual(counters.get(("kb_answer_total", (("refused", "0"), ("stage", "ok")))), 1.0)
        self.assertEqual(counters.get(("kb_llm_tokens_total", (("kind", "prompt"),))), 1200.0)
        self.assertEqual(counters.get(("kb_llm_tokens_total", (("kind", "completion"),))), 180.0)

    def test_retrieval_level_refusal_is_distinguished(self) -> None:
        """一条都没召回的拒答，和召回了但模型说不够，要能分开看。"""
        answerer = FakeAnswerer(refused=True, hits=[])
        with self.build_client(answerer) as client:
            self.assertEqual(client.post("/api/ask", json={"q": "不存在"}).status_code, 200)
            counters = self.counters(client)

        self.assertEqual(counters.get(
            ("kb_answer_total", (("refused", "1"), ("stage", "retrieval")))), 1.0)

    def test_recent_trace_keeps_query_and_stage_timings(self) -> None:
        with self.build_client() as client:
            client.get("/api/search", params={"q": "HashMap", "top_k": 2})
            payload = client.get("/api/metrics").json()

        trace = payload["recent"][-1]
        self.assertEqual(trace["endpoint"], "/api/search")
        self.assertEqual(trace["q"], "HashMap")
        self.assertEqual(trace["count"], 2)
        self.assertIn("retrieve", trace["stages_ms"])

    def test_prometheus_body_is_parseable_text(self) -> None:
        with self.build_client() as client:
            client.get("/api/search", params={"q": "HashMap"})
            response = client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/plain"))
        body = response.text
        self.assertIn("# TYPE kb_search_total counter", body)
        self.assertIn("# TYPE kb_search_latency_ms summary", body)
        self.assertIn("# TYPE kb_index_chunks gauge", body)
        # 每一行都必须是「名字{标签} 数值」或注释，不能有空的数值位
        for line in body.splitlines():
            if not line or line.startswith("#"):
                continue
            self.assertRegex(line, r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?\d")

    def test_gauges_reflect_startup_caches(self) -> None:
        with self.build_client() as client:
            gauges = {item["name"]: item["value"]
                      for item in client.get("/api/metrics").json()["gauges"]}

        self.assertEqual(gauges["kb_index_chunks"], 2.0)
        # 单索引模式在集合列表里也是一项（名字是空串），不是 0
        self.assertEqual(gauges["kb_collections"], 1.0)
        self.assertEqual(gauges["kb_answerer_enabled"], 0.0)

    # ---- 成本 ----

    def test_cost_is_off_when_no_price_configured(self) -> None:
        """没填单价就不算钱：宁可没有，也不要拿猜的价格乘出精确感。"""
        answerer = FakeAnswerer(hits=self.hits,
                                usage={"prompt_tokens": 1000, "completion_tokens": 500})
        with self.build_client(answerer) as client:
            client.post("/api/ask", json={"q": "HashMap"})
            cost = client.get("/api/metrics").json()["cost"]

        self.assertFalse(cost["configured"])
        self.assertEqual(cost["yuan_total"], 0.0)
        self.assertEqual(cost["answers"], 1)

    def test_cost_uses_per_million_token_prices(self) -> None:
        """1000 输入 × 2 元/M + 500 输出 × 8 元/M = 6000 微元 = 0.006 元。"""
        answerer = FakeAnswerer(hits=self.hits,
                                usage={"prompt_tokens": 1000, "completion_tokens": 500})
        app = create_app(FakeRetriever(self.hits), answerer, raw_dir=self.raw,
                         editor="code",
                         prices={"input_per_mtok": 2.0, "output_per_mtok": 8.0})
        with TestClient(app) as client:
            client.post("/api/ask", json={"q": "HashMap"})
            cost = client.get("/api/metrics").json()["cost"]

        self.assertTrue(cost["configured"])
        self.assertEqual(cost["micros"]["prompt"], 2000.0)
        self.assertEqual(cost["micros"]["completion"], 4000.0)
        self.assertEqual(cost["yuan_total"], 0.006)
        self.assertEqual(cost["yuan_per_answer"], 0.006)

    def test_cost_per_answer_averages_over_answers(self) -> None:
        answerer = FakeAnswerer(hits=self.hits,
                                usage={"prompt_tokens": 1000, "completion_tokens": 0})
        app = create_app(FakeRetriever(self.hits), answerer, raw_dir=self.raw,
                         editor="code", prices={"input_per_mtok": 1.0})
        with TestClient(app) as client:
            for _ in range(4):
                client.post("/api/ask", json={"q": "HashMap"})
            cost = client.get("/api/metrics").json()["cost"]

        self.assertEqual(cost["answers"], 4)
        self.assertEqual(cost["yuan_total"], 0.004)
        self.assertEqual(cost["yuan_per_answer"], 0.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
