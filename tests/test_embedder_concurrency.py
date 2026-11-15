"""ONNX 推理的串行化回归测试。

背景：这台机器上的 onnxruntime-directml 在**并发调用 `session.run()`** 时会让
整个进程堆损坏（实测退出码 0xC0000374，没有 traceback，捕获不到）。现场是
Web 服务收到两个同时到达的 `/api/search` 就整体消失。

所以 `OnnxEmbedder._run_batch` / `OnnxReranker.score` 外面必须有锁。这个文件
用一个「会记录并发峰值」的假 session 断言这一点 —— 不需要真模型，0.1 秒跑完。

如果有人为了「提升并发」把锁去掉，这里会立刻红。
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb.retrieve.embedder import OnnxEmbedder, OnnxReranker  # noqa: E402


class RecordingSession:
    """假 InferenceSession：记录同时进入 `run()` 的线程数峰值。"""

    def __init__(self, width: int = 4, sleep: float = 0.01) -> None:
        self.width = width
        self.sleep = sleep
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def run(self, output_names, feed):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.sleep)
        with self._lock:
            self.active -= 1
        batch = len(next(iter(feed.values())))
        return [np.ones((batch, self.width), dtype=np.float32)]


class _Enc:
    def __init__(self) -> None:
        self.ids = [1, 2, 3]
        self.attention_mask = [1, 1, 1]


class FakeTokenizer:
    def encode_batch(self, texts):
        return [_Enc() for _ in texts]


def make_embedder(session: RecordingSession) -> OnnxEmbedder:
    """绕过 __init__（不碰真模型），只装出 `_run_batch` 需要的那几个属性。"""
    embedder = OnnxEmbedder.__new__(OnnxEmbedder)
    embedder.max_length = 16
    embedder.tokenizer = FakeTokenizer()
    embedder.session = session
    embedder.input_names = ["input_ids", "attention_mask"]
    embedder.output_names = ["sentence_embedding"]
    embedder._lock = threading.Lock()
    return embedder


def make_reranker(session: RecordingSession) -> OnnxReranker:
    reranker = OnnxReranker.__new__(OnnxReranker)
    reranker.max_length = 16
    reranker.tokenizer = FakeTokenizer()
    reranker.session = session
    reranker.input_names = ["input_ids", "attention_mask"]
    reranker._lock = threading.Lock()
    return reranker


def run_threads(target, count: int = 4):
    threads = [threading.Thread(target=target) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


class EmbedderSerializationTests(unittest.TestCase):
    def test_encode_calls_never_overlap(self) -> None:
        session = RecordingSession()
        embedder = make_embedder(session)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                embedder.encode(["并发查询"], batch_size=1)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        run_threads(worker, count=4)

        self.assertEqual(errors, [])
        self.assertEqual(session.calls, 4)
        # 峰值必须是 1：并发进入 session.run 就是堆损坏的触发条件
        self.assertEqual(session.max_active, 1)

    def test_encode_still_works_after_serialization(self) -> None:
        session = RecordingSession(width=4)
        embedder = make_embedder(session)
        vector = embedder.encode(["第一句", "第二句"], batch_size=2)
        self.assertEqual(vector.shape, (2, 4))


class RerankerSerializationTests(unittest.TestCase):
    def test_score_calls_never_overlap(self) -> None:
        session = RecordingSession(width=1)
        reranker = make_reranker(session)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                reranker.score("查询", ["段落一", "段落二"], batch_size=2)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        run_threads(worker, count=4)

        self.assertEqual(errors, [])
        self.assertEqual(session.max_active, 1)

    def test_empty_passages_returns_empty_without_calling_session(self) -> None:
        session = RecordingSession()
        reranker = make_reranker(session)
        scores = reranker.score("查询", [])
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(session.calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
