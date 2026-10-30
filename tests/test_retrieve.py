"""Phase 2 检索层测试（标准库 unittest，无需 pytest）。

运行：
    python tests\\test_retrieve.py
    python -m unittest tests.test_retrieve -v

每个用例对应一条真实会出错的地方，不是凑覆盖率：

  · bigram 数量       —— 少一个 bigram 就少一整类召回，且不会报错
  · 内置 hash()       —— 跨进程随机盐，索引进程与查询进程向量空间不同
  · 稀有词 IDF        —— 停用字若权重不归零，排序会被「的」主导
  · 存盘往返          —— 倒排区间算错时，加载后只是「少几条结果」，很隐蔽
  · dense 对照暴力实现 —— argpartition 的边界（k==N、并列分数）
  · RRF 权重          —— 权重没生效时融合结果看起来依然「合理」
  · embedder 类型混用 —— 不报错、只返回垃圾结果，所以要有断言守着
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb.retrieve import (  # noqa: E402
    BM25Index,
    DenseIndex,
    HashingEmbedder,
    Retriever,
    SearchHit,
    build_index,
    rrf_fuse,
    tokenize,
    tokenize_counts,
)

# 临时目录放在工作区内的 .tmp 下，不用 tempfile 的默认位置。
# 本机沙箱只允许写工作区，系统 %TEMP% 会被拒绝（PermissionError），
# 而这个测试要在受限环境里也能一键跑起来。
TMP_ROOT = ROOT / ".tmp"


@contextmanager
def temp_dir(prefix: str = "t"):
    """工作区内的临时目录，退出时清理。"""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"{prefix}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# 测试用的小语料：四个主题各一篇，外加一篇同时含「缓存」和「一致性」的
# 交叉文档。故意让块长差异大（长度归一化的测试靠它）。
CORPUS = [
    {
        "chunk_id": "c-tcp",
        "doc_id": "d-net",
        "repo": "CS-Notes",
        "rel_path": "network/tcp.md",
        "breadcrumb": "网络 / TCP",
        "line_start": 10,
        "line_end": 40,
        "text": "TCP 三次握手：客户端发送 SYN，服务端回复 SYN+ACK，"
                "客户端再发送 ACK。三次握手是为了确认双方的收发能力。",
        "embed_text": "网络 / TCP\n\nTCP 三次握手：客户端发送 SYN，服务端回复 SYN+ACK，"
                      "客户端再发送 ACK。三次握手是为了确认双方的收发能力。",
    },
    {
        "chunk_id": "c-hashmap",
        "doc_id": "d-java",
        "repo": "JavaGuide",
        "rel_path": "docs/hashmap.md",
        "breadcrumb": "Java / 集合 / HashMap",
        "line_start": 100,
        "line_end": 130,
        "text": "HashMap 在 JDK1.8 中引入红黑树，当链表长度超过 8 时转换为红黑树，"
                "扩容阈值为 0.75。红黑树的引入是为了避免哈希冲突导致的查询退化。",
        "embed_text": "Java / 集合 / HashMap\n\nHashMap 在 JDK1.8 中引入红黑树，"
                      "当链表长度超过 8 时转换为红黑树，扩容阈值为 0.75。",
    },
    {
        "chunk_id": "c-redis",
        "doc_id": "d-redis",
        "repo": "JavaGuide",
        "rel_path": "docs/redis.md",
        "breadcrumb": "数据库 / Redis",
        "line_start": 5,
        "line_end": 60,
        "text": "Redis 的持久化有 RDB 和 AOF 两种方式。RDB 是快照，恢复快但可能丢数据；"
                "AOF 记录写命令，数据安全但文件大。生产环境通常混合使用。"
                "Redis 缓存穿透、缓存击穿、缓存雪崩是常见的面试考点。",
        "embed_text": "数据库 / Redis\n\nRedis 的持久化有 RDB 和 AOF 两种方式。",
    },
    {
        "chunk_id": "c-thread",
        "doc_id": "d-java",
        "repo": "JavaGuide",
        "rel_path": "docs/thread.md",
        "breadcrumb": "Java / 并发 / 线程池",
        "line_start": 1,
        "line_end": 25,
        "text": "线程池的核心参数包括核心线程数、最大线程数、阻塞队列和拒绝策略。"
                "任务提交后先看核心线程是否已满，再进队列，队列满了才扩容到最大线程数。",
        "embed_text": "Java / 并发 / 线程池\n\n线程池的核心参数包括核心线程数、最大线程数。",
    },
    {
        "chunk_id": "c-cache",
        "doc_id": "d-mix",
        "repo": "notes",
        "rel_path": "cache.md",
        "breadcrumb": "分布式 / 缓存",
        "line_start": 3,
        "line_end": 9,
        "text": "缓存与数据库一致性：先更新数据库再删除缓存，配合延迟双删降低不一致窗口。",
        "embed_text": "分布式 / 缓存\n\n缓存与数据库一致性：先更新数据库再删除缓存。",
    },
]


def _write_corpus(path: Path, chunks: list[dict] | None = None) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for c in (chunks or CORPUS):
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    return path


def _cosine_bruteforce(mat: np.ndarray, q: np.ndarray) -> np.ndarray:
    """教科书式余弦：逐条算点积除以模长。用来对照 DenseIndex。"""
    qn = q / np.linalg.norm(q)
    out = np.zeros(mat.shape[0], dtype=np.float64)
    for i in range(mat.shape[0]):
        row = mat[i].astype(np.float64)
        out[i] = float(row @ qn.astype(np.float64) / (np.linalg.norm(row) + 1e-12))
    return out


class TestTokenizer(unittest.TestCase):
    def test_pure_chinese_bigram_count(self):
        """4 个字的纯中文：4 个 unigram + 3 个 bigram = 7 个 token。"""
        tokens = tokenize("深度学习")
        self.assertEqual(len(tokens), 7)
        self.assertEqual(tokens[:4], ["深", "度", "学", "习"])
        self.assertEqual(tokens[4:], ["深度", "度学", "学习"])

    def test_single_char_has_no_bigram(self):
        self.assertEqual(tokenize("猫"), ["猫"])

    def test_long_chinese_run(self):
        """n 个字的连续中文串 -> n 个 unigram + (n-1) 个 bigram。"""
        text = "一二三四五六七八九十"
        tokens = tokenize(text)
        self.assertEqual(len(tokens), 10 + 9)

    def test_pure_ascii(self):
        self.assertEqual(tokenize("Hello World"), ["hello", "world"])
        # 数字和下划线算同一个 token 的一部分
        self.assertEqual(tokenize("Python3.13 and snake_case"),
                         ["python3", "13", "and", "snake_case"])

    def test_mixed_cjk_ascii_does_not_cross_boundaries(self):
        """中英混排：不产生跨语种的假 bigram（「习p」这种必须不存在）。"""
        tokens = tokenize("中文abc英文")
        self.assertIn("abc", tokens)
        self.assertIn("中文", tokens)
        self.assertIn("英文", tokens)
        for bad in ("文a", "abc英", "c英"):
            self.assertNotIn(bad, tokens)

    def test_cjk_punctuation_skipped_and_not_glued(self):
        """CJK 标点不产生 token，也不参与 bigram（「文，英」不能出现）。"""
        tokens = tokenize("中文，English；测试。")
        self.assertEqual(tokens.count("english"), 1)
        self.assertIn("中文", tokens)
        self.assertIn("测试", tokens)
        for bad in ("文，", "，e", "；", "。"):
            self.assertNotIn(bad, tokens)
        # 标点两边的中文串不能连成一个 bigram
        self.assertNotIn("文英", tokens)

    def test_fullwidth_folded_to_halfwidth(self):
        """NFKC：全角字母数字折成半角，全角查询能命中半角语料。"""
        self.assertEqual(tokenize("ＡＢＣ１２３"), tokenize("abc123"))

    def test_case_folded(self):
        self.assertEqual(tokenize("HashMap"), tokenize("hashmap"))

    def test_zero_width_removed(self):
        self.assertEqual(tokenize("深度\u200b学习"), tokenize("深度学习"))

    def test_empty_and_blank(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])
        self.assertEqual(tokenize("   \n\t "), [])

    def test_counts_consistent_with_tokenize(self):
        counts = tokenize_counts("深度学习，深度学习")
        self.assertEqual(counts["深度"], 2)
        self.assertEqual(counts["学习"], 2)
        self.assertEqual(sum(counts.values()), len(tokenize("深度学习，深度学习")))


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.texts = [c["embed_text"] for c in CORPUS]
        self.index = BM25Index(k1=1.5, b=0.75)
        self.index.build(self.texts)

    def test_ranking_is_sensible(self):
        """查「红黑树」应该把 HashMap 那篇排第一。"""
        hits = self.index.search("红黑树 链表", top_k=3)
        self.assertTrue(hits, "应该有命中")
        top_doc = CORPUS[hits[0][0]]["chunk_id"]
        self.assertEqual(top_doc, "c-hashmap")
        # 分数必须严格降序
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rare_term_wins(self):
        """只出现在一篇里的稀有词（三次握手），那篇必须排第一。

        这条同时验 IDF 生效：如果 IDF 是常数，「三次握手」这种只在 c-tcp
        出现的词就无法压过其他文档里大量重复的常见字。
        """
        hits = self.index.search("三次握手", top_k=5)
        self.assertEqual(CORPUS[hits[0][0]]["chunk_id"], "c-tcp")
        # 其他文档完全不含这几个字，不该有分数
        self.assertEqual(len(hits), 1)
        self.assertGreater(hits[0][1], 0.0)

    def test_idf_is_positive_and_rare_higher(self):
        """IDF 恒正（平滑的 +1 项），且 df 越小 IDF 越大。"""
        self.assertTrue((self.index.idf > 0).all(), "IDF 必须恒正，否则负权重会搅乱排序")
        rare = self.index.lookup("握手")
        common = self.index.lookup("的")
        self.assertIsNotNone(rare)
        self.assertIsNotNone(common)
        self.assertGreater(self.index.idf[rare], self.index.idf[common])

    def test_common_stopword_contributes_little(self):
        """只由常见字（的、是、在）组成的查询，总分应该远低于稀有词查询。"""
        common_score = self.index.search("的", top_k=1)
        rare_score = self.index.search("红黑树", top_k=1)
        self.assertTrue(common_score and rare_score)
        self.assertLess(common_score[0][1], rare_score[0][1] / 5)

    def test_length_normalization_active(self):
        """b=0.75 时，长文档里的同一个词贡献低于短文档。"""
        # 两篇文档词频完全相同，只有长度不同
        short = "缓存 一致性"
        long = "缓存 一致性 " + " ".join(f"填充{k}" for k in range(50))
        idx = BM25Index()
        idx.build([short, long])
        hits = dict(idx.search("缓存", top_k=2))
        self.assertGreater(hits[0], hits[1], "短文档里的同样词频应该得更高分")
        # b=0 时长度归一化关闭，两篇分数应该相同
        idx_flat = BM25Index(b=0.0)
        idx_flat.build([short, long])
        flat = dict(idx_flat.search("缓存", top_k=2))
        self.assertAlmostEqual(flat[0], flat[1], places=5)

    def test_unknown_query_returns_empty(self):
        self.assertEqual(self.index.search("zzzz-nonexistent-token-qqq", top_k=5), [])
        self.assertEqual(self.index.search("", top_k=5), [])

    def test_top_k_respected(self):
        hits = self.index.search("缓存 数据库 线程", top_k=2)
        self.assertLessEqual(len(hits), 2)

    def test_save_load_roundtrip_identical(self):
        """存盘再加载，检索结果（doc_index 与分数）必须逐条一致。"""
        with temp_dir("bm25") as tmp:
            path = tmp / "bm25"
            self.index.save(path)
            loaded = BM25Index.load(path)
            self.assertEqual(len(loaded), len(self.index))
            self.assertEqual(loaded.vocab_size, self.index.vocab_size)
            self.assertEqual(loaded.n_postings, self.index.n_postings)
            self.assertAlmostEqual(loaded.avgdl, self.index.avgdl, places=6)
            for query in ["红黑树", "三次握手", "缓存 一致性", "线程池 队列", "Redis AOF"]:
                before = self.index.search(query, top_k=5)
                after = loaded.search(query, top_k=5)
                self.assertEqual([d for d, _ in before], [d for d, _ in after],
                                 f"查询 {query!r} 的 doc 顺序变了")
                for (d1, s1), (d2, s2) in zip(before, after):
                    self.assertEqual(d1, d2)
                    self.assertAlmostEqual(s1, s2, places=5)

    def test_saved_files_are_not_pickle(self):
        """落盘文件必须是 numpy 纯数据 + JSON，不能含 pickle。"""
        with temp_dir("bm25files") as tmp:
            path = tmp / "bm25"
            self.index.save(path)
            self.assertTrue((path / "postings.npz").exists())
            self.assertTrue((path / "vocab.json").exists())
            self.assertTrue((path / "meta.json").exists())
            vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
            self.assertIsInstance(vocab, dict)
            # allow_pickle=False 能读出来，说明不是 object 数组
            with np.load(path / "postings.npz", allow_pickle=False) as data:
                self.assertEqual(data["term_ids"].dtype, np.int32)
                self.assertEqual(data["doc_ids"].dtype, np.int32)
                self.assertEqual(data["tfs"].dtype, np.int16)

    def test_load_missing_raises_actionable(self):
        with temp_dir("bm25miss") as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                BM25Index.load(tmp / "nope")
            self.assertIn("ingest.py index", str(ctx.exception))

    def test_inverted_ranges_cover_all_postings(self):
        """倒排区间的最小不变式：相邻区间首尾相接，总数等于 posting 数。"""
        starts = self.index.seg_starts
        self.assertEqual(starts[0], 0)
        self.assertEqual(int(starts[-1]), self.index.n_postings)
        self.assertTrue(np.all(np.diff(starts) >= 0))


class TestDenseIndex(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20240915)
        raw = rng.normal(size=(64, 32)).astype(np.float32)
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        self.mat = raw
        self.meta = [{"chunk_id": f"c{i}"} for i in range(64)]
        self.index = DenseIndex(self.mat, self.meta)
        self.query = rng.normal(size=32).astype(np.float32)

    def test_matches_bruteforce_cosine(self):
        """top-k 的 doc 顺序与教科书式余弦实现一致。"""
        expected = _cosine_bruteforce(self.mat, self.query)
        expected_order = list(np.argsort(-expected, kind="stable")[:10])
        got = self.index.search(self.query, top_k=10)
        self.assertEqual([d for d, _ in got], expected_order)
        for doc_index, score in got:
            self.assertAlmostEqual(score, expected[doc_index], places=5)

    def test_top_k_larger_than_n(self):
        """k >= N 是 argpartition 的边界，不能崩也不能少结果。"""
        got = self.index.search(self.query, top_k=1000)
        self.assertEqual(len(got), 64)
        self.assertEqual(got[0][0],
                         int(np.argmax(_cosine_bruteforce(self.mat, self.query))))

    def test_top_k_one(self):
        got = self.index.search(self.query, top_k=1)
        self.assertEqual(len(got), 1)

    def test_normalized_and_unnormalized_query_agree(self):
        """查询向量归一化与否结果必须一致（内部会归一化）。"""
        a = [d for d, _ in self.index.search(self.query, top_k=5)]
        b = [d for d, _ in self.index.search(self.query * 7.3, top_k=5)]
        self.assertEqual(a, b)

    def test_score_equals_cosine(self):
        query = self.query / np.linalg.norm(self.query)
        got = self.index.search(query, top_k=3)
        for doc_index, score in got:
            manual = float(self.mat[doc_index] @ query)
            self.assertAlmostEqual(score, manual, places=5)

    def test_dim_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.index.search(np.zeros(8, dtype=np.float32), top_k=3)
        self.assertIn("维度", str(ctx.exception))

    def test_meta_length_checked(self):
        with self.assertRaises(ValueError):
            DenseIndex(self.mat, self.meta[:-1])

    def test_save_load_mmap_roundtrip(self):
        """mmap 加载后检索结果与内存中一致。"""
        with temp_dir("dense") as tmp:
            path = tmp / "dense"
            self.index.save(path)
            loaded = DenseIndex.load(path, mmap=True)
            self.assertEqual(len(loaded), 64)
            self.assertEqual(loaded.dim, 32)
            self.assertEqual(len(loaded.meta), 64)  # 占位列表长度不变式
            self.assertEqual(
                [d for d, _ in loaded.search(self.query, top_k=10)],
                [d for d, _ in self.index.search(self.query, top_k=10)],
            )

    def test_batch_matches_single(self):
        qs = np.stack([self.query, self.query[::-1]])
        batch = self.index.search_batch(qs, top_k=5)
        self.assertEqual(len(batch), 2)
        self.assertEqual([d for d, _ in batch[0]],
                         [d for d, _ in self.index.search(qs[0], top_k=5)])

    def test_empty_index(self):
        empty = DenseIndex(np.zeros((0, 4), dtype=np.float32), [])
        self.assertEqual(empty.search(np.ones(4, dtype=np.float32), top_k=5), [])


class TestRRF(unittest.TestCase):
    def test_known_ranking(self):
        """构造已知名次，验证融合顺序。

        doc 1 出现在两路的前二 -> 应该第一；doc 7 两路都有但一路第一一路
        第二 -> 第二；只被一路召回且名次靠后的 doc 9 垫底。
        """
        fused = rrf_fuse([[7, 1, 2], [1, 7, 9]], k=60)
        order = [d for d, _ in fused]
        self.assertEqual(order[0], 1)
        self.assertEqual(order[1], 7)
        self.assertEqual(set(order[2:]), {2, 9})
        scores = dict(fused)
        self.assertGreater(scores[1], scores[2])
        self.assertGreater(scores[7], scores[9])
        # 精确公式：1/(k+rank)
        self.assertAlmostEqual(scores[2], 1.0 / 63, places=12)
        self.assertAlmostEqual(scores[9], 1.0 / 63, places=12)
        self.assertAlmostEqual(scores[1], 1.0 / 62 + 1.0 / 61, places=12)

    def test_k_effect(self):
        """k 越小，头部名次的优势越大。（差值形式不依赖 k 的具体取值）"""
        a = dict(rrf_fuse([[1, 2]], k=1))
        b = dict(rrf_fuse([[1, 2]], k=1000))
        self.assertGreater(a[1] - a[2], b[1] - b[2])

    def test_weights_effective(self):
        """给第二路更大权重时，只被第二路排第一的文档应该反超。"""
        rankings = [[5, 6], [6, 5]]
        even = dict(rrf_fuse(rankings, k=60))
        self.assertAlmostEqual(even[5], even[6], places=12)
        skewed = dict(rrf_fuse(rankings, k=60, weights=[0.1, 3.0]))
        self.assertGreater(skewed[6], skewed[5])
        self.assertEqual(rrf_fuse(rankings, weights=[0.1, 3.0])[0][0], 6)

    def test_zero_weight_disables_path(self):
        """权重 0 等于关掉那一路：只被它召回的文档不该出现。"""
        fused = rrf_fuse([[1, 2], [3, 4]], weights=[1.0, 0.0])
        self.assertEqual([d for d, _ in fused], [1, 2])

    def test_duplicates_in_one_ranking(self):
        fused = dict(rrf_fuse([[3, 3, 4]]))
        self.assertAlmostEqual(fused[3], 1.0 / 61)  # 只按最好名次算一次
        self.assertAlmostEqual(fused[4], 1.0 / 63)

    def test_single_ranking_preserves_order(self):
        self.assertEqual([d for d, _ in rrf_fuse([[9, 2, 5]])], [9, 2, 5])

    def test_empty_input(self):
        self.assertEqual(rrf_fuse([]), [])
        self.assertEqual(rrf_fuse([[], []]), [])

    def test_ties_break_by_doc_index(self):
        """分数并列时按 doc_index 升序，保证可复现。"""
        self.assertEqual([d for d, _ in rrf_fuse([[5], [3], [9]])], [3, 5, 9])

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            rrf_fuse([[1]], k=0)
        with self.assertRaises(ValueError):
            rrf_fuse([[1], [2]], weights=[1.0])


class TestHashingEmbedder(unittest.TestCase):
    def test_deterministic_same_instance(self):
        emb = HashingEmbedder(dim=256)
        a = emb.encode(["缓存 一致性", "红黑树"])
        b = emb.encode(["缓存 一致性", "红黑树"])
        np.testing.assert_array_equal(a, b)

    def test_deterministic_across_processes(self):
        """跨进程一致：内置 hash() 有随机盐，这个断言就是用来钉死这个坑的。

        子进程用不同的 PYTHONHASHSEED 跑同一段编码，结果必须逐位相同。
        """
        script = (
            "import sys, numpy as np;"
            f"sys.path.insert(0, r'{ROOT / 'src'}');"
            "from kb.retrieve import HashingEmbedder;"
            "e = HashingEmbedder(dim=128);"
            "v = e.encode(['深度学习 框架 TCP 握手', 'redis 持久化']);"
            "print(','.join(f'{x:.6f}' for x in v.ravel()))"
        )
        outs = []
        for seed in ("0", "12345"):
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True,
                env={"PYTHONHASHSEED": seed, "PATH": "",
                     "SYSTEMROOT": "C:\\Windows"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outs.append(proc.stdout.strip())
        self.assertEqual(outs[0], outs[1], "不同 PYTHONHASHSEED 下嵌入不一致")

        # 与当前进程的结果也要一致
        local = HashingEmbedder(dim=128).encode(
            ["深度学习 框架 TCP 握手", "redis 持久化"])
        remote = np.array([float(x) for x in outs[0].split(",")], dtype=np.float32)
        np.testing.assert_allclose(local.ravel(), remote, atol=1e-6)

    def test_l2_normalized(self):
        emb = HashingEmbedder(dim=512)
        vecs = emb.encode(["TCP 三次握手", "缓存穿透与缓存雪崩", "short", ""])
        norms = np.linalg.norm(vecs, axis=1)
        for i in range(3):
            self.assertAlmostEqual(float(norms[i]), 1.0, places=5)
        # 空文本 -> 零向量，不产生 NaN
        self.assertEqual(float(norms[3]), 0.0)
        self.assertFalse(np.isnan(vecs).any())

    def test_similar_text_scores_higher(self):
        emb = HashingEmbedder(dim=512)
        emb.build_idf(["缓存 一致性 数据库 更新 删除", "红黑树 链表 哈希冲突",
                       "线程池 队列 拒绝策略", "TCP 握手 连接"])
        v = emb.encode([
            "缓存与数据库一致性",
            "缓存与数据库的一致性如何保证",
            "线程池的核心参数",
        ])
        cos = v @ v.T
        self.assertGreater(cos[0, 1], cos[0, 2])
        self.assertGreater(cos[1, 0], cos[1, 2])

    def test_dissimilar_texts_not_collinear(self):
        """哈希桶不带符号位的话，任意两条文本的余弦都会偏高。"""
        emb = HashingEmbedder(dim=512)
        v = emb.encode(["TCP 三次握手 网络协议", "红烧肉 菜谱 家常做法"])
        self.assertLess(float(v[0] @ v[1]), 0.3)

    def test_dim_enforced(self):
        emb = HashingEmbedder(dim=64)
        self.assertEqual(emb.encode(["abc"], batch_size=1).shape, (1, 64))
        self.assertEqual(emb.dim, 64)
        with self.assertRaises(ValueError):
            HashingEmbedder(dim=0)

    def test_batch_size_does_not_change_result(self):
        emb = HashingEmbedder(dim=128)
        texts = ["中文一", "中文二", "英文 three", "混合 mixed 文本"]
        a = emb.encode(texts, batch_size=1)
        b = emb.encode(texts, batch_size=3)
        np.testing.assert_array_equal(a, b)


class TestSearchHit(unittest.TestCase):
    def test_roundtrip(self):
        hit = SearchHit(
            chunk_id="abc-0001-def", doc_id="abc", repo="JavaGuide",
            rel_path="docs/hashmap.md", breadcrumb="Java / 集合",
            line_start=12, line_end=88, text="中文正文 with english",
            score=0.8125, source="hybrid", rank=3,
        )
        d = hit.to_dict()
        self.assertEqual(
            set(d),
            {"chunk_id", "doc_id", "repo", "rel_path", "breadcrumb",
             "line_start", "line_end", "text", "score", "source", "rank"},
        )
        back = SearchHit.from_dict(d)
        self.assertEqual(back, hit)
        self.assertEqual(SearchHit.from_dict(hit.to_dict()).to_dict(), d)

    def test_from_dict_tolerates_missing_optional(self):
        hit = SearchHit.from_dict({"chunk_id": "x", "text": "t", "score": 1})
        self.assertEqual(hit.rank, 0)
        self.assertEqual(hit.line_start, 0)

    def test_location(self):
        hit = SearchHit("c", "d", "JavaGuide", "docs/a.md", "", 3, 9, "t", 0.1, "dense")
        self.assertEqual(hit.location(), "JavaGuide/docs/a.md:3-9")

    def test_json_serializable(self):
        hit = SearchHit("c", "d", "r", "p", "b", 1, 2, "中文", 0.5, "sparse", 1)
        text = json.dumps(hit.to_dict(), ensure_ascii=False)
        self.assertIn("中文", text)


class TestRetrieverEndToEnd(unittest.TestCase):
    """端到端：HashingEmbedder + BM25 + RRF，全部在小语料上。"""

    @classmethod
    def setUpClass(cls):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        cls.tmp = TMP_ROOT / f"e2e-{uuid.uuid4().hex[:8]}"
        cls.tmp.mkdir(parents=True, exist_ok=True)
        cls.chunks_path = _write_corpus(cls.tmp / "chunks.jsonl")
        cls.index_dir = cls.tmp / "index"
        cls.embedder = HashingEmbedder(dim=256)
        cls.embedder.build_idf([c["embed_text"] for c in CORPUS])
        cls.info = build_index(cls.chunks_path, cls.index_dir, cls.embedder,
                               batch_size=2, progress=False)
        cls.retriever = Retriever(cls.index_dir, embedder=cls.embedder,
                                  cfg={"final_top_k": 5, "dense_top_k": 5,
                                       "sparse_top_k": 5, "fusion_top_k": 5})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_index_files_written(self):
        for name in ("embeddings.npy", "dense_meta.json", "postings.npz",
                     "vocab.json", "meta.json", "chunks_meta.jsonl",
                     "index_meta.json"):
            self.assertTrue((self.index_dir / name).exists(), f"缺 {name}")
        self.assertEqual(self.info["n_chunks"], len(CORPUS))
        self.assertEqual(self.info["dim"], 256)

    def test_meta_keeps_text_drops_embed_text(self):
        """索引元数据保留正文（重排要用），但不重复存 embed_text。"""
        first = json.loads((self.index_dir / "chunks_meta.jsonl")
                           .read_text(encoding="utf-8").splitlines()[0])
        self.assertIn("text", first)
        self.assertIn("chunk_id", first)
        self.assertNotIn("embed_text", first)

    def test_sparse_only_recall(self):
        hits = self.retriever.search("三次握手", top_k=3, use_dense=False)
        self.assertTrue(hits)
        self.assertEqual(hits[0].source, "sparse")
        self.assertEqual(hits[0].chunk_id, "c-tcp")
        self.assertEqual(hits[0].rank, 1)
        self.assertGreater(hits[0].score, 0.0)

    def test_dense_only_recall(self):
        hits = self.retriever.search("缓存 一致性", top_k=3, use_dense=True,
                                     use_sparse=False)
        self.assertTrue(hits)
        self.assertEqual(hits[0].source, "dense")
        self.assertIn("c-cache", [h.chunk_id for h in hits])

    def test_hybrid_recall_ranks_relevant_first(self):
        hits = self.retriever.search("线程池 核心线程数 拒绝策略", top_k=5)
        self.assertEqual(hits[0].source, "hybrid")
        self.assertEqual(hits[0].chunk_id, "c-thread")
        self.assertEqual([h.rank for h in hits], list(range(1, len(hits) + 1)))

    def test_dense_and_sparse_disabled_raises(self):
        with self.assertRaises(ValueError):
            self.retriever.search("x", use_dense=False, use_sparse=False)

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.retriever.search("   ", top_k=3), [])

    def test_hit_fields_populated(self):
        hit = self.retriever.search("红黑树", top_k=1, use_sparse=True,
                                    use_dense=False)[0]
        self.assertEqual(hit.repo, "JavaGuide")
        self.assertEqual(hit.rel_path, "docs/hashmap.md")
        self.assertEqual(hit.breadcrumb, "Java / 集合 / HashMap")
        self.assertEqual((hit.line_start, hit.line_end), (100, 130))
        self.assertIn("红黑树", hit.text)
        self.assertTrue(hit.location().endswith("docs/hashmap.md:100-130"))

    def test_json_output_roundtrips_through_dict(self):
        hits = self.retriever.search("缓存", top_k=3)
        payload = json.dumps([h.to_dict() for h in hits], ensure_ascii=False)
        back = [SearchHit.from_dict(d) for d in json.loads(payload)]
        self.assertEqual(back, hits)

    def test_embedder_type_mismatch_raises(self):
        """hashing 索引配 onnx 查询必须报错，不能静默返回垃圾。"""
        from kb.retrieve.embedder import Embedder

        class FakeOnnx(Embedder):
            dim = 256

            def encode(self, texts, batch_size=16):
                return np.zeros((len(texts), 256), dtype=np.float32)

        with self.assertRaises(ValueError) as ctx:
            Retriever(self.index_dir, embedder=FakeOnnx())
        self.assertIn("不匹配", str(ctx.exception))

    def test_dim_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Retriever(self.index_dir, embedder=HashingEmbedder(dim=64))
        self.assertIn("维度", str(ctx.exception))

    def test_auto_embedder_from_index_meta(self):
        """不传 embedder 时按 index_meta.json 自动装配 HashingEmbedder。"""
        r = Retriever(self.index_dir, cfg={"dense_top_k": 5, "sparse_top_k": 5,
                                           "fusion_top_k": 5})
        hits = r.search("三次握手", top_k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk_id, "c-tcp")

    def test_missing_index_dir_raises_actionable(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            Retriever(self.tmp / "no-such-index")
        self.assertIn("ingest.py index", str(ctx.exception))

    def test_rerank_without_dir_does_not_crash(self):
        """--rerank 但没配 reranker_dir 时降级为不重排，并给出提示。"""
        r = Retriever(self.index_dir, embedder=self.embedder,
                      cfg={"dense_top_k": 5, "sparse_top_k": 5, "fusion_top_k": 5,
                           "reranker_dir": None})
        hits = r.search("缓存", top_k=3, use_rerank=True)
        self.assertTrue(hits)
        self.assertNotEqual(hits[0].source, "rerank")

    def test_limit_load(self):
        from kb.retrieve.pipeline import load_chunks
        self.assertEqual(len(load_chunks(self.chunks_path, limit=2)), 2)
        self.assertEqual(len(load_chunks(self.chunks_path)), len(CORPUS))

    def test_onnx_missing_model_is_actionable(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            from kb.retrieve.pipeline import _make_embedder
            _make_embedder("onnx", self.tmp / "no-model", 512, 4)
        msg = str(ctx.exception)
        self.assertIn("fetch_model.py", msg)
        self.assertIn("hashing", msg)


class TestOnnxErrorPaths(unittest.TestCase):
    """ONNX 路径在模型不可用时的报错必须是可操作的。

    这些断言故意不加载真实模型（2 GB 权重还在下载，tokenizers 也还没装），
    只覆盖「文件不齐」这一层 —— 这恰恰是同事拿不到模型时最先撞上的路径。
    """

    def test_embedder_missing_dir_message(self):
        from kb.retrieve.embedder import OnnxEmbedder

        with self.assertRaises(FileNotFoundError) as ctx:
            OnnxEmbedder(ROOT / ".tmp" / "definitely-no-model")
        msg = str(ctx.exception)
        self.assertIn("fetch_model.py", msg)
        self.assertIn("hf-mirror.com", msg)
        self.assertIn("hashing", msg)

    def test_embedder_incomplete_dir_lists_missing_files(self):
        from kb.retrieve.embedder import OnnxEmbedder

        with temp_dir("onnxpartial") as tmp:
            (tmp / "config.json").write_text('{"hidden_size": 1024}', encoding="utf-8")
            with self.assertRaises(FileNotFoundError) as ctx:
                OnnxEmbedder(tmp)
            msg = str(ctx.exception)
            self.assertIn("model.onnx", msg)
            self.assertIn("tokenizer.json", msg)
            self.assertIn("下载", msg)

    def test_reranker_missing_dir_message(self):
        from kb.retrieve.embedder import OnnxReranker

        with self.assertRaises(FileNotFoundError) as ctx:
            OnnxReranker(ROOT / ".tmp" / "definitely-no-reranker")
        msg = str(ctx.exception)
        self.assertIn("fetch_model.py", msg)
        self.assertIn("--rerank", msg)

    def test_reranker_incomplete_dir(self):
        from kb.retrieve.embedder import OnnxReranker

        with temp_dir("rerankerpartial") as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                OnnxReranker(tmp)
            self.assertIn("model.onnx", str(ctx.exception))

    def test_live_model_dir_reports_download_state(self):
        """如果本机模型目录存在但没下完，报错里要指出下载未完成。

        这条在模型下好之后会自动变成「能加载」的分支，所以用条件断言：
        要么构造成功，要么报错信息里明确指向 model.onnx_data 不完整。
        """
        from kb.retrieve.embedder import OnnxEmbedder

        model_dir = Path(r"G:\kb\models\bge-m3")
        if not model_dir.exists():
            self.skipTest("本机没有 bge-m3 模型目录")
        try:
            OnnxEmbedder(model_dir, max_length=64)
        except FileNotFoundError as exc:
            self.assertIn("下载", str(exc))
        except RuntimeError as exc:
            # tokenizers 缺失或 onnxruntime 加载失败
            msg = str(exc)
            self.assertTrue(
                "tokenizers" in msg or "model.onnx_data" in msg or "ONNX" in msg,
                f"报错信息没有指向真正的原因：{msg[:200]}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
