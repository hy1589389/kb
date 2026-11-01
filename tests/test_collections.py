"""多集合（collection）测试：配置解析、目录隔离、指纹幂等、删除同步、质量指标、多索引检索。

运行：
    python tests\\test_collections.py
    python -m unittest tests.test_collections -v

这里不碰 `data/` 下的任何真实目录：所有 raw / index / processed 都在
`.tmp/` 里新建（本机沙箱只允许写工作区，系统 %TEMP% 会被拒绝）。
"""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import kb  # noqa: E402,F401 - 导入即把 .pylibs 插进 sys.path
from kb.config import (  # noqa: E402
    CollectionConfig,
    CollectionsConfig,
    IngestConfig,
)
from kb.ingest.pipeline import (  # noqa: E402
    build_stats,
    ingest_collection,
    legacy_index_dir,
    load_fingerprints,
    run_ingest,
    write_outputs,
)
from kb.ingest.quality import (  # noqa: E402
    THRESHOLDS,
    assess_texts,
    garbled_ratio,
    is_blank_text,
)
from kb.retrieve import HashingEmbedder, Retriever, build_index  # noqa: E402

TMP_ROOT = ROOT / ".tmp"


@contextmanager
def temp_dir(prefix: str = "coll"):
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"{prefix}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def write_pdf(path: Path, *, title: str, heading: str, paragraphs: list[str],
              page: int = 1) -> Path:
    """造一份最小的单页 PDF（用于验证 locator=P3 这种页码）。"""
    import pymupdf  # 由 kb/__init__.py 注入的 .pylibs 提供

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    doc.set_metadata({"title": title})
    pg = doc.new_page(width=595, height=842)
    y = 180
    pg.insert_text((72, y), heading, fontsize=18, fontname="china-s")
    y += 50
    for text in paragraphs:
        pg.insert_textbox(pymupdf.Rect(72, y, 520, y + 90), text, fontsize=10,
                          fontname="china-s")
        y += 100
    doc.save(str(path))
    doc.close()
    return path


def make_config(tmp: Path) -> IngestConfig:
    """两个集合：tech（Markdown，index 在 data/index）+ profile（auto，PDF）。

    `out_dir` 放在 `<tmp>/data/processed`，这样 `legacy_index_dir()` 推出来的
    就是 `<tmp>/data/index` —— 与正式项目里 tech 用 `data/index` 的结构一致。
    """
    cfg = IngestConfig()
    cfg.raw_dir = tmp / "raw"
    cfg.out_dir = tmp / "data" / "processed"
    cfg.chunk.min_chars = 60
    cfg.collections = CollectionsConfig(
        default_search=["tech"],
        items={
            "tech": CollectionConfig(
                name="tech", raw_dir=tmp / "raw" / "tech",
                index_dir=tmp / "data" / "index", parser="markdown",
            ),
            "profile": CollectionConfig(
                name="profile", raw_dir=tmp / "raw" / "profile",
                index_dir=tmp / "data" / "index" / "profile", parser="auto",
                generation="disabled",
            ),
        },
    )
    return cfg


def seed_corpus(cfg: IngestConfig) -> None:
    """写入一份 tech Markdown + 一份 profile PDF。"""
    tech_doc = cfg.raw_dir / "tech" / "JavaGuide" / "docs" / "hashmap.md"
    tech_doc.parent.mkdir(parents=True, exist_ok=True)
    tech_doc.write_text(
        "# Java 集合\n\n## HashMap\n\n"
        "HashMap 在 JDK1.8 中引入红黑树：当链表长度超过 8 时转成红黑树，"
        "节点数降到 6 时退回链表。默认容量 16，负载因子 0.75，扩容时容量翻倍，"
        "并把旧数组的元素重新分配到新数组里。为什么阈值是 8：理想哈希下链表长度"
        "服从泊松分布，长度达到 8 的概率约为千万分之六，这种概率下转成树的成本可以"
        "忽略；而红黑树节点的内存占用约为链表节点的两倍，规模小的时候维护树反而更慢，"
        "所以 8 是「树化概率足够低」和「树化收益足够大」之间的折中。\n",
        encoding="utf-8",
    )
    profile_pdf = cfg.raw_dir / "profile" / "何毓-简历.pdf"
    write_pdf(
        profile_pdf, title="何毓-简历", heading="何毓",
        paragraphs=[
            "求职意向：后端开发。精通 Python，做过本地知识库问答系统，"
            "用 BM25 与 bge-m3 双路召回加 RRF 融合，简历里出现项目经历四个字。",
            "项目经历：本地知识库问答系统。把五个中文技术仓库切成一万八千个块，"
            "按 Markdown 标题层级切分并保留标题路径与原文行号，答案可以回溯到具体行。",
            "技能：Python、FastAPI、SQL 与 Git；熟悉大模型 API 对接、流式输出解析"
            "与结构化提示词，能独立完成检索增强问答的端到端实现。",
        ],
    )


def ingest(cfg: IngestConfig, name: str, **kwargs):
    """跑一个集合并落盘（dry_run 时不落盘）。"""
    result = ingest_collection(cfg, cfg.collections.get(name), verbose=False, **kwargs)
    if not kwargs.get("dry_run"):
        write_outputs(result, cfg, build_stats(result, cfg, 0.0))
    return result


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
COLLECTIONS_TOML = """
[path]
raw_dir = "data/raw"
out_dir = "data/processed"

[collections]
default_search = ["tech"]

[collections.tech]
raw_dir = "data/raw/tech"
index_dir = "data/index"
parser = "markdown"

[collections.papers]
raw_dir = "data/raw/papers"
index_dir = "data/index/papers"
parser = "auto"

[collections.profile]
raw_dir = "data/raw/profile"
index_dir = "data/index/profile"
parser = "auto"
generation = "disabled"
weight = 0.8
"""

LEGACY_TOML = """
[path]
raw_dir = "data/raw"
out_dir = "data/processed"

[chunk]
max_chars = 900
"""


class TestCollectionConfig(unittest.TestCase):
    def _load(self, tmp: Path, text: str) -> IngestConfig:
        path = tmp / "config.toml"
        path.write_text(text, encoding="utf-8")
        return IngestConfig.load(path)

    def test_parses_collections_section(self):
        with temp_dir("cfg") as tmp:
            cfg = self._load(tmp, COLLECTIONS_TOML)
            self.assertEqual(cfg.collections.names(), ["tech", "papers", "profile"])
            self.assertEqual(cfg.collections.default_search, ["tech"])

            tech = cfg.collections.get("tech")
            self.assertEqual(tech.raw_dir, Path("data/raw/tech"))
            self.assertEqual(tech.index_dir, Path("data/index"))
            self.assertEqual(tech.parser, "markdown")
            self.assertEqual(tech.generation, "enabled")

            papers = cfg.collections.get("papers")
            self.assertEqual(papers.parser, "auto")

            profile = cfg.collections.get("profile")
            self.assertEqual(profile.parser, "auto")
            self.assertEqual(profile.generation, "disabled")
            self.assertAlmostEqual(profile.weight, 0.8)

    def test_missing_section_falls_back_to_single_tech(self):
        """没有 [collections] 段 == 只有一个 tech，路径与旧版一致。"""
        with temp_dir("cfg2") as tmp:
            cfg = self._load(tmp, LEGACY_TOML)
            self.assertEqual(cfg.collections.names(), ["tech"])
            self.assertEqual(cfg.collections.default_search, ["tech"])
            tech = cfg.collections.get("tech")
            self.assertEqual(tech.raw_dir, Path("data/raw"))
            self.assertEqual(tech.index_dir, Path("data/index"))
            self.assertEqual(tech.parser, "markdown")
            # 旧的 [chunk] 段照常生效
            self.assertEqual(cfg.chunk.max_chars, 900)

    def test_empty_section_is_legacy(self):
        cfg = CollectionsConfig()
        cfg.apply_section({})
        self.assertEqual(cfg.names(), ["tech"])

    def test_default_search_as_comma_string(self):
        cfg = CollectionsConfig()
        cfg.apply_section({
            "default_search": "tech, profile , ",
            "tech": {"raw_dir": "a", "index_dir": "b", "parser": "markdown"},
            "profile": {"raw_dir": "c", "index_dir": "d", "parser": "pdf"},
        })
        self.assertEqual(cfg.default_search, ["tech", "profile"])

    def test_default_search_filters_unknown_names(self):
        cfg = CollectionsConfig()
        cfg.apply_section({
            "default_search": ["tech", "nope"],
            "tech": {"raw_dir": "a", "index_dir": "b", "parser": "markdown"},
        })
        self.assertEqual(cfg.default_search, ["tech"])

    def test_default_search_empty_means_all(self):
        cfg = CollectionsConfig()
        cfg.apply_section({
            "default_search": [],
            "a": {"parser": "markdown"},
            "b": {"parser": "pdf"},
        })
        self.assertEqual(cfg.default_search, ["a", "b"])

    def test_invalid_parser_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CollectionsConfig().apply_section(
                {"x": {"raw_dir": "a", "index_dir": "b", "parser": "docx"}})
        self.assertIn("markdown", str(ctx.exception))

    def test_invalid_generation_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CollectionsConfig().apply_section(
                {"x": {"raw_dir": "a", "index_dir": "b", "generation": "local"}})
        self.assertIn("disabled", str(ctx.exception))

    def test_unknown_top_level_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CollectionsConfig().apply_section({"default": "tech"})
        self.assertIn("[collections.", str(ctx.exception))

    def test_select_unknown_raises(self):
        cfg = CollectionsConfig()
        with self.assertRaises(KeyError) as ctx:
            cfg.select(["nope"])
        self.assertIn("nope", str(ctx.exception))

    def test_select_defaults_and_dedupes(self):
        cfg = CollectionsConfig()
        cfg.apply_section({
            "default_search": ["b", "a"],
            "a": {"parser": "markdown"},
            "b": {"parser": "markdown"},
        })
        self.assertEqual([c.name for c in cfg.select()], ["b", "a"])
        self.assertEqual([c.name for c in cfg.select(["a", "a"])], ["a"])

    def test_index_dirs_and_weights(self):
        cfg = CollectionsConfig()
        cfg.apply_section({
            "a": {"index_dir": "data/index/a", "weight": 0.5, "parser": "markdown"},
            "b": {"index_dir": "data/index/b", "parser": "markdown"},
        })
        self.assertEqual(cfg.index_dirs(), {"a": Path("data/index/a"),
                                            "b": Path("data/index/b")})
        self.assertEqual(cfg.weight_map(), {"a": 0.5, "b": 1.0})

    def test_resolve_makes_paths_absolute(self):
        with temp_dir("cfg3") as tmp:
            cfg = self._load(tmp, COLLECTIONS_TOML)
            cfg.resolve(tmp)
            tech = cfg.collections.get("tech")
            self.assertTrue(tech.raw_dir.is_absolute())
            self.assertEqual(tech.raw_dir, (tmp / "data/raw/tech").resolve())


# --------------------------------------------------------------------------- #
# 目录隔离
# --------------------------------------------------------------------------- #
class TestCollectionIsolation(unittest.TestCase):
    def test_files_stay_in_their_collection(self):
        with temp_dir("iso") as tmp:
            cfg = make_config(tmp)
            seed_corpus(cfg)
            tech = ingest(cfg, "tech")
            profile = ingest(cfg, "profile")

            tech_paths = {c["rel_path"] for c in tech.chunks}
            profile_paths = {c["rel_path"] for c in profile.chunks}
            self.assertTrue(tech_paths)
            self.assertTrue(profile_paths)
            self.assertFalse(tech_paths & profile_paths)
            self.assertTrue(all("hashmap" in p for p in tech_paths), tech_paths)
            self.assertTrue(all("简历" in p for p in profile_paths), profile_paths)

            tech_file = read_jsonl(cfg.out_dir / "chunks_tech.jsonl")
            profile_file = read_jsonl(cfg.out_dir / "chunks_profile.jsonl")
            self.assertTrue(all(row["collection"] == "tech" for row in tech_file))
            self.assertTrue(all(row["collection"] == "profile" for row in profile_file))
            self.assertFalse(any("简历" in row["rel_path"] for row in tech_file))
            self.assertFalse(any("hashmap" in row["rel_path"] for row in profile_file))

    def test_locator_markdown_vs_pdf(self):
        with temp_dir("loc") as tmp:
            cfg = make_config(tmp)
            seed_corpus(cfg)
            tech = ingest(cfg, "tech")
            profile = ingest(cfg, "profile")

            for chunk in tech.chunks:
                self.assertRegex(chunk["locator"], r"^L\d+-\d+$")
                self.assertGreater(chunk["line_start"], 0)
            for chunk in profile.chunks:
                self.assertRegex(chunk["locator"], r"^P\d+(-\d+)?$")
                # line_start / line_end 在 PDF 里填页码，字段保持存在
                self.assertGreaterEqual(chunk["line_start"], 1)
                self.assertGreaterEqual(chunk["line_end"], chunk["line_start"])

    def test_pdf_repo_is_file_stem(self):
        with temp_dir("repo") as tmp:
            cfg = make_config(tmp)
            seed_corpus(cfg)
            profile = ingest(cfg, "profile")
            self.assertEqual({c["repo"] for c in profile.chunks}, {"何毓-简历"})

    def test_pdf_collection_does_not_scan_markdown(self):
        """parser=pdf 的集合只收 .pdf。"""
        with temp_dir("parserpdf") as tmp:
            cfg = make_config(tmp)
            cfg.collections.items["profile"].parser = "pdf"
            (cfg.raw_dir / "profile").mkdir(parents=True, exist_ok=True)
            (cfg.raw_dir / "profile" / "note.md").write_text(
                "# 标题\n\n这是一段 Markdown，pdf 集合不该收它。" * 5, encoding="utf-8")
            result = ingest(cfg, "profile")
            self.assertEqual(result.chunks, [])
            self.assertEqual(result.docs, [])

    def test_legacy_files_only_for_data_index_collection(self):
        """tech（index_dir=data/index）额外写 chunks_all/manifest/stats，
        别的集合绝不写这些历史文件名。"""
        with temp_dir("legacy") as tmp:
            cfg = make_config(tmp)
            seed_corpus(cfg)
            ingest(cfg, "tech")
            ingest(cfg, "profile")
            self.assertEqual(legacy_index_dir(cfg), (tmp / "data" / "index").resolve())

            out = cfg.out_dir
            for name in ("chunks_all.jsonl", "manifest.jsonl", "stats.json",
                         "chunks_JavaGuide.jsonl"):
                self.assertTrue((out / name).exists(), name)
            # profile 的文件
            for name in ("chunks_profile.jsonl", "manifest_profile.jsonl",
                         "fingerprints_profile.json", "stats_profile.json"):
                self.assertTrue((out / name).exists(), name)
            # chunks_all 里只有 tech 的内容
            rows = read_jsonl(out / "chunks_all.jsonl")
            self.assertTrue(all(row["collection"] == "tech" for row in rows))

    def test_no_legacy_files_without_data_index(self):
        with temp_dir("nolegacy") as tmp:
            cfg = make_config(tmp)
            cfg.collections.items["tech"].index_dir = tmp / "data" / "index" / "tech"
            seed_corpus(cfg)
            ingest(cfg, "tech")
            out = cfg.out_dir
            self.assertTrue((out / "chunks_tech.jsonl").exists())
            self.assertFalse((out / "chunks_all.jsonl").exists())
            self.assertFalse((out / "manifest.jsonl").exists())

    def test_run_ingest_returns_one_result_per_collection(self):
        with temp_dir("run") as tmp:
            cfg = make_config(tmp)
            seed_corpus(cfg)
            results = run_ingest(cfg, collections=["tech", "profile"], verbose=False)
            self.assertEqual([r.collection for r in results], ["tech", "profile"])
            self.assertTrue(all(r.chunks for r in results))


# --------------------------------------------------------------------------- #
# 指纹幂等 / 删除同步
# --------------------------------------------------------------------------- #
class TestFingerprints(unittest.TestCase):
    def test_second_run_skips_everything(self):
        with temp_dir("fp") as tmp:
            cfg = make_config(tmp)
            seed_corpus(cfg)
            first = ingest(cfg, "tech")
            self.assertEqual(first.added, 1)
            before = read_jsonl(cfg.out_dir / "chunks_tech.jsonl")

            second = ingest(cfg, "tech")
            self.assertEqual(second.unchanged, 1)
            self.assertEqual(second.added, 0)
            self.assertEqual(second.updated, 0)
            self.assertEqual(second.skipped, [])
            after = read_jsonl(cfg.out_dir / "chunks_tech.jsonl")
            # 跳过时必须复用上次的块，chunk_id 不能漂
            self.assertEqual([r["chunk_id"] for r in before],
                             [r["chunk_id"] for r in after])

    def test_fingerprint_file_contents(self):
        with temp_dir("fp2") as tmp:
            cfg = make_config(tmp)
            seed_corpus(cfg)
            ingest(cfg, "tech")
            data = json.loads(
                (cfg.out_dir / "fingerprints_tech.json").read_text(encoding="utf-8"))
            self.assertIn("JavaGuide/docs/hashmap.md", data)
            entry = data["JavaGuide/docs/hashmap.md"]
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertIsInstance(entry["mtime"], int)
            self.assertGreaterEqual(entry["chunks"], 1)

    def test_changed_content_reparsed(self):
        with temp_dir("fp3") as tmp:
            cfg = make_config(tmp)
            seed_corpus(cfg)
            ingest(cfg, "tech")
            doc = cfg.raw_dir / "tech" / "JavaGuide" / "docs" / "hashmap.md"
            doc.write_text(
                "# Java 集合\n\n## HashMap\n\n"
                "HashMap 的扩容阈值是 0.75，这个知识库段落被改写过，"
                "旧内容应该从块里消失。\n",
                encoding="utf-8",
            )
            result = ingest(cfg, "tech")
            self.assertEqual(result.updated, 1)
            self.assertEqual(result.added, 0)
            text = "\n".join(c["text"] for c in result.chunks)
            self.assertIn("这个知识库段落被改写过", text)
            self.assertNotIn("节点数降到 6", text)

    def test_deleted_file_chunks_removed(self):
        with temp_dir("fp4") as tmp:
            cfg = make_config(tmp)
            seed_corpus(cfg)
            (cfg.raw_dir / "tech" / "CS-Notes").mkdir(parents=True, exist_ok=True)
            extra = cfg.raw_dir / "tech" / "CS-Notes" / "redis.md"
            extra.write_text(
                "# Redis\n\n## 持久化\n\n"
                "RDB 是快照，AOF 记录写命令，生产环境通常把两种方式混合使用，"
                "兼顾恢复速度和数据安全。\n",
                encoding="utf-8",
            )
            first = ingest(cfg, "tech")
            self.assertEqual(first.added, 2)
            self.assertTrue(any("redis" in c["rel_path"] for c in first.chunks))

            extra.unlink()
            second = ingest(cfg, "tech")
            self.assertEqual(second.removed, ["CS-Notes/redis.md"])
            self.assertFalse(any("redis" in c["rel_path"] for c in second.chunks))
            on_disk = read_jsonl(cfg.out_dir / "chunks_tech.jsonl")
            self.assertFalse(any("redis" in row["rel_path"] for row in on_disk))
            # 指纹里也不能留着已删文件
            fps = load_fingerprints(cfg.out_dir, "tech")
            self.assertNotIn("CS-Notes/redis.md", fps)

    def test_auto_collection_root_markdown_idempotent(self):
        """auto 集合里 raw_dir 根下的 Markdown：repo=文件名，重跑要跳过。

        这是指纹重建最容易踩的坑：Markdown 的 rel_path 相对仓库、PDF 的
        rel_path 相对 raw_dir、根下 Markdown 又是第三种，靠 rel_path 反推
        raw_dir 相对路径会漏掉这一种，导致每次重跑都当新文件重新解析。
        """
        with temp_dir("autoroot") as tmp:
            cfg = make_config(tmp)
            cfg.chunk.drop_below = 10
            profile_dir = cfg.raw_dir / "profile"
            profile_dir.mkdir(parents=True, exist_ok=True)
            (profile_dir / "note.md").write_text(
                "# 根下笔记\n\n放在集合根目录下的 Markdown，内容要够长才不会被块长过滤。\n",
                encoding="utf-8",
            )
            first = ingest(cfg, "profile")
            self.assertEqual(first.added, 1)
            self.assertEqual({(c["repo"], c["rel_path"]) for c in first.chunks},
                             {("note", "note.md")})

            second = ingest(cfg, "profile")
            self.assertEqual(second.unchanged, 1)
            self.assertEqual(second.added, 0)
            self.assertEqual(sorted(load_fingerprints(cfg.out_dir, "profile")),
                             ["note.md"])

    def test_dry_run_writes_nothing(self):
        with temp_dir("fp5") as tmp:
            cfg = make_config(tmp)
            seed_corpus(cfg)
            result = ingest(cfg, "tech", dry_run=True)
            self.assertTrue(result.chunks)
            self.assertFalse(cfg.out_dir.exists())
            self.assertEqual(result.added, 1)
            self.assertTrue(result.reports)
            self.assertEqual(result.reports[0].n_chunks, len(result.chunks))
            self.assertEqual(len(result.reports[0].preview), min(2, len(result.chunks)))

    def test_dry_run_reports_unchanged_status(self):
        with temp_dir("fp6") as tmp:
            cfg = make_config(tmp)
            seed_corpus(cfg)
            ingest(cfg, "tech")
            result = ingest(cfg, "tech", dry_run=True)
            self.assertEqual(result.reports[0].status, "未变更")


# --------------------------------------------------------------------------- #
# 质量指标
# --------------------------------------------------------------------------- #
class TestQuality(unittest.TestCase):
    def test_thresholds_match_spec(self):
        self.assertEqual(THRESHOLDS["avg_chars"], 100)
        self.assertAlmostEqual(THRESHOLDS["garbled_rate"], 0.15)
        self.assertAlmostEqual(THRESHOLDS["blank_ratio"], 0.30)

    def test_garbled_ratio_ignores_chinese_ascii_and_punctuation(self):
        self.assertEqual(garbled_ratio("正常的中文内容，还有 English 和 123。"), 0.0)
        self.assertEqual(garbled_ratio("你好"), 0.0)
        self.assertGreater(garbled_ratio("\ufffd\ufffd\ufffd"), 0.9)
        self.assertEqual(garbled_ratio(""), 0.0)

    def test_blank_text(self):
        self.assertTrue(is_blank_text(""))
        self.assertTrue(is_blank_text("   \n\t "))
        self.assertTrue(is_blank_text("。、，；| |"))
        self.assertFalse(is_blank_text("有字"))
        self.assertFalse(is_blank_text("abc"))

    def test_assess_flags_short_chunks(self):
        report = assess_texts(["短" * 30, "也短" * 20])
        self.assertTrue(report.low_quality)
        self.assertTrue(any("平均块字符数" in r for r in report.reasons), report.reasons)

    def test_assess_flags_garbled(self):
        report = assess_texts(["正常内容" * 20 + "\ufffd" * 100])
        self.assertTrue(report.low_quality)
        self.assertTrue(any("乱码率" in r for r in report.reasons), report.reasons)

    def test_assess_flags_blank_ratio(self):
        texts = ["。", "| |", "，"] + ["正常的一段中文内容，长度足够长。" * 5]
        report = assess_texts(texts)
        self.assertGreater(report.blank_ratio, 0.30)
        self.assertTrue(any("空白块" in r for r in report.reasons), report.reasons)

    def test_assess_clean_text(self):
        report = assess_texts(["这是一段正常的中文内容，长度也够，可以安全入库。" * 6] * 10)
        self.assertFalse(report.low_quality, report.reasons)
        self.assertEqual(report.n_chunks, 10)

    def test_low_quality_marked_in_jsonl(self):
        """乱码文档照样入库，但带上 low_quality: true。"""
        with temp_dir("q1") as tmp:
            cfg = make_config(tmp)
            cfg.chunk.min_chars = 1
            doc = cfg.raw_dir / "tech" / "JavaGuide" / "garbled.md"
            doc.parent.mkdir(parents=True, exist_ok=True)
            doc.write_text(
                "# 乱码文档\n\n## 第一节\n\n" + "\ufffd" * 200 + "\n", encoding="utf-8")
            result = ingest(cfg, "tech")
            self.assertTrue(result.chunks)
            self.assertTrue(all(c.get("low_quality") is True for c in result.chunks))
            self.assertTrue(result.docs[0].low_quality)
            rows = read_jsonl(cfg.out_dir / "chunks_tech.jsonl")
            self.assertTrue(all(row.get("low_quality") is True for row in rows))

    def test_short_chunk_document_flagged(self):
        """平均块字符数 < 100 的文档被标记。关掉合并（min_chars=1）让块保持很短。"""
        with temp_dir("q2") as tmp:
            cfg = make_config(tmp)
            cfg.chunk.min_chars = 1
            doc = cfg.raw_dir / "tech" / "JavaGuide" / "short.md"
            doc.parent.mkdir(parents=True, exist_ok=True)
            doc.write_text(
                "# 短文档\n\n## 一\n\n" + "短" * 50 + "\n\n## 二\n\n" + "短" * 50 + "\n",
                encoding="utf-8",
            )
            result = ingest(cfg, "tech")
            self.assertTrue(result.docs[0].low_quality)
            self.assertTrue(all(c.get("low_quality") for c in result.chunks))

    def test_clean_document_not_flagged(self):
        with temp_dir("q3") as tmp:
            cfg = make_config(tmp)
            doc = cfg.raw_dir / "tech" / "JavaGuide" / "good.md"
            doc.parent.mkdir(parents=True, exist_ok=True)
            doc.write_text(
                "# 正常文档\n\n## 一节\n\n"
                + "这是一段正常的中文技术内容，讲的是 HashMap 的红黑树转换和扩容阈值。" * 8
                + "\n",
                encoding="utf-8",
            )
            result = ingest(cfg, "tech")
            self.assertFalse(result.docs[0].low_quality)
            self.assertFalse(any(c.get("low_quality") for c in result.chunks))


# --------------------------------------------------------------------------- #
# 多索引检索
# --------------------------------------------------------------------------- #
def _write_chunks(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _row(chunk_id: str, *, collection: str, repo: str, rel_path: str,
         text: str, locator: str, low_quality: bool = False) -> dict:
    row = {
        "chunk_id": chunk_id,
        "doc_id": f"d-{chunk_id}",
        "repo": repo,
        "rel_path": rel_path,
        "breadcrumb": f"{collection} / {repo}",
        "line_start": 1,
        "line_end": 2,
        "char_count": len(text),
        "cjk_count": 0,
        "text": text,
        "embed_text": text,
        "collection": collection,
        "locator": locator,
    }
    if low_quality:
        row["low_quality"] = True
    return row


TECH_ROWS = [
    _row("c-hashmap", collection="tech", repo="JavaGuide",
         rel_path="docs/hashmap.md",
         text="HashMap 在 JDK1.8 中引入红黑树，链表长度超过 8 时转换，扩容阈值 0.75。",
         locator="L10-40"),
    _row("c-redis", collection="tech", repo="CS-Notes", rel_path="redis.md",
         text="Redis 持久化有 RDB 和 AOF 两种方式，生产环境通常混合使用。",
         locator="L1-20"),
    _row("c-tcp", collection="tech", repo="CS-Notes", rel_path="tcp.md",
         text="TCP 三次握手：SYN、SYN+ACK、ACK，用来确认双方的收发能力。",
         locator="L3-9"),
]

PROFILE_ROWS = [
    _row("p-project", collection="profile", repo="何毓-简历",
         rel_path="何毓-简历.pdf",
         text="项目经历：本地知识库问答系统，BM25 与 bge-m3 双路召回，RRF 融合。",
         locator="P1"),
    _row("p-skill", collection="profile", repo="何毓-简历",
         rel_path="何毓-简历.pdf",
         text="技能：Python、FastAPI、SQL，熟悉大模型 API 对接与流式输出。",
         locator="P1"),
    _row("p-badscan", collection="profile", repo="扫描件-bak",
         rel_path="扫描件-bak.pdf",
         text="\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd",
         locator="P2", low_quality=True),
]


class TestRetrieverMultiIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        cls.tmp = TMP_ROOT / f"multi-{uuid.uuid4().hex[:8]}"
        cls.tmp.mkdir(parents=True, exist_ok=True)

        cls.embedder = HashingEmbedder(dim=256)
        texts = [r["text"] for r in TECH_ROWS + PROFILE_ROWS]
        cls.embedder.build_idf(texts)

        cls.tech_index = cls.tmp / "index_tech"
        cls.profile_index = cls.tmp / "index_profile"
        for rows, index_dir in ((TECH_ROWS, cls.tech_index),
                                (PROFILE_ROWS, cls.profile_index)):
            chunks_path = _write_chunks(cls.tmp / f"{index_dir.name}.jsonl", rows)
            build_index(chunks_path, index_dir, cls.embedder,
                        batch_size=2, progress=False)

        cls.cfg = {"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 10,
                   "final_top_k": 10}
        cls.retriever = Retriever(
            {"tech": cls.tech_index, "profile": cls.profile_index},
            embedder=cls.embedder, cfg=dict(cls.cfg))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_parts_loaded_once(self):
        self.assertEqual(sorted(self.retriever.parts), ["profile", "tech"])
        self.assertEqual(len(self.retriever.meta),
                         len(TECH_ROWS) + len(PROFILE_ROWS))

    def test_merge_returns_both_collections(self):
        hits = self.retriever.search("HashMap 红黑树 项目经历 双路召回", top_k=6)
        self.assertTrue(hits)
        collections = {hit.collection for hit in hits}
        self.assertEqual(collections, {"tech", "profile"})
        self.assertEqual([h.rank for h in hits], list(range(1, len(hits) + 1)))

    def test_hit_carries_collection_and_locator(self):
        hits = self.retriever.search("HashMap 红黑树", top_k=3)
        top = hits[0]
        self.assertEqual(top.collection, "tech")
        self.assertEqual(top.locator, "L10-40")
        self.assertTrue(top.location().endswith("docs/hashmap.md:L10-40"))

        pdf_hits = self.retriever.search("项目经历 双路召回", top_k=3,
                                         collections=["profile"])
        self.assertTrue(pdf_hits)
        self.assertEqual(pdf_hits[0].collection, "profile")
        self.assertEqual(pdf_hits[0].locator, "P1")

    def test_to_dict_has_collection_and_locator(self):
        data = self.retriever.search("HashMap", top_k=1)[0].to_dict()
        self.assertEqual(data["collection"], "tech")
        self.assertIn("locator", data)

    def test_collection_filter(self):
        tech_only = self.retriever.search("HashMap 项目经历", top_k=10,
                                          collections=["tech"])
        self.assertTrue(tech_only)
        self.assertTrue(all(h.collection == "tech" for h in tech_only))

        profile_only = self.retriever.search("HashMap 项目经历", top_k=10,
                                             collections="profile")
        self.assertTrue(profile_only)
        self.assertTrue(all(h.collection == "profile" for h in profile_only))

    def test_collection_filter_isolates_content(self):
        """查 HashMap 时只在 tech 里找，返回的块里不会出现简历内容。"""
        hits = self.retriever.search("HashMap 红黑树 扩容", top_k=10,
                                     collections=["tech"])
        self.assertTrue(hits)
        self.assertFalse(any("项目经历" in h.text for h in hits))
        self.assertEqual(hits[0].chunk_id, "c-hashmap")

    def test_unknown_collection_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.retriever.search("x", collections=["nope"])
        self.assertIn("nope", str(ctx.exception))

    def test_low_quality_excluded_by_default(self):
        hits = self.retriever.search("扫描件", top_k=10, collections=["profile"])
        self.assertFalse(any(h.chunk_id == "p-badscan" for h in hits))

        hits_all = self.retriever.search("扫描件", top_k=10, collections=["profile"],
                                         include_low_quality=True)
        self.assertTrue(any(h.chunk_id == "p-badscan" for h in hits_all))

    def test_low_quality_excluded_from_all_collections(self):
        hits = self.retriever.search("扫描件 简历 技能", top_k=20)
        self.assertFalse(any(h.chunk_id == "p-badscan" for h in hits))

    def test_collection_weights_affect_order(self):
        """给 tech 极小权重后，profile 的块应该排到前面。"""
        r = Retriever({"tech": self.tech_index, "profile": self.profile_index},
                      embedder=self.embedder,
                      cfg={**self.cfg, "collections": {"tech": 0.01, "profile": 5.0}})
        hits = r.search("HashMap 项目经历 双路召回", top_k=6)
        self.assertTrue(hits)
        self.assertEqual(hits[0].collection, "profile")

    def test_single_index_mode_unchanged(self):
        """旧的 Retriever(Path(...)) 用法仍然只查一个索引。"""
        r = Retriever(self.tech_index, embedder=self.embedder, cfg=dict(self.cfg))
        hits = r.search("HashMap", top_k=3)
        self.assertTrue(hits)
        self.assertTrue(all(h.collection == "tech" for h in hits))
        # 单索引模式下 collections 参数被忽略（索引里没有集合名可比）
        self.assertTrue(r.search("HashMap", top_k=3, collections=["tech"]))

    def test_embedder_mismatch_across_collections_raises(self):
        """两个索引的 embedder 类型不一致时必须报错，不能静默混用。"""
        with temp_dir("mixed") as tmp:
            hashing_dir = tmp / "h"
            chunks_path = _write_chunks(tmp / "h.jsonl", TECH_ROWS)
            build_index(chunks_path, hashing_dir, self.embedder,
                        batch_size=2, progress=False)
            # 伪造一个 onnx 索引：只改 index_meta.json 的 embedder 字段
            onnx_dir = tmp / "o"
            shutil.copytree(hashing_dir, onnx_dir)
            info = json.loads((onnx_dir / "index_meta.json").read_text(encoding="utf-8"))
            info["embedder"] = "OnnxEmbedder"
            (onnx_dir / "index_meta.json").write_text(
                json.dumps(info), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                Retriever({"a": hashing_dir, "b": onnx_dir}, embedder=self.embedder)
            self.assertIn("不匹配", str(ctx.exception))

    def test_missing_index_dir_mentions_collection(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            Retriever({"profile": self.tmp / "no-such"})
        self.assertIn("index_meta.json", str(ctx.exception))

    def test_empty_index_map_raises(self):
        with self.assertRaises(ValueError):
            Retriever({})

    def test_search_does_not_reread_disk(self):
        """索引在 __init__ 里加载一次：构造完再把元数据文件清空，检索照常。

        不删 embeddings.npy —— DenseIndex 用 mmap 打开它，Windows 上文件被
        锁住删不掉。清空 chunks_meta.jsonl 已经足够说明「没在每次请求时回读」。
        """
        with temp_dir("once") as tmp:
            index_dir = tmp / "idx"
            chunks_path = _write_chunks(tmp / "c.jsonl", TECH_ROWS)
            build_index(chunks_path, index_dir, self.embedder,
                        batch_size=2, progress=False)
            r = Retriever(index_dir, embedder=self.embedder, cfg=dict(self.cfg))
            (index_dir / "chunks_meta.jsonl").write_text("", encoding="utf-8")
            hits = r.search("HashMap", top_k=1)
            self.assertTrue(hits)
            self.assertEqual(hits[0].chunk_id, "c-hashmap")
            self.assertEqual(hits[0].breadcrumb, "tech / JavaGuide")


# --------------------------------------------------------------------------- #
# CLI 端到端（ingest / index / search / drop 的参数接线）
# --------------------------------------------------------------------------- #
def cli_config(tmp: Path) -> Path:
    """给 CLI 用的临时 config.toml：路径必须是绝对路径。

    `cmd_*` 会按 PROJECT_ROOT 展开相对路径，而 PROJECT_ROOT 是真实项目根，
    所以这里只能写绝对路径。
    """
    base = tmp.as_posix()
    text = f"""
[path]
raw_dir = "{base}/raw"
out_dir = "{base}/data/processed"

[chunk]
max_chars = 1200
min_chars = 60
drop_below = 40
overlap = 120
code_block_max_chars = 2400

[retrieval]
index_dir = "{base}/data/index"
model_dir = ""
reranker_dir = ""
embed_batch_size = 4
embed_max_length = 512
dense_top_k = 50
sparse_top_k = 50
fusion_top_k = 50
final_top_k = 10
rrf_k = 60
use_reranker = false

[collections]
default_search = ["tech"]

[collections.tech]
raw_dir = "{base}/raw/tech"
index_dir = "{base}/data/index"
parser = "markdown"

[collections.profile]
raw_dir = "{base}/raw/profile"
index_dir = "{base}/data/index/profile"
parser = "auto"
generation = "disabled"
"""
    path = tmp / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def run_cli(argv: list[str]) -> tuple[int, str]:
    """跑一次 kb.cli.main，返回 (退出码, stdout)。"""
    import io
    from contextlib import redirect_stdout

    from kb.cli import main

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(argv)
    return code, buffer.getvalue()


class TestCli(unittest.TestCase):
    def _setup(self, tmp: Path) -> Path:
        cfg = make_config(tmp)
        cfg.out_dir = tmp / "data" / "processed"
        seed_corpus(cfg)
        return cli_config(tmp)

    def test_ingest_index_search(self):
        with temp_dir("cli1") as tmp:
            config = self._setup(tmp)
            code, out = run_cli(["ingest", "--config", str(config)])
            self.assertEqual(code, 0, out)
            self.assertIn("chunks_tech.jsonl", out)
            self.assertTrue((tmp / "data/processed/chunks_profile.jsonl").exists())

            for name in ("tech", "profile"):
                code, out = run_cli(["index", "--collection", name, "--config",
                                     str(config), "--embedder", "hashing"])
                self.assertEqual(code, 0, out)
                self.assertTrue((tmp / "data/index" / name / "index_meta.json").exists()
                                or (tmp / "data/index" / "index_meta.json").exists())

            code, out = run_cli(["search", "HashMap 红黑树", "--config", str(config),
                                 "--top-k", "3"])
            self.assertEqual(code, 0, out)
            self.assertIn("[tech]", out)
            self.assertIn("hashmap", out.lower())

            code, out = run_cli(["search", "项目经历", "--collection", "profile",
                                 "--config", str(config), "--top-k", "2"])
            self.assertEqual(code, 0, out)
            self.assertIn("[profile]", out)
            self.assertIn("P1", out)

    def test_ingest_dry_run_writes_nothing(self):
        with temp_dir("cli2") as tmp:
            config = self._setup(tmp)
            code, out = run_cli(["ingest", "--collection", "profile", "--dry-run",
                                 "--config", str(config)])
            self.assertEqual(code, 0, out)
            self.assertIn("dry-run", out)
            self.assertIn("预览", out)
            self.assertFalse((tmp / "data/processed").exists())

    def test_ingest_unknown_collection(self):
        with temp_dir("cli3") as tmp:
            config = self._setup(tmp)
            code, out = run_cli(["ingest", "--collection", "nope", "--config",
                                 str(config)])
            self.assertEqual(code, 2)
            self.assertIn("nope", out)

    def test_index_unknown_collection(self):
        with temp_dir("cli4") as tmp:
            config = self._setup(tmp)
            code, out = run_cli(["index", "--collection", "nope", "--config",
                                 str(config), "--embedder", "hashing"])
            self.assertEqual(code, 2)
            self.assertIn("nope", out)

    def test_search_without_index_is_actionable(self):
        with temp_dir("cli5") as tmp:
            config = self._setup(tmp)
            code, out = run_cli(["search", "HashMap", "--config", str(config)])
            self.assertEqual(code, 2)
            self.assertIn("index", out.lower())

    def test_drop_by_path_keeps_other_collections(self):
        with temp_dir("cli6") as tmp:
            config = self._setup(tmp)
            run_cli(["ingest", "--config", str(config)])
            for name in ("tech", "profile"):
                run_cli(["index", "--collection", name, "--config", str(config),
                         "--embedder", "hashing"])
            tech_meta = (tmp / "data/index/index_meta.json").read_text(encoding="utf-8")

            code, out = run_cli(["drop", "--collection", "profile", "--path",
                                 "何毓-简历.pdf", "--yes", "--config", str(config)])
            self.assertEqual(code, 0, out)
            self.assertIn("删除", out)
            # 文件没了、块也没了
            self.assertFalse((tmp / "raw/profile/何毓-简历.pdf").exists())
            self.assertEqual(read_jsonl(tmp / "data/processed/chunks_profile.jsonl"), [])
            # profile 索引目录被清掉（集合已空）
            self.assertFalse((tmp / "data/index/profile").exists())
            # tech 的索引原封不动
            self.assertEqual(
                (tmp / "data/index/index_meta.json").read_text(encoding="utf-8"),
                tech_meta)
            self.assertTrue((tmp / "data/processed/chunks_tech.jsonl").exists())

    def test_drop_by_doc_id(self):
        with temp_dir("cli7") as tmp:
            config = self._setup(tmp)
            run_cli(["ingest", "--collection", "profile", "--config", str(config)])
            rows = read_jsonl(tmp / "data/processed/manifest_profile.jsonl")
            self.assertTrue(rows)
            doc_id = rows[0]["doc_id"]

            code, out = run_cli(["drop", "--collection", "profile", "--doc-id",
                                 doc_id, "--yes", "--config", str(config)])
            self.assertEqual(code, 0, out)
            self.assertFalse((tmp / "raw/profile/何毓-简历.pdf").exists())

    def test_drop_requires_explicit_collection(self):
        with temp_dir("cli8") as tmp:
            config = self._setup(tmp)
            # --collection 是 argparse 的 required=True，缺了直接退出 2
            with self.assertRaises(SystemExit) as ctx:
                run_cli(["drop", "--path", "x.pdf", "--yes", "--config", str(config)])
            self.assertEqual(ctx.exception.code, 2)

    def test_drop_rebuilds_index_when_docs_remain(self):
        """删掉一份、集合里还有别的文档时，重建索引而不是删目录。"""
        with temp_dir("cli10") as tmp:
            config = self._setup(tmp)
            write_pdf(
                tmp / "raw/profile/另一份.pdf", title="另一份", heading="另一份",
                paragraphs=[
                    "这是一份用于测试删除后重建索引的材料，内容长度要超过一百个字符，"
                    "否则会被质量指标标成低质量块，检索时默认排除，测起来就不直观了。"
                    "再补一句凑够长度，确保平均块字符数高于阈值。",
                ],
            )
            run_cli(["ingest", "--collection", "profile", "--config", str(config)])
            run_cli(["index", "--collection", "profile", "--config", str(config),
                     "--embedder", "hashing"])
            before = json.loads(
                (tmp / "data/index/profile/index_meta.json").read_text(encoding="utf-8"))

            code, out = run_cli(["drop", "--collection", "profile", "--path",
                                 "何毓-简历.pdf", "--yes", "--config", str(config)])
            self.assertEqual(code, 0, out)
            after = json.loads(
                (tmp / "data/index/profile/index_meta.json").read_text(encoding="utf-8"))
            self.assertTrue((tmp / "data/index/profile").exists())
            self.assertLess(after["n_chunks"], before["n_chunks"])
            self.assertGreater(after["n_chunks"], 0)

            rows = read_jsonl(tmp / "data/processed/chunks_profile.jsonl")
            self.assertTrue(rows)
            self.assertFalse(any("何毓" in row["rel_path"] for row in rows))
            self.assertTrue(any("另一份" in row["rel_path"] for row in rows))

    def test_drop_unknown_path(self):
        with temp_dir("cli9") as tmp:
            config = self._setup(tmp)
            run_cli(["ingest", "--collection", "profile", "--config", str(config)])
            code, out = run_cli(["drop", "--collection", "profile", "--path",
                                 "不存在的文件.pdf", "--yes", "--config", str(config)])
            self.assertEqual(code, 2)
            self.assertIn("找不到", out)


class TestManifestMatching(unittest.TestCase):
    def _manifest(self) -> list[dict]:
        return [
            {"doc_id": "a", "repo": "JavaGuide", "rel_path": "docs/hashmap.md"},
            {"doc_id": "b", "repo": "CS-Notes", "rel_path": "docs/hashmap.md"},
            {"doc_id": "c", "repo": "何毓-简历", "rel_path": "何毓-简历.pdf"},
        ]

    def test_match_by_repo_and_rel_path(self):
        from kb.cli import _match_manifest

        with temp_dir("match") as tmp:
            rel, candidates = _match_manifest(self._manifest(),
                                              "JavaGuide/docs/hashmap.md", tmp)
            self.assertEqual(rel, "JavaGuide/docs/hashmap.md")
            self.assertEqual(len(candidates), 1)

    def test_match_by_pdf_name(self):
        from kb.cli import _match_manifest

        with temp_dir("match2") as tmp:
            rel, _ = _match_manifest(self._manifest(), "何毓-简历.pdf", tmp)
            self.assertEqual(rel, "何毓-简历.pdf")

    def test_ambiguous_filename_lists_candidates(self):
        from kb.cli import _match_manifest

        with temp_dir("match3") as tmp:
            _, candidates = _match_manifest(self._manifest(), "hashmap.md", tmp)
            self.assertEqual(len(candidates), 2)

    def test_untracked_file_falls_back_to_raw_dir(self):
        from kb.cli import _match_manifest

        with temp_dir("match4") as tmp:
            (tmp / "profile").mkdir(parents=True, exist_ok=True)
            (tmp / "profile" / "new.pdf").write_bytes(b"%PDF-1.4")
            rel, _ = _match_manifest([], "profile/new.pdf", tmp)
            self.assertEqual(rel, "profile/new.pdf")

    def test_path_outside_raw_dir_rejected(self):
        from kb.cli import _match_manifest

        with temp_dir("match5") as tmp:
            rel, _ = _match_manifest([], "../outside.pdf", tmp)
            self.assertEqual(rel, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
