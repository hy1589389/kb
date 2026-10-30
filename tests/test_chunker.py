"""切块器回归测试（仅用标准库，无需 pytest）。

运行：
    python tests/test_chunker.py
    python -m unittest discover -s tests -v

这里每个用例都对应一个真实踩过的坑，不是凑数的：
  · 行数错位        —— 清洗压缩空行导致标记数组与文本错位，标题被当正文、整节丢失
  · 静默丢内容      —— drop_below 把「一句话小节」直接删了
  · 代码块被切断    —— 按字符硬切把一个函数劈成两半
  · breadcrumb 撒谎 —— 合并多节后却沿用第一节的标题
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb.ingest.clean import clean_text  # noqa: E402
from kb.ingest.chunker import chunk_document, parse_sections  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
_RE_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")


def _fixture_files() -> list[Path]:
    out = []
    for p in sorted(FIXTURES.rglob("*.md")):
        if "assets" in p.parts or p.name.lower().startswith(("license", "changelog")):
            continue  # 这两个本来就该被跳过
        out.append(p)
    return out


class TestStructure(unittest.TestCase):
    def test_deep_headings_become_sections(self):
        """### 级标题必须成为独立章节，不能混进正文（行数错位 bug 的直接回归）。"""
        raw = (FIXTURES / "JavaGuide" / "docs" / "hashmap.md").read_text(encoding="utf-8")
        _, sections = parse_sections(raw)
        paths = [s.path for s in sections]

        self.assertIn(("HashMap 底层原理", "扩容机制", "为什么是 0.75"), paths)
        self.assertIn(("HashMap 底层原理", "扩容机制", "为什么是 8"), paths)

    def test_heading_inside_code_fence_is_not_a_section(self):
        """代码块里以 # 开头的行，绝不能当成标题。"""
        md = (
            "# 标题\n\n"
            "```python\n"
            "# 这是注释，不是标题\n"
            "x = 1\n"
            "```\n\n"
            "## 真正的二级标题\n\n"
            "正文内容在这里，足够长一些以免被合并掉。\n"
        )
        _, sections = parse_sections(md)
        paths = [s.path for s in sections]
        self.assertEqual(len(sections), 2, f"应只有 2 个章节，实际 {paths}")
        self.assertIn(("标题", "真正的二级标题"), paths)
        # 注释行应当留在第一个章节的正文里
        self.assertIn("# 这是注释，不是标题", sections[0].text)

    def test_front_matter_and_comments_removed_but_lines_preserved(self):
        """front matter / HTML 注释要被清掉，但行号必须与原文对齐。"""
        md = "---\ntitle: x\ntags: [a, b]\n---\n\n# 标题\n\n正文。\n"
        lines, _ = clean_text(md)
        # 原文 8 行，清洗后仍应是 8 行
        self.assertEqual(len(lines), md.count("\n") + 1)
        self.assertEqual(lines[5].strip(), "# 标题")

    def test_badges_and_images_removed(self):
        body = "正文内容在这里，写得长一些以免整个章节被当成噪声丢弃。" * 6
        md = (
            "# 标题\n\n"
            "[![Build](https://img.shields.io/badge/build-passing.svg)](https://x.com)\n\n"
            "![图](assets/a.png)\n\n"
            f"{body}\n"
        )
        _, chunks = chunk_document(md)
        blob = "\n".join(c.text for c in chunks)
        self.assertNotIn("shields.io", blob)
        self.assertNotIn("assets/a.png", blob)
        self.assertIn("正文内容", blob)


class TestChunking(unittest.TestCase):
    def test_long_section_is_split_into_parts(self):
        # 注意：这里刻意写成「没有空行的大段落」，覆盖「单行就超长」这一类真实脏数据。
        md = "# 标题\n\n## 长节\n\n" + ("这是一段用于测试的很长的正文内容。" * 200 + "\n\n") * 2
        max_chars, overlap = 600, 50
        _, chunks = chunk_document(md, max_chars=max_chars, overlap=overlap)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(
                len(c.text), max_chars + overlap + 10, f"切分后仍超长: {len(c.text)}"
            )

    def test_split_never_breaks_code_fence(self):
        """超长代码块宁可整块保留（超长），也不能被切成半截。"""
        code = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(300)) + "\n```"
        md = f"# 标题\n\n## 大代码块\n\n{code}\n"
        _, chunks = chunk_document(md, max_chars=500, overlap=0)
        joined = "\n".join(c.text for c in chunks)
        for c in chunks:
            self.assertEqual(c.text.count("```") % 2, 0, f"块内围栏不成对：\n{c.text[:120]}")
        self.assertIn("line_0 = 0", joined)
        self.assertIn("line_299 = 299", joined)

    def test_small_siblings_merge_with_honest_breadcrumb(self):
        md = "# 根\n\n## 父\n\n### 甲\n\n短。\n\n### 乙\n\n也短。\n"
        # drop_below=1：本用例只验证「合并」逻辑，不掺进「噪声丢弃」逻辑
        _, chunks = chunk_document(md, min_chars=200, drop_below=1)
        self.assertEqual(len(chunks), 1)
        # 合并后 breadcrumb 应是公共父节点，而不是「甲」
        self.assertEqual(chunks[0].path, ("根", "父"))
        self.assertIn("短。", chunks[0].text)
        self.assertIn("也短。", chunks[0].text)

    def test_base64_blob_is_compressed(self):
        """base64 图片不能把 chunk 撑爆 —— 实测 self-llm 里有 56KB 的 base64 示例。"""
        blob = "A" * 5000
        md = (
            "# 标题\n\n## 图片示例\n\n"
            f'```python\npayload = "data:image/png;base64,{blob}"\n```\n\n'
            "这就是全部内容了，写得长一点以免整个章节被当成噪声丢弃。" * 4 + "\n"
        )
        _, chunks = chunk_document(md)
        joined = "\n".join(c.text for c in chunks)
        self.assertNotIn(blob, joined, "base64 长串没有被压缩")
        self.assertIn("已省略", joined)
        self.assertLess(max(len(c.text) for c in chunks), 2000)

    def test_oversized_code_block_is_split_and_stays_valid(self):
        """超长代码块要在空行处拆开，且拆完每块围栏都要成对。"""
        body = "\n\n".join(
            "\n".join(f"v{i}_{j} = {j}" for j in range(40)) for i in range(12)
        )
        md = f"# 标题\n\n## 超大代码块\n\n```python\n{body}\n```\n"
        _, chunks = chunk_document(md, max_chars=800, overlap=0, code_block_max_chars=1000)

        self.assertGreater(len(chunks), 1, "超大代码块没有被拆开")
        for c in chunks:
            self.assertEqual(c.text.count("```") % 2, 0, f"围栏不成对：\n{c.text[:120]}")

        blob = "\n".join(c.text for c in chunks)
        self.assertIn("v0_0 = 0", blob)
        self.assertIn("v11_39 = 39", blob)

    def test_giant_single_line_inside_code_block_is_bounded(self):
        """代码块里单行就 7000 字符（真实案例：self-llm 的推理结果 repr dump）。

        该块整块只有 3 行，按行切完全切不动，必须退到字符级切分。
        """
        long_line = "ChatCompletion(" + ", ".join(f"field_{i}='value_{i}'" for i in range(400)) + ")"
        md = f"# 标题\n\n## 输出示例\n\n```text\n{long_line}\n```\n"
        _, chunks = chunk_document(md, max_chars=800, overlap=0, code_block_max_chars=1000)

        self.assertGreater(len(chunks), 1, "单行超长代码块没有被切开")
        for c in chunks:
            # 代码块允许放宽到 code_block_max_chars（刻意取舍），再加围栏开销
            self.assertLessEqual(len(c.text), 1000 + 40, f"仍有超长块: {len(c.text)}")
            self.assertEqual(c.text.count("```") % 2, 0, "围栏不成对")

    def test_adjacent_code_blocks_split_apart(self):
        """三个代码块紧挨着、中间没有空行，也必须各自成块。

        真实案例：leetcode-master 的「其他语言版本 > Java」一节，
        闭合围栏的下一行直接就是下一个开启围栏，导致 5000+ 字巨块。
        """
        block = "```Java\n" + "\n".join(f"line_{i}();" for i in range(60)) + "\n```"
        md = f"# 标题\n\n## 其他语言版本 > Java\n\n{block}\n{block}\n{block}\n"
        _, chunks = chunk_document(md, max_chars=400, overlap=0, code_block_max_chars=1000)

        self.assertEqual(len(chunks), 3, f"紧邻代码块没有被分开，得到 {len(chunks)} 块")
        for c in chunks:
            self.assertLessEqual(len(c.text), 1000 + 40)
            self.assertEqual(c.text.count("```") % 2, 0, "围栏不成对")

    def test_tiny_tail_is_absorbed_not_dropped(self):
        """一句话小节必须被吸收进相邻块，而不是被 drop_below 静默删除。"""
        md = (
            "# 根\n\n"
            "## 正常节\n\n" + "这是一段足够长的正文，超过 min_chars 阈值。" * 10 + "\n\n"
            "## 极短节\n\n就一句话。\n"
        )
        _, chunks = chunk_document(md, min_chars=120, drop_below=40)
        blob = "\n".join(c.text for c in chunks)
        self.assertIn("就一句话", blob, "极短章节被静默丢弃了")


class TestNoContentLoss(unittest.TestCase):
    """最重要的不变量：清洗后的每一行有内容的原文，都必须在某个 chunk 里出现。"""

    def test_every_content_line_survives(self):
        checked = 0
        for path in _fixture_files():
            raw = path.read_text(encoding="utf-8")
            lines, _ = clean_text(raw)
            _, chunks = chunk_document(raw)
            blob = "\n".join(c.text for c in chunks)

            for ln in lines:
                s = ln.strip()
                if len(s) < 10:
                    continue
                if _RE_HEADING.match(s):
                    continue  # 标题以 breadcrumb 形式保留，不在正文里
                self.assertIn(s, blob, f"{path.name} 丢失内容: {s[:60]}")
            checked += 1

        self.assertGreater(checked, 0, "没有找到任何测试语料")

    def test_line_numbers_point_at_real_content(self):
        """chunk 的行号区间内，必须真的能找到它的首行内容。"""
        for path in _fixture_files():
            raw = path.read_text(encoding="utf-8")
            lines, _ = clean_text(raw)
            _, chunks = chunk_document(raw)
            for c in chunks:
                first = next((x.strip() for x in c.text.split("\n") if x.strip()), "")
                if not first:
                    continue
                window = "\n".join(lines[max(0, c.line_start - 1): c.line_end])
                self.assertIn(
                    first,
                    window,
                    f"{path.name} 行号 {c.line_start}-{c.line_end} 对不上内容: {first[:50]}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
