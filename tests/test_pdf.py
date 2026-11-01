"""PDF 解析测试：用 pymupdf 现造 PDF，不依赖任何真实文件。

运行：
    python tests\\test_pdf.py
    python -m unittest tests.test_pdf -v

每个用例对应一条真会出错的地方：

  · 阅读顺序        —— PDF 的抽取顺序取决于写入顺序，不是眼睛看到的顺序；
                        BOSS 直聘那份简历的姓名就跑到最后一页去了
  · 标题识别        —— 中位字号算错，13pt 的小标题会被当正文，面包屑直接塌成一层
  · 页眉页脚        —— 不剔除的话每页都多出一行「机密」，检索时会被反复召回
  · 短行合并        —— PDF 把一段拆成多行，不合并的话每句都成了独立段落
  · 单页不剔除      —— 一页里统计不出「重复」，把标题当页脚删掉是最糟的失败模式
  · 加密 / 零页 / 无文本层 / 坏文件 —— 必须返回原因而不是抛栈
"""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import kb  # noqa: E402,F401 - 导入即把 .pylibs 插进 sys.path
from kb.ingest.pdf import (  # noqa: E402
    SHORT_LINE_CHARS,
    merge_short_lines,
    page_locator,
    parse_pdf,
)

TMP_ROOT = ROOT / ".tmp"
PAGE_W, PAGE_H = 595, 842


@contextmanager
def temp_dir(prefix: str = "pdf"):
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"{prefix}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def pymupdf_module():
    import pymupdf  # noqa: PLC0415 - 由 kb/__init__.py 注入的 .pylibs 提供
    return pymupdf


# 三段正文都长于 20 字符（不会触发短行合并）、也长于 drop_below
LONG_1 = "这是第一段正文内容，长度超过二十个字符所以不会被当成短行合并掉。"
LONG_2 = "这是第二段正文内容，同样超过二十个字符，用来验证段落的先后顺序。"
LONG_3 = "这是第三段正文内容，位置在页面最下方，按坐标排序应该排在最后面。"
# 窄文本框里会被拆成 11/12/3 字的三行，合并后是下面这句话
WRAPPED = "这是一段需要自动换行的正文内容用来验证短行合并逻辑是否生效"


def make_pdf(path: Path, *, pages: int = 1, title: str = "测试报告",
             header: str | None = "测试报告 · 机密", footer: bool = True) -> Path:
    """造一份「标题 + 正文 + 页眉页脚」的 PDF。"""
    pymupdf = pymupdf_module()
    doc = pymupdf.open()
    if title:
        doc.set_metadata({"title": title})
    for i in range(1, pages + 1):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        if header:
            page.insert_text((72, 28), header, fontsize=9, fontname="china-s")
        page.insert_text((72, 180), f"第{i}章 总览", fontsize=18, fontname="china-s")
        page.insert_text((72, 220), "背景", fontsize=13, fontname="china-s")
        # 窄文本框：故意让 pymupdf 把一个段落拆成三行
        page.insert_textbox(pymupdf.Rect(72, 240, 210, 360), WRAPPED,
                            fontsize=10, fontname="china-s")
        for offset, text in ((380, LONG_1), (410, LONG_2), (440, LONG_3)):
            page.insert_text((72, offset), text, fontsize=10, fontname="china-s")
        if footer:
            page.insert_text((72, 812), f"第{i}页", fontsize=9, fontname="china-s")
    doc.save(str(path))
    doc.close()
    return path


class TestReadingOrder(unittest.TestCase):
    """按 (round(y0,1), x0) 排序恢复阅读顺序。"""

    def test_blocks_sorted_by_coordinates_not_insertion(self):
        pymupdf = pymupdf_module()
        with temp_dir("order") as tmp:
            path = tmp / "order.pdf"
            doc = pymupdf.open()
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            # 故意逆序写入：先写最下面，再写中间，最后写最上面
            page.insert_text((72, 320), LONG_3, fontsize=10, fontname="china-s")
            page.insert_text((72, 220), LONG_1, fontsize=10, fontname="china-s")
            page.insert_text((72, 270), LONG_2, fontsize=10, fontname="china-s")
            doc.save(str(path))
            doc.close()

            result = parse_pdf(path)
            self.assertEqual(result.reason, "")
            text = "\n".join(chunk.text for chunk in result.chunks)
            self.assertIn("第一段", text)
            self.assertLess(text.index("第一段"), text.index("第二段"))
            self.assertLess(text.index("第二段"), text.index("第三段"))

    def test_two_columns_sorted_left_to_right(self):
        """两栏版面：两栏的块 y 相同，按 x 从左到右排。

        注意：**同一行里的文字如果被 pymupdf 归到同一个块内**，块内顺序是
        写入顺序，按坐标排序救不回来（`insert_text` 在同一基线上的多次调用
        会被合并成一块）。已知限制，见模块注释。
        """
        pymupdf = pymupdf_module()
        with temp_dir("cols") as tmp:
            path = tmp / "cols.pdf"
            doc = pymupdf.open()
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            for i in range(3):
                page.insert_text((300, 200 + i * 14),
                                 f"右栏第{i + 1}行内容写得比较长一些，确保不会合并掉。",
                                 fontsize=10, fontname="china-s")
            for i in range(3):
                page.insert_text((72, 200 + i * 14),
                                 f"左栏第{i + 1}行内容写得比较长一些，确保不会合并掉。",
                                 fontsize=10, fontname="china-s")
            doc.save(str(path))
            doc.close()

            result = parse_pdf(path)
            text = "\n".join(chunk.text for chunk in result.chunks)
            self.assertLess(text.index("左栏第1行"), text.index("右栏第1行"))
            # 块内顺序保持：左栏三行按 1/2/3 排列
            self.assertLess(text.index("左栏第1行"), text.index("左栏第2行"))
            self.assertLess(text.index("左栏第2行"), text.index("左栏第3行"))


class TestHeading(unittest.TestCase):
    """字号 > 中位数 * 1.15 的行当标题，按字号从大到小映射 1~3 级。"""

    def test_breadcrumb_from_font_size(self):
        with temp_dir("heading") as tmp:
            path = make_pdf(tmp / "h.pdf")
            result = parse_pdf(path)
            crumbs = [chunk.breadcrumb for chunk in result.chunks]
            self.assertTrue(crumbs)
            self.assertEqual(crumbs[0], "测试报告 > 第1章 总览 > 背景")

    def test_heading_text_not_in_body(self):
        with temp_dir("heading2") as tmp:
            path = make_pdf(tmp / "h.pdf")
            result = parse_pdf(path)
            body = "\n".join(chunk.text for chunk in result.chunks)
            self.assertNotIn("第1章 总览", body)
            self.assertNotIn("背景", body)

    def test_median_not_dragged_up_by_headings(self):
        """标题块数量和正文块数量接近时，中位字号必须仍落在正文上。

        按块取中位数会在这种版面上翻车：一页里 18pt/13pt/10pt 各一块，
        中位数会落在中间那一档，于是小标题不再「比中位数大 1.15 倍」，
        标题识别整层失效。按字符数加权后中位数落在正文上。
        """
        body = ("正文的一句话要写得足够长，长到远超标题的字数，"
                "这样按字符加权的中位数才会落在正文字号上。")
        with temp_dir("heading3") as tmp:
            path = tmp / "many_headings.pdf"
            pymupdf = pymupdf_module()
            doc = pymupdf.open()
            doc.set_metadata({"title": "报告"})
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            for i, y in enumerate((180, 300, 420, 540)):
                page.insert_text((72, y), f"第{i + 1}节 小节标题", fontsize=13,
                                 fontname="china-s")
                page.insert_text((72, y + 30), f"{body}第{i + 1}节。", fontsize=10,
                                 fontname="china-s")
            doc.save(str(path))
            doc.close()

            result = parse_pdf(path, min_chars=1)  # 关掉「小相邻块合并」，只看标题
            crumbs = {chunk.breadcrumb for chunk in result.chunks}
            self.assertTrue(any("第1节 小节标题" in c for c in crumbs), crumbs)
            self.assertTrue(all(c.startswith("报告") for c in crumbs), crumbs)

    def test_no_headings_single_root(self):
        """全篇同一字号时不该凭空造出标题，面包屑就是文档标题。"""
        pymupdf = pymupdf_module()
        with temp_dir("noheading") as tmp:
            path = tmp / "flat.pdf"
            doc = pymupdf.open()
            doc.set_metadata({"title": "扁平文档"})
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            page.insert_text((72, 200), LONG_1, fontsize=10, fontname="china-s")
            page.insert_text((72, 230), LONG_2, fontsize=10, fontname="china-s")
            doc.save(str(path))
            doc.close()

            result = parse_pdf(path)
            self.assertTrue(result.chunks)
            self.assertEqual({c.breadcrumb for c in result.chunks}, {"扁平文档"})


class TestHeaderFooter(unittest.TestCase):
    """出现率 > 60% 且位于页面上下 15% 区域的行，从正文剔除。"""

    def test_repeated_header_and_footer_removed(self):
        with temp_dir("margins") as tmp:
            path = make_pdf(tmp / "m.pdf", pages=3)
            result = parse_pdf(path)
            body = "\n".join(chunk.text for chunk in result.chunks)
            self.assertNotIn("测试报告 · 机密", body)
            self.assertNotIn("第1页", body)
            self.assertNotIn("第2页", body)
            # 页眉页脚被剔除后，正文三段都还在
            self.assertIn("第一段", body)
            self.assertIn("第三段", body)

    def test_page_numbers_with_different_digits_removed(self):
        """页码每页都不一样，靠数字归一（第 1 页 -> 第 # 页）才能识别。"""
        with temp_dir("margins2") as tmp:
            path = make_pdf(tmp / "m.pdf", pages=3)
            result = parse_pdf(path)
            for chunk in result.chunks:
                self.assertNotRegex(chunk.text, r"第\d+页")

    def test_single_page_skips_margin_removal(self):
        """单页 PDF 跳过页眉页脚剔除：一页里统计不出「重复」。

        用 `drop_below=1` 跑，避免这一行被「太短」的规则丢掉 —— 本用例只关心
        「有没有被当成页眉」，不关心块长过滤。
        """
        with temp_dir("margins3") as tmp:
            path = make_pdf(tmp / "one.pdf", pages=1)
            result = parse_pdf(path, min_chars=1, drop_below=1)
            body = "\n".join(chunk.text for chunk in result.chunks)
            self.assertIn("测试报告 · 机密", body)

    def test_unique_top_line_kept(self):
        """只在一页顶部出现一次的行不能被当成页眉删掉。"""
        pymupdf = pymupdf_module()
        with temp_dir("margins4") as tmp:
            path = tmp / "unique.pdf"
            doc = pymupdf.open()
            doc.set_metadata({"title": "文档"})
            for i in (1, 2, 3):
                page = doc.new_page(width=PAGE_W, height=PAGE_H)
                page.insert_text((72, 28), "固定页眉", fontsize=9, fontname="china-s")
                if i == 1:
                    page.insert_text((72, 60), "仅首页出现的副标题", fontsize=10,
                                     fontname="china-s")
                page.insert_text((72, 200), f"{LONG_1}第{i}页正文", fontsize=10,
                                 fontname="china-s")
                page.insert_text((72, 812), f"第{i}页", fontsize=9, fontname="china-s")
            doc.save(str(path))
            doc.close()

            result = parse_pdf(path)
            body = "\n".join(chunk.text for chunk in result.chunks)
            self.assertIn("仅首页出现的副标题", body)
            self.assertNotIn("固定页眉", body)


class TestShortLineMerge(unittest.TestCase):
    """相邻且都以中文/字母结尾、长度 < 20 的行合并成一段。"""

    def test_helper_merges_wrapped_paragraph(self):
        lines = ["这是一段需要自动换行的正文", "内容用来验证短行合并逻辑是", "否生效"]
        merged = merge_short_lines(lines)
        self.assertEqual(merged, [WRAPPED])

    def test_helper_keeps_long_lines_apart(self):
        long_line = "这一行的长度已经超过二十个字符了所以不该和下一行合并在一起"
        merged = merge_short_lines([long_line, "下一行"])
        self.assertEqual(len(merged), 2)

    def test_helper_does_not_merge_sentence_ending_with_punctuation(self):
        """以标点结尾的行本身就是一个完整句子，不往后粘。"""
        merged = merge_short_lines(["第一句话。", "第二句话。"])
        self.assertEqual(merged, ["第一句话。", "第二句话。"])

    def test_helper_joins_english_words_with_space(self):
        merged = merge_short_lines(["local knowledge", "base system"])
        self.assertEqual(merged, ["local knowledge base system"])

    def test_end_to_end_wrapped_block_merged(self):
        with temp_dir("merge") as tmp:
            path = make_pdf(tmp / "m.pdf")
            result = parse_pdf(path)
            body = "\n".join(chunk.text for chunk in result.chunks)
            self.assertIn(WRAPPED, body)

    def test_separate_paragraphs_not_glued(self):
        """不同块之间的短行不能合并（简历里「何毓 / 男 / 电话」是三行三块）。

        用 `drop_below=1` 跑，否则两三个字的块会被块长过滤直接丢掉，测不到合并。
        """
        pymupdf = pymupdf_module()
        with temp_dir("merge2") as tmp:
            path = tmp / "fields.pdf"
            doc = pymupdf.open()
            doc.set_metadata({"title": "简历"})
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            for y, text in ((200, "何毓"), (230, "男"), (260, "求职意向")):
                page.insert_text((72, y), text, fontsize=10, fontname="china-s")
            doc.save(str(path))
            doc.close()

            result = parse_pdf(path, min_chars=1, drop_below=1)
            body = "\n".join(chunk.text for chunk in result.chunks)
            self.assertNotIn("何毓男", body)
            self.assertIn("何毓", body)
            self.assertIn("男", body)


class TestLocator(unittest.TestCase):
    def test_page_locator(self):
        self.assertEqual(page_locator(3, 3), "P3")
        self.assertEqual(page_locator(3, 4), "P3-4")
        self.assertEqual(page_locator(0, 0), "")

    def test_cross_page_chunk_uses_range(self):
        """同一个章节跨页时标成 P1-2：第 2 页没有新标题，正文接着上一页。"""
        pymupdf = pymupdf_module()
        with temp_dir("cross") as tmp:
            path = tmp / "c.pdf"
            doc = pymupdf.open()
            doc.set_metadata({"title": "跨页文档"})
            first = doc.new_page(width=PAGE_W, height=PAGE_H)
            first.insert_text((72, 180), "第一章 总览", fontsize=18, fontname="china-s")
            first.insert_text((72, 240), LONG_1, fontsize=10, fontname="china-s")
            second = doc.new_page(width=PAGE_W, height=PAGE_H)
            second.insert_text((72, 240), LONG_2, fontsize=10, fontname="china-s")
            doc.save(str(path))
            doc.close()

            result = parse_pdf(path)
            locators = {page_locator(c.line_start, c.line_end) for c in result.chunks}
            self.assertIn("P1-2", locators)


class TestEdgeCases(unittest.TestCase):
    """加密 / 零页 / 纯图片 / 坏文件：返回跳过原因，不抛栈。"""

    def test_encrypted_pdf(self):
        pymupdf = pymupdf_module()
        with temp_dir("enc") as tmp:
            path = tmp / "enc.pdf"
            doc = pymupdf.open()
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            page.insert_text((72, 200), LONG_1, fontsize=10, fontname="china-s")
            doc.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256,
                     owner_pw="owner", user_pw="user")
            doc.close()

            result = parse_pdf(path)
            self.assertTrue(result.reason.startswith("pdf_encrypted"), result.reason)
            self.assertEqual(result.chunks, [])

    def test_zero_page_pdf(self):
        with temp_dir("zero") as tmp:
            path = tmp / "zero.pdf"
            # pymupdf 不让你保存零页文档，手写一个最小的 /Count 0 文件
            path.write_bytes(
                b"%PDF-1.4\n"
                b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
                b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
                b"trailer\n<< /Size 3 /Root 1 0 R >>\n%%EOF\n"
            )
            result = parse_pdf(path)
            self.assertTrue(result.reason.startswith("pdf_zero_pages"), result.reason)
            self.assertEqual(result.chunks, [])

    def test_image_only_pdf(self):
        pymupdf = pymupdf_module()
        with temp_dir("img") as tmp:
            path = tmp / "img.pdf"
            doc = pymupdf.open()
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            page.draw_rect(pymupdf.Rect(100, 100, 400, 400),
                           color=(0, 0, 0), fill=(0.6, 0.6, 0.6))
            doc.save(str(path))
            doc.close()

            result = parse_pdf(path)
            self.assertTrue(result.reason.startswith("pdf_no_text_layer"),
                            result.reason)
            self.assertEqual(result.chunks, [])

    def test_corrupt_file(self):
        with temp_dir("bad") as tmp:
            path = tmp / "bad.pdf"
            path.write_bytes(b"not a pdf at all, just bytes")
            result = parse_pdf(path)
            self.assertTrue(result.reason.startswith("pdf_unreadable"), result.reason)
            self.assertEqual(result.chunks, [])

    def test_pdf_without_chunks_after_filtering(self):
        """有文本层但每段都短于 drop_below 时给 pdf_no_chunks，而不是空结果。"""
        pymupdf = pymupdf_module()
        with temp_dir("tiny") as tmp:
            path = tmp / "tiny.pdf"
            doc = pymupdf.open()
            doc.set_metadata({"title": "碎"})
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            page.insert_text((72, 200), "短", fontsize=10, fontname="china-s")
            doc.save(str(path))
            doc.close()

            result = parse_pdf(path)
            self.assertTrue(result.reason.startswith("pdf_no_chunks"), result.reason)

    def test_parse_is_deterministic(self):
        with temp_dir("det") as tmp:
            path = make_pdf(tmp / "d.pdf", pages=2)
            first = parse_pdf(path)
            second = parse_pdf(path)
            self.assertEqual([c.text for c in first.chunks],
                             [c.text for c in second.chunks])
            self.assertEqual([c.breadcrumb for c in first.chunks],
                             [c.breadcrumb for c in second.chunks])


class TestPymupdfAvailability(unittest.TestCase):
    def test_pymupdf_importable_via_kb(self):
        """kb 包把 .pylibs 注入 sys.path，pymupdf 在任何模块里都能 import。"""
        try:
            pymupdf = pymupdf_module()
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"本机没有 PyMuPDF：{exc}")
        self.assertTrue(callable(pymupdf.open))
        self.assertIn(".pylibs", str(Path(pymupdf.__file__).parent).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
