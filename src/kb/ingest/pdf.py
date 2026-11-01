"""PDF 解析：坐标排序恢复阅读顺序 → 字号识别标题 → 剔除页眉页脚 → 短行合并。

## 为什么不能直接 `page.get_text()`

BOSS 直聘导出的简历，`get_text()` 直接取出来的顺序是乱的：姓名「何毓」跑到
了最后一页最后一行。原因是 PDF 里没有「段落」这个概念，只有「在某个坐标画
了某个字形」；抽取顺序取决于写入顺序，跟人眼读的顺序无关。

`page.get_text("blocks")` 返回 ``(x0, y0, x1, y1, text, block_no, block_type)``。
按 ``(round(y0, 1), x0)`` 排序就能恢复阅读顺序：先从上到下，同一行内从左到右。
拿那份简历验证过，排序后变成「何毓 / 男 / 15893890714 / 求职意向 / 个人优势 / ...」，
和眼睛看到的一致。`round(y0, 1)` 是必要的：同一行不同块的 y 会差 0.001 级别
的浮点噪声，不取整会在两列之间来回跳。

已知限制：排序只作用于**块之间**。同一基线上的两段文字如果被 pymupdf 归进
同一个块（典型的「姓名 / 性别」并排表单，或 `insert_text` 在同一 y 上的多次
调用），块内顺序仍是写入顺序，这里救不回来 —— 想修只能按 span 的 x 坐标重排，
那要放弃块级结构，代价比收益大。两栏排版（各栏多行）不受影响，仍是两个块。

## 四步处理

1. **阅读顺序**：只留 `block_type == 0`（文本）的块，按 `(round(y0, 1), x0)` 排序。
2. **标题识别**：用 `get_text("dict")` 拿 span 的 `size`，算出正文的**中位字号**；
   比中位数大 1.15 倍以上的行当标题，按字号从大到小映射成 1~3 级，拼成 breadcrumb。
3. **页眉页脚**：统计所有页里出现率 > 60%、且落在页面顶部/底部 15% 区域内的
   文本，从正文剔除。单页 PDF 跳过这一步（一页里没法统计「重复」）。
   数字会被归一成 ``#`` 再比对，所以「第 1 页 / 第 2 页」这种页码也能识别。
4. **短行合并**：PDF 常把一个段落拆成多行，相邻且都以中文/字母结尾、长度 < 20
   的行合并回一段。只在同一个块内部合并 —— 跨块合并会把「姓名 / 性别」这类
   紧挨着的表单字段粘成一句（实测那份简历就是这样）。

切分本身不另写一套：把「标题 + 段落」装成 `chunker.Section`（`line_start` /
`line_end` 填页码），交给 `chunker.chunk_sections`，参数全部来自 `config.toml`
的 `[chunk]` 段。

## 失败情形

加密、零页、纯图片（无文本层）、文件损坏都**返回跳过原因**而不是抛栈：
一份上传错的文件不该让整次 ingest 崩掉。调用方拿到 `PdfParseResult.reason`
非空就跳过该文件，打印原因。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import Chunk, Section, chunk_sections

__all__ = [
    "PDF_SUFFIXES",
    "PdfBlock",
    "PdfPage",
    "PdfParseResult",
    "merge_short_lines",
    "page_locator",
    "parse_pdf",
]

PDF_SUFFIXES = (".pdf",)

# ---- 可调常量：数值都来自实测，改动前先看 tests/test_pdf.py 的用例 ----
HEADING_SIZE_RATIO = 1.15   # 字号 > 中位数 * 1.15 视为标题
HEADING_MAX_CHARS = 80      # 超过这个长度的一律当正文（大字号的长段落不是标题）
HEADING_MAX_LINES = 1       # 标题块最多几行
MARGIN_ZONE_RATIO = 0.15    # 页面顶部/底部各 15% 是页眉页脚区
MARGIN_REPEAT_RATE = 0.60   # 在 > 60% 的页里重复出现才算页眉页脚
SHORT_LINE_CHARS = 20       # 短行合并的长度上限
TITLE_MAX_CHARS = 40        # 无元数据标题时，首页最大字号那行要短于这个才当标题

_CJK_OR_LETTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z]$")
_ASCII_ALNUM = re.compile(r"[0-9A-Za-z]$")
_PAGE_NUMBER_RE = re.compile(r"\d+")


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class PdfBlock:
    """一个文本块（`get_text("blocks")` 的一项）及其字号、所在区域。"""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float = 0.0   # 块内最大 span 字号
    in_margin: bool = False  # 落在页面顶部/底部 15% 区域内


@dataclass
class PdfPage:
    number: int  # 1-based
    height: float
    blocks: list[PdfBlock] = field(default_factory=list)


@dataclass
class PdfParseResult:
    """一次 PDF 解析的结果。

    `reason` 非空表示这个文件被跳过（加密 / 零页 / 无文本层 / 读不出来），
    此时 `chunks` 为空、`title` 可能是文件名。调用方按跳过处理，不要当异常。
    """

    title: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    n_pages: int = 0
    text_chars: int = 0
    reason: str = ""
    notes: list[str] = field(default_factory=list)


class PdfUnavailableError(RuntimeError):
    """PyMuPDF 没装上时抛这个（带可操作的提示，不是裸 ImportError）。"""


# --------------------------------------------------------------------------- #
# 依赖
# --------------------------------------------------------------------------- #
def _load_pymupdf():
    """延迟导入 pymupdf。

    `import fitz` 是旧名字，2.x 里已废弃且打 FutureWarning；统一用 `pymupdf`。
    `kb/__init__.py` 已经把 `.pylibs` 目录加进 sys.path，正常环境这里不会失败。
    """
    try:
        import pymupdf  # noqa: PLC0415 - 故意延迟，import 本模块不该拉起 25 MB 的扩展
    except ImportError as exc:  # pragma: no cover - 正常环境不会走到
        raise PdfUnavailableError(
            "解析 PDF 需要 PyMuPDF（import pymupdf），当前环境导入失败。\n"
            "本机把 wheel 解压在 G:\\kb\\.pylibs；换机器时用 KB_PYLIBS 环境变量"
            "指向解压目录即可。"
        ) from exc
    return pymupdf


# --------------------------------------------------------------------------- #
# 单个块的辅助
# --------------------------------------------------------------------------- #
def _norm_key(text: str) -> str:
    """页眉页脚比对用的归一化键：压空白 + 数字归一。

    「第 1 页」「第 2 页」只有数字不同，归一成「第 # 页」后才能在多页之间
    统计出现率 —— 页码是最常见的页脚，不处理它就白做这一步了。
    """
    flat = " ".join(str(text or "").split())
    return _PAGE_NUMBER_RE.sub("#", flat)


def _ends_with_text(text: str) -> bool:
    return bool(text) and _CJK_OR_LETTER.search(text[-1]) is not None


def _needs_space(prev: str, cur: str) -> bool:
    """英文断行合并时要补空格，中文不补。"""
    return bool(_ASCII_ALNUM.search(prev[-1])) and bool(cur) and cur[0].isalnum() \
        and ord(cur[0]) < 0x2E80


def merge_short_lines(lines: list[str], max_chars: int = SHORT_LINE_CHARS) -> list[str]:
    """把被版面拆开的短行合并回段落。

    条件（三条同时满足）：相邻、两行**原始长度**都短于 `max_chars`、都以中文
    或字母结尾。以标点结尾的行多半本来就是一个短句的结束，不合并。

    注意「短」看的是原始行，不是合并后的累积结果：一段 12 字 + 13 字 + 3 字
    的三行正文，合并到第二行就有 25 字了，如果拿累积长度去比，第三行永远
    接不上，段落会被切成「前半段 + 一个孤零零的尾巴」。

    只在同一个块内部合并：块与块之间通常是不同的版面区域，跨块合并会把
    简历里紧挨着的「何毓」「男」「15893890714」粘成一句 —— 那正是这份简历
    上真实出现的排版。
    """
    out: list[str] = []
    prev_short = False
    for raw in lines:
        line = (raw or "").strip()
        if not line:
            continue
        short = len(line) < max_chars
        if out and short and prev_short \
                and _ends_with_text(out[-1]) and _ends_with_text(line):
            out[-1] = out[-1] + (" " if _needs_space(out[-1], line) else "") + line
        else:
            out.append(line)
        prev_short = short
    return out


def page_locator(start: int, end: int) -> str:
    """页码 → locator：`P3` / 跨页 `P3-4`。"""
    start = int(start or 0)
    end = int(end or 0)
    if start <= 0:
        return ""
    if end and end != start:
        return f"P{start}-{end}"
    return f"P{start}"


# --------------------------------------------------------------------------- #
# 读页面
# --------------------------------------------------------------------------- #
def _line_sizes(page) -> list[tuple[tuple[float, float, float, float], float]]:
    """`(line_bbox, 该行最大 span 字号)`。字号只能从 dict 模式拿到。"""
    out: list[tuple[tuple[float, float, float, float], float]] = []
    try:
        data = page.get_text("dict")
    except (RuntimeError, ValueError):  # 坏页面：当作没有字号信息
        return out
    for block in data.get("blocks", []):
        if int(block.get("type", 0) or 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans") or []
            text = "".join(str(span.get("text", "")) for span in spans)
            if not text.strip():
                continue
            size = max((float(span.get("size") or 0.0) for span in spans), default=0.0)
            bbox = line.get("bbox") or block.get("bbox") or (0, 0, 0, 0)
            out.append((tuple(float(v) for v in bbox), size))
    return out


def _size_for_bbox(lines, bbox: tuple[float, float, float, float]) -> float:
    """块内所有落在其 bbox 里的行的最大字号。

    `get_text("blocks")` 和 `get_text("dict")` 的块划分是同一份内部结构，
    但两边的 bbox 不保证逐位相等，所以按「行的中心点落在块内」匹配，
    而不是靠列表下标对齐。
    """
    if not lines:
        return 0.0
    x0, y0, x1, y1 = bbox
    best = 0.0
    for (lx0, ly0, lx1, ly1), size in lines:
        cx = (lx0 + lx1) / 2
        cy = (ly0 + ly1) / 2
        if x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1:
            best = max(best, size)
    return round(best, 1)


def _read_page(page, number: int) -> PdfPage:
    """读一页：过滤图片块、按坐标排序、补上字号和所在区域。"""
    height = float(page.rect.height) or 1.0
    lines = _line_sizes(page)

    raw_blocks = []
    for block in page.get_text("blocks"):
        if len(block) >= 7 and int(block[6] or 0) != 0:
            continue  # block_type 1 = 图片
        text = str(block[4]) if len(block) > 4 else ""
        if not text.strip():
            continue
        raw_blocks.append(block)

    # 阅读顺序：先 y 后 x。y 取整到 0.1 是为了吸收同一行里的浮点噪声
    raw_blocks.sort(key=lambda b: (round(float(b[1]), 1), float(b[0])))

    out: list[PdfBlock] = []
    for block in raw_blocks:
        x0, y0, x1, y1 = (float(block[0]), float(block[1]),
                          float(block[2]), float(block[3]))
        out.append(
            PdfBlock(
                x0=x0, y0=y0, x1=x1, y1=y1,
                text=str(block[4]),
                size=_size_for_bbox(lines, (x0, y0, x1, y1)),
                # 整块都在顶部/底部 15% 带里才算页眉页脚候选
                in_margin=(y1 <= height * MARGIN_ZONE_RATIO
                           or y0 >= height * (1.0 - MARGIN_ZONE_RATIO)),
            )
        )
    return PdfPage(number=number, height=height, blocks=out)


# --------------------------------------------------------------------------- #
# 页眉页脚
# --------------------------------------------------------------------------- #
def _margin_keys(pages: list[PdfPage]) -> set[str]:
    """出现率 > 60% 且位于页面上下边缘的文本集合。

    单页文档直接返回空集：一页里的「重复」没有意义，把标题当页脚删掉
    是最糟的失败模式。
    """
    if len(pages) < 2:
        return set()

    counts: dict[str, int] = {}
    for page in pages:
        seen: set[str] = set()
        for block in page.blocks:
            if not block.in_margin:
                continue
            key = _norm_key(block.text)
            if not key or key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1

    threshold = MARGIN_REPEAT_RATE * len(pages)
    return {key for key, count in counts.items() if count > threshold}


# --------------------------------------------------------------------------- #
# 标题
# --------------------------------------------------------------------------- #
def _is_heading(block: PdfBlock, median: float) -> bool:
    if median <= 0 or block.size <= 0:
        return False
    if block.size <= median * HEADING_SIZE_RATIO:
        return False
    if len(" ".join(block.text.split())) > HEADING_MAX_CHARS:
        return False
    if block.text.count("\n") > HEADING_MAX_LINES:
        return False
    return True


def _heading_levels(pages: list[PdfPage], skip: set[str],
                    median: float) -> dict[float, int]:
    """字号 → 标题层级（1..3）。字号最大的那档是 1 级。"""
    sizes = sorted(
        {
            round(block.size, 1)
            for page in pages
            for block in page.blocks
            if _norm_key(block.text) not in skip and _is_heading(block, median)
        },
        reverse=True,
    )
    return {size: min(index + 1, 3) for index, size in enumerate(sizes)}


def _median_size(pages: list[PdfPage], skip: set[str]) -> float:
    """正文的中位字号，**按字符数加权**。

    直接对块取中位数会翻车：一页上标题块和正文块数量差不多时（简历、封面、
    每页都有小标题的论文版面），中位数会被抬到标题字号上，于是 13pt 的小标题
    不再「比中位数大 1.15 倍」，全部退化成正文字号，标题识别直接失效。
    正文字数远多于标题字数，按字数加权后中位数就落在正文上。
    """
    weights: dict[float, int] = {}
    for page in pages:
        for block in page.blocks:
            if block.size <= 0 or _norm_key(block.text) in skip:
                continue
            size = round(block.size, 1)
            weights[size] = weights.get(size, 0) + max(1, len(block.text))

    total = sum(weights.values())
    if not total:
        return 0.0
    half = total / 2
    seen = 0
    for size in sorted(weights):
        seen += weights[size]
        if seen >= half:
            return float(size)
    return float(max(weights))


def _document_title(doc, path: Path, pages: list[PdfPage], median: float,
                    skip: set[str]) -> tuple[str, set[tuple[int, int]]]:
    """文档标题：优先 PDF 元数据，其次首页最大字号的那一行，最后文件名。

    返回 `(title, consumed)`；`consumed` 是被当作标题消耗掉的块，后面不再
    作为章节标题重复出现。
    """
    try:
        meta = str((doc.metadata or {}).get("title") or "").strip()
    except (RuntimeError, ValueError):  # 元数据坏了不影响解析
        meta = ""
    if meta:
        return meta, set()

    if pages:
        best: tuple[float, int, str] | None = None
        for index, block in enumerate(pages[0].blocks):
            if _norm_key(block.text) in skip or not _is_heading(block, median):
                continue
            flat = " ".join(block.text.split())
            if not flat or len(flat) > TITLE_MAX_CHARS:
                continue
            if best is None or block.size > best[0]:
                best = (block.size, index, flat)
        if best is not None:
            return best[2], {(1, best[1])}

    return path.stem, set()


# --------------------------------------------------------------------------- #
# 组装 Section
# --------------------------------------------------------------------------- #
def _build_sections(pages: list[PdfPage], skip: set[str], levels: dict[float, int],
                    median: float, consumed: set[tuple[int, int]],
                    title: str) -> list[Section]:
    root: tuple[str, ...] = (title,) if title else ()
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    cur_path: tuple[str, ...] = root
    buf: list[str] = []
    start_page = 0
    end_page = 0

    def flush() -> None:
        nonlocal buf, start_page, end_page
        text = "\n\n".join(part for part in buf if part).strip()
        if text:
            sections.append(
                Section(
                    path=cur_path,
                    level=len(cur_path),
                    text=text,
                    # PDF 没有行号，这两个字段填页码，字段名保持不变
                    line_start=start_page or 1,
                    line_end=end_page or start_page or 1,
                )
            )
        buf = []
        start_page = 0
        end_page = 0

    for page in pages:
        for index, block in enumerate(page.blocks):
            if (page.number, index) in consumed:
                continue
            if _norm_key(block.text) in skip:
                continue
            lines = [ln for ln in block.text.splitlines() if ln.strip()]
            if not lines:
                continue

            if _is_heading(block, median):
                flush()
                level = levels.get(round(block.size, 1), len(root) + 1)
                heading = " ".join(ln.strip() for ln in lines)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading))
                cur_path = root + tuple(name for _, name in stack)
                start_page = end_page = page.number
                continue

            merged = merge_short_lines(lines)
            if not merged:
                continue
            if not buf:
                start_page = page.number
            buf.append("\n\n".join(merged))
            end_page = page.number

    flush()
    return sections


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #
def parse_pdf(
    path: Path,
    *,
    max_chars: int = 1200,
    min_chars: int = 120,
    drop_below: int = 40,
    overlap: int = 120,
    code_block_max_chars: int = 2400,
) -> PdfParseResult:
    """解析一个 PDF，返回带页码 locator 的 `Section` 切出来的 `Chunk` 列表。

    切分参数与 `chunker.chunk_document` 同名同义，直接来自 `[chunk]` 配置段。
    失败（加密 / 零页 / 无文本层 / 文件损坏）返回带 `reason` 的结果，不抛异常。
    """
    path = Path(path)
    pymupdf = _load_pymupdf()

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:  # noqa: BLE001 - 损坏的 PDF 抛什么类型取决于版本
        return PdfParseResult(
            title=path.stem,
            reason=f"pdf_unreadable: {exc.__class__.__name__}: {exc}",
        )

    try:
        if getattr(doc, "needs_pass", 0):
            return PdfParseResult(
                title=path.stem, n_pages=int(getattr(doc, "page_count", 0) or 0),
                reason="pdf_encrypted: 需要打开密码，内容无法解析",
            )

        n_pages = int(getattr(doc, "page_count", 0) or 0)
        if n_pages <= 0:
            return PdfParseResult(title=path.stem, reason="pdf_zero_pages: 文件里没有页面")

        pages: list[PdfPage] = []
        for number in range(1, n_pages + 1):
            try:
                pages.append(_read_page(doc[number - 1], number))
            except (RuntimeError, ValueError) as exc:
                return PdfParseResult(
                    title=path.stem, n_pages=n_pages,
                    reason=f"pdf_unreadable: 第 {number} 页读不出来（{exc}）",
                )

        text_chars = sum(len(block.text) for page in pages for block in page.blocks)
        if text_chars == 0:
            return PdfParseResult(
                title=path.stem, n_pages=n_pages,
                reason="pdf_no_text_layer: 没有文本层（纯图片或扫描件），"
                       "需要先 OCR 才能入库",
            )

        skip = _margin_keys(pages)
        median = _median_size(pages, skip)
        title, consumed = _document_title(doc, path, pages, median, skip)
        levels = _heading_levels(pages, skip, median)
        sections = _build_sections(pages, skip, levels, median, consumed, title)
    finally:
        doc.close()

    chunks = chunk_sections(
        sections,
        max_chars=max_chars,
        min_chars=min_chars,
        drop_below=drop_below,
        overlap=overlap,
        code_block_max_chars=code_block_max_chars,
    )

    notes: list[str] = [f"{n_pages} 页 / {text_chars:,} 字符"]
    if skip:
        notes.append(f"剔除页眉页脚 {len(skip)} 条")
    if levels:
        notes.append(f"识别标题字号 {len(levels)} 档")

    result = PdfParseResult(
        title=title, chunks=chunks, n_pages=n_pages, text_chars=text_chars,
        notes=notes,
    )
    if not chunks:
        result.reason = (
            "pdf_no_chunks: 有文本层但切不出合格的块"
            f"（{text_chars} 字符，都短于 drop_below={drop_below}）"
        )
    return result
