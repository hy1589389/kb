"""入库质量指标：三项阈值，任一超标就把文档标成 low_quality。

为什么入库要做这一步？
    上传的 PDF 质量不受控：扫描件没有文本层、双栏论文被抽成一团、编码错乱
    的导出件全是 ``□`` 和 ``\ufffd``。这些东西一旦进了索引，会以「看起来
    正常、就是搜不到」的方式污染检索，比直接报错难查得多。

三项指标（阈值见 `THRESHOLDS`）：

| 指标 | 阈值 | 说明 |
| --- | --- | --- |
| 平均块字符数 | < 100 | 多半是扫描件或版面切分失败 |
| 乱码率 | > 15% | 编码错乱，或公式/符号密集 |
| 空白块占比 | > 30% | 切出来的块没有有效文字 |

**low_quality 的块仍然入库**，只是带上 `low_quality: true`；检索默认排除，
`include_low_quality=True` 才纳入。删掉就再也查不到了，标记则还能复查。

乱码率的算法（题目要求）：统计「非中文、非 ASCII 可打印字符（0x20-0x7E）、
非常见中文标点」的字符占比。空白字符不计入分母 —— 换行和空格不是乱码，
把它们算进去会让长文档的乱码率被稀释。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

__all__ = [
    "THRESHOLDS",
    "QualityReport",
    "assess_texts",
    "garbled_ratio",
    "is_blank_text",
    "is_garbled_char",
    "mark_low_quality",
]

# 阈值常量，测试和 CLI 都从这里取，不另抄一份
THRESHOLDS: dict[str, float] = {
    "avg_chars": 100.0,     # 平均块字符数低于此值 -> 警告
    "garbled_rate": 0.15,   # 乱码率高于此值 -> 警告
    "blank_ratio": 0.30,    # 空白块占比高于此值 -> 警告
}

# 允许的字符区间：中文（含扩展 A、CJK 标点、全角字符）+ ASCII 可打印
_ALLOWED_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),  # CJK 统一表意文字
    (0x3400, 0x4DBF),  # 扩展 A
    (0x3000, 0x303F),  # CJK 标点（，。「」等）
    (0xFF00, 0xFFEF),  # 全角字符（：；？！等）
    (0x0020, 0x007E),  # ASCII 可打印
)

# 不计入乱码率的空白字符（含全角空格、不换行空格）
_WHITESPACE = frozenset(" \t\r\n\v\f\u3000\u00a0\u200b\ufeff")


def is_garbled_char(ch: str) -> bool:
    """这个字符算不算「乱码」。空白字符不算（也不算进分母）。"""
    if ch in _WHITESPACE:
        return False
    code = ord(ch)
    return not any(lo <= code <= hi for lo, hi in _ALLOWED_RANGES)


def garbled_ratio(text: str) -> float:
    """乱码字符占「非空白字符」的比例，0.0 ~ 1.0。"""
    total = garbled = 0
    for ch in text or "":
        if ch in _WHITESPACE:
            continue
        total += 1
        if is_garbled_char(ch):
            garbled += 1
    return garbled / total if total else 0.0


def is_blank_text(text: str) -> bool:
    """块是否「没有有效文字」。

    只判 `strip()` 是不够的：版面切分失败产出的块常常是一堆标点或符号
    （``。、，`` / ``| | |``），长度不为零但对检索毫无价值。这里要求块里
    至少有一个中文、字母或数字。
    """
    for ch in text or "":
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            return False
        if ch.isalnum():
            return False
    return True


@dataclass
class QualityReport:
    """一个文档的质量体检结果。`reasons` 非空即 `low_quality`。"""

    n_chunks: int = 0
    avg_chars: float = 0.0
    garbled_rate: float = 0.0
    blank_ratio: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def low_quality(self) -> bool:
        return bool(self.reasons)

    def summary(self) -> str:
        """一行摘要，dry-run 和入库日志都用它。"""
        head = (f"块 {self.n_chunks}  平均 {self.avg_chars:.0f} 字符  "
                f"乱码 {self.garbled_rate:.1%}  空白块 {self.blank_ratio:.0%}")
        if not self.reasons:
            return f"{head}  [正常]"
        return f"{head}  [警告] {'；'.join(self.reasons)}"


def assess_texts(texts: list[str]) -> QualityReport:
    """按三项指标体检一组块文本。

    乱码率按字符加权（总乱码字符 / 总非空白字符），不是「各块乱码率的平均」：
    一个 3000 字的乱码块比一个 20 字的乱码块严重，按块平均会把前者稀释掉。
    """
    n = len(texts)
    if n == 0:
        return QualityReport()

    sizes = [len(t or "") for t in texts]
    avg = statistics.fmean(sizes) if sizes else 0.0
    blank = sum(1 for t in texts if is_blank_text(t))
    blank_ratio = blank / n

    total = garbled = 0
    for text in texts:
        for ch in text or "":
            if ch in _WHITESPACE:
                continue
            total += 1
            if is_garbled_char(ch):
                garbled += 1
    rate = garbled / total if total else 0.0

    report = QualityReport(
        n_chunks=n,
        avg_chars=round(avg, 1),
        garbled_rate=round(rate, 4),
        blank_ratio=round(blank_ratio, 4),
    )
    if avg < THRESHOLDS["avg_chars"]:
        report.reasons.append(
            f"平均块字符数 {avg:.0f} < {THRESHOLDS['avg_chars']:.0f}"
            f"（可能是扫描件，或版面切分把段落切碎了）"
        )
    if rate > THRESHOLDS["garbled_rate"]:
        report.reasons.append(
            f"乱码率 {rate:.1%} > {THRESHOLDS['garbled_rate']:.0%}"
            f"（编码错乱，或公式/符号密集）"
        )
    if blank_ratio > THRESHOLDS["blank_ratio"]:
        report.reasons.append(
            f"空白块占比 {blank_ratio:.0%} > {THRESHOLDS['blank_ratio']:.0%}"
            f"（切出来的块没有有效文字）"
        )
    return report


def mark_low_quality(chunks: list[dict], flag: bool) -> list[dict]:
    """给一批 chunk 记录打上 / 清掉 `low_quality` 字段，返回原列表。"""
    for chunk in chunks:
        if flag:
            chunk["low_quality"] = True
        else:
            chunk.pop("low_quality", None)
    return chunks
