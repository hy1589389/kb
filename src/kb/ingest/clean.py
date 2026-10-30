"""文本清洗：剔除对检索无贡献的噪声，同时尽量不破坏围栏代码块。

清洗策略（按杀伤力从大到小）：
  1. YAML front matter —— 元数据，不是正文；
  2. 徽章 / 图片 —— README 顶部的 shields.io 之类，纯噪声；
  3. HTML 注释与裸标签 —— 排版残留；
  4. 只含图片的行 —— 图片没了就只剩空行。

★ 核心约束：所有清洗都必须「保持行数不变」。
  因为代码块标记、行号回链都依赖「清洗后的第 i 行 == 原文的第 i 行」。
  早期版本在清洗时压缩了空行，导致标记数组与文本错位，
  后果是标题被当成正文、整节内容被静默丢弃 —— 这类 bug 极难发现，
  所以这里用「等量换行替换」的方式做删除，从设计上杜绝错位。
"""

from __future__ import annotations

import re

_RE_FENCE_OPEN = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
_RE_FRONT_MATTER = re.compile(r"\A\s*---\s*\n.*?\n---\s*\n", re.DOTALL)
_RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_RE_HTML_TAG_ONLY = re.compile(r"^\s*</?[A-Za-z][A-Za-z0-9]*(\s[^>]*)?/?>\s*$")
_RE_BADGE_HOST = re.compile(
    r"(img\.shields\.io|badgen\.net|badge\.svg|codecov\.io|travis-ci|appveyor|circleci|gitee\.com/.*/badge)",
    re.IGNORECASE,
)
_RE_MULTI_BLANK = re.compile(r"\n{3,}")
_RE_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)

# 超长的「无空白字符串」——base64 图片、压缩过的数据、超长哈希。
# 这类内容对检索零贡献，却能把 chunk 撑到几十 KB，还会占满嵌入模型的上下文。
_RE_LONG_TOKEN = re.compile(r"[A-Za-z0-9+/=_\-]{200,}")
LONG_TOKEN_THRESHOLD = 200

# 供 chunker 复用（拆超大代码块时要重新识别围栏），避免同一条正则两处维护
FENCE_RE = _RE_FENCE_OPEN


def _blank_out(match: re.Match) -> str:
    """把匹配到的内容替换成等量的换行 —— 删内容但不动行号。"""
    return "\n" * match.group(0).count("\n")


def normalize(text: str) -> str:
    """去 BOM、统一换行、剥离 front matter 与 HTML 注释（保持行数不变）。"""
    text = text.lstrip("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_FRONT_MATTER.sub(_blank_out, text, count=1)
    text = _RE_HTML_COMMENT.sub(_blank_out, text)
    return text


def mark_code_fences(lines: list[str]) -> list[bool]:
    """标记每行是否位于围栏代码块内部（围栏行本身也算内部）。

    这样后续切块可以保证「绝不切断代码块」，对技术文档尤其重要。
    闭合围栏必须满足 CommonMark 规则：同种字符、长度不短于开启围栏、且后面无内容。
    """
    flags: list[bool] = []
    in_code = False
    fence_char = ""
    fence_len = 0

    for line in lines:
        m = _RE_FENCE_OPEN.match(line)
        if m:
            marker, rest = m.group(1), m.group(2).strip()
            char, length = marker[0], len(marker)
            if not in_code:
                in_code, fence_char, fence_len = True, char, length
                flags.append(True)
                continue
            if char == fence_char and length >= fence_len and rest == "":
                in_code, fence_char, fence_len = False, "", 0
                flags.append(True)
                continue
        flags.append(in_code)
    return flags


def compress_long_tokens(line: str, threshold: int = LONG_TOKEN_THRESHOLD) -> str:
    """把超长无空白串替换成占位符。

    实测来源：self-llm 的 Gemma3 教程里有 `data:image/png;base64,...` 的示例，
    单个代码块 56KB、100% 是 base64。这种内容既搜不到也用不上，
    但会把 chunk 尺寸分布拉出一个荒谬的长尾。压缩后同样的块降到几百字符。

    只替换长度 >= threshold 的连续串，正常标识符（哪怕很长）不受影响。
    """
    if len(line) < threshold:
        return line

    def repl(m: re.Match) -> str:
        return f"[已省略 {len(m.group(0))} 字符的长串数据]"

    return _RE_LONG_TOKEN.sub(repl, line)


def clean_lines(lines: list[str], flags: list[bool]) -> list[str]:
    """逐行清洗，代码块内部只做长串压缩（行数不变，保证行号可回链）。"""
    out: list[str] = []
    for line, in_code in zip(lines, flags):
        if in_code:
            # base64 通常藏在代码块里，所以代码块也要压缩长串，只是不做其它删改
            out.append(compress_long_tokens(line).rstrip())
            continue
        if _RE_BADGE_HOST.search(line):
            out.append("")
            continue
        without_images = _RE_IMAGE.sub("", line)
        if line.strip() and not without_images.strip():
            # 整行只有图片
            out.append("")
            continue
        if _RE_HTML_TAG_ONLY.match(without_images):
            out.append("")
            continue
        out.append(compress_long_tokens(without_images).rstrip())
    return out


def clean_text(text: str) -> tuple[list[str], list[bool]]:
    """一站式：normalize → 切行 → 标记代码块 → 逐行清洗。

    返回 (清洗后的行列表, 代码块标记)。行数与原文严格一致，可用于行号回链。
    """
    lines = normalize(text).split("\n")
    flags = mark_code_fences(lines)
    return clean_lines(lines, flags), flags


def collapse_blanks(text: str) -> str:
    """压缩空行、去掉行尾空白。

    注意：这是对「拼装好的字符串」做的处理，不要用在行数组上 ——
    它会让行数与原文不一致，破坏行号回链。
    """
    return _RE_MULTI_BLANK.sub("\n\n", _RE_TRAILING_WS.sub("", text))
