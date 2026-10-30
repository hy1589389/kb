"""零依赖分词：CJK 走字符 unigram + bigram，ASCII 走正则提词。

为什么不用分词器（jieba / pkuseg / HanLP）：

1. 装不上。本机 pip 在解包阶段会被沙箱拒绝，而 BM25 只需要一个稳定的
   token 序列，为它引入一个 30 MB 的词典 + 一个需要编译的分词器不划算。
2. 对检索没必要。中文分词的错误会**不可逆地**污染召回：把「深度学习」
   切成「深度/学习」而查询里是「深度学习框架」，或者反过来把「南京市
   长江大桥」切错，「分词错了 → 检索不到」这条链路上没有任何补救手段。
   unigram + bigram 把两种粒度同时喂给 BM25：unigram 保证「任何单字都
   不会漏」，bigram 提供短语搭配的区分度（「学习」的 bigram 让「深度
   学习」和「机器学习」能区分开）。检索里这个组合的召回率与分词器方案
   相当，且**不会因为词典未收录新词而失效**（术语、模型名、缩写天天在变，
   词典永远滞后）。
3. 打分逻辑更干净。BM25 的 IDF 会自动把 unigram 里的停用字（的、是、在）
   压到接近 0，不需要额外的停用词表；bigram 的 IDF 才是真正干活的信号。

代价：索引会变大（一篇 500 字中文 → 约 1000 个 token 位置，unigram +
bigram 各占一半），去重后的词表也更大。实测 18,276 篇文档的倒排索引约
一百多 MB，可以接受。

中英混排、全角、CJK 标点的处理见 `tokenize` 的注释。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

__all__ = ["tokenize", "tokenize_counts", "is_cjk", "normalize_text"]

# CJK 统一表意文字（含扩展 A）、兼容表意文字、假名。
# 这些字符按「字符」处理（unigram + bigram），不做词切分。
_CJK_RANGES = (
    (0x3400, 0x4DBF),   # CJK 扩展 A
    (0x4E00, 0x9FFF),   # CJK 基本区
    (0xF900, 0xFAFF),   # CJK 兼容表意文字
    (0x3040, 0x309F),   # 平假名
    (0x30A0, 0x30FF),   # 片假名
)

# 一个连续的 CJK 串，或一个连续的 ASCII/数字/下划线词。
# 用 finditer 而不是 split：split 会在标点处产生大量空串，且拿不到位置。
_TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff]+"
    r"|[0-9A-Za-z_]+"
)

# 全角 ASCII（！到～）与全角空格，统一折成半角。
_FULLWIDTH_SPACE = "\u3000"


def is_cjk(ch: str) -> bool:
    """单字符是否属于按字符处理的 CJK 区段。"""
    o = ord(ch)
    for lo, hi in _CJK_RANGES:
        if lo <= o <= hi:
            return True
    return False


def normalize_text(text: str) -> str:
    """NFKC 归一 + 大小写折叠 + 去除零宽字符。

    NFKC 会把全角字母数字（ＡＢＣ１２３）折成半角，「／」折成「/」，这样
    全角输入的查询和半角语料能对上。注意 NFKC 也会把某些兼容字符（如 ㈱、
    Ⅻ）折成多字符序列，对检索是好事：折完两边一致。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_FULLWIDTH_SPACE, " ")
    # 零宽空格 / BOM / 软连字符：出现在语料里会切断词，清理掉
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"):
        if ch in text:
            text = text.replace(ch, "")
    return text.lower()


def tokenize(text: str) -> list[str]:
    """把一段文本切成 token 列表（保留重复与顺序）。

    规则：
      · CJK 连续串 -> 每个字符一个 unigram，再每个相邻字符对一个 bigram。
        单字串只产生一个 unigram，不产生 bigram（bigram 需要两个字符）。
      · ASCII/数字/下划线连续串 -> 整个串作为一个 token（不拆词干）。
      · CJK 标点（，。！？「」《》）不属于上面的区段，被正则跳过，不会
        产生 token，也不会成为 bigram 的一部分。所以「中文，英文」里的
        bigram 只在各自串内部生成，不会跨标点粘连。
      · 全角字母数字先被 NFKC 折成半角，因此「ＡＢＣ」与「abc」等价。
      · 中英混排如「用Python写爬虫」会被切成三段：CJK「用」/ ASCII
        「python」/ CJK「写爬虫」，三段各自内部生成 bigram，不会出现
        跨语种的假 bigram。

    全大写的缩写（HTTP、TCP）lower 之后与语料一致；不保留大小写版本，
    因为检索查询里的大小写是随意的。
    """
    if not text:
        return []

    text = normalize_text(text)
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        piece = m.group(0)
        if is_cjk(piece[0]):
            out.extend(piece)                                  # unigram
            if len(piece) > 1:
                out.extend(piece[i:i + 2] for i in range(len(piece) - 1))  # bigram
        else:
            out.append(piece)                                  # ASCII 词
    return out


def tokenize_counts(text: str) -> Counter[str]:
    """`tokenize` 的计数版。建索引时只关心词频，不需要保留位置。"""
    return Counter(tokenize(text))
