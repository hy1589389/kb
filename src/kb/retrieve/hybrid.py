"""混合检索的融合：Reciprocal Rank Fusion（RRF）。

## 为什么用 RRF，不用「归一化后加权求和」

稠密检索给的是余弦相似度（-1 ~ 1，实际集中在 0.3~0.8），BM25 给的是
无上界的分数（英文长文档可以到 30+，中文 bigram 语料上常见 5~20）。
这两个数**不在一个量纲上**，直接 `0.5*dense + 0.5*sparse` 的结果完全由
BM25 的尺度决定 —— 稠密那一路等于没参与。

常见的补救是归一化：min-max 到 [0,1]，或者 z-score。但归一化本身不可靠：

  · min-max 依赖**当前查询返回的候选集**。同一个文档在「返回 50 条」
    和「返回 10 条」下归一化后的分数不同，融合权重随 top_k 漂移。
  · 极端值主导。某次查询 BM25 最高分 40、次高 3，min-max 之后第二名
    变成 0.075，一个噪声很大的高分把其他所有结果压平。
  · 分布在查询之间变化。余弦相似度的分布区间随查询长度和主题变化，
    用全局常数做 z-score 同样不准。
  · 调参没有终点。要标定就得为每种查询类型定一套参数，评测集稍微换
    分布就得重标。

RRF 绕开标定：只用**名次**，不用分数。

    score(d) = Σ_i  w_i / (k + rank_i(d))

`rank_i(d)` 是文档 d 在第 i 路结果里的名次，从 1 开始。k 默认 60（论文
*Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning
Methods*, Cormack et al. 2009 的推荐值），作用是削弱头部名次的绝对优势：
k 越小，第 1 名和第 2 名的差距越大；k=60 时 1/61 与 1/62 只差 1.6%，
融合结果更依赖「多路都认可」而不是「某一路给了第一」。

`weights` 是每路的权重，默认全 1。它调的是**路与路之间的信任度**（比如
觉得 BM25 更靠谱就给 1.5），而不是分数的尺度，所以不会遇到上面的问题。

## 一个实现细节：某路没召回该文档

标准 RRF 里没被某路召回的文档，那一路不贡献分数（等价于名次无穷大）。
代码里靠 `rankings[i]` 里不出现该 doc 来实现。

输入里的 doc_index 若在某一路重复出现，只取第一次（名次最小的那次）。
"""

from __future__ import annotations

from collections import defaultdict

__all__ = ["rrf_fuse"]


def rrf_fuse(
    rankings: list[list[int]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """融合若干条按名次排好的 doc_index 列表。

    参数
    ----
    rankings : 每条是一路检索的结果，**按名次升序**（最好的在前）。
               doc_index 是任意整数标签，函数不关心它指向什么。
    k        : RRF 的平滑常数，默认 60。必须 > 0。
    weights  : 与 rankings 等长的权重列表，默认全 1。

    返回
    ----
    [(doc_index, score)]，按 score 降序；分数相同的按 doc_index 升序，
    保证结果稳定（评测脚本需要可复现的排序）。

    例子
    ----
    >>> rrf_fuse([[7, 1, 2], [1, 7, 9]], k=60)
    [(1, ...), (7, ...), (2, ...), (9, ...)]
    # 1 和 7 都被两路召回 -> 分数接近；2 和 9 只被一路召回 -> 更低
    """
    if k <= 0:
        raise ValueError(f"RRF 的 k 必须 > 0，收到 {k}")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"weights 长度({len(weights)}) 与 rankings 路数({len(rankings)}) 不一致"
        )

    scores: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        if weight == 0.0:
            continue
        seen: set[int] = set()
        for position, doc_index in enumerate(ranking, start=1):
            if doc_index in seen:
                continue  # 同一路里重复出现，只认名次最好的那次
            seen.add(doc_index)
            scores[doc_index] += weight / (k + position)

    # 先按分数降序，再按 doc_index 升序，保证完全可复现
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
