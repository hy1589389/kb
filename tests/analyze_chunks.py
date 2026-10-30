"""切块质量体检：找出尺寸异常值，定位是「谁」把分布尾巴拉长的。

用法：  python tests/analyze_chunks.py G:\\kb\\data\\processed\\chunks_all.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    sizes = [r["char_count"] for r in rows]
    total = sum(sizes)

    print(f"文件        : {path}")
    print(f"切块总数    : {len(rows):,}")
    print(f"总字符      : {total:,}")
    print()

    buckets = [
        (0, 100), (100, 300), (300, 600), (600, 1000),
        (1000, 1500), (1500, 2000), (2000, 4000), (4000, 10000), (10000, 10**9),
    ]
    print("尺寸分布")
    for lo, hi in buckets:
        n = sum(1 for s in sizes if lo <= s < hi)
        label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        bar = "#" * int(n / len(rows) * 60)
        print(f"  {label:>12} : {n:>7,}  {n / len(rows) * 100:5.1f}%  {bar}")
    print()

    over = [r for r in rows if r["char_count"] > limit]
    print(f"超过 {limit:,} 字符的块 : {len(over):,}  ({len(over) / len(rows) * 100:.2f}%)")
    print(f"这些块贡献的字符占比  : {sum(r['char_count'] for r in over) / total * 100:.2f}%")
    print()

    print(f"最大的 {min(15, len(over))} 个块")
    for r in sorted(over, key=lambda x: -x["char_count"])[:15]:
        text = r["text"]
        code_chars = 0
        in_code = False
        fence_char = ""
        for ln in text.split("\n"):
            stripped = ln.strip()
            if stripped[:3] in ("```", "~~~"):
                if not in_code:
                    in_code, fence_char = True, stripped[0]
                elif stripped[0] == fence_char and len(stripped) >= 3 and stripped.strip(stripped[0]) == "":
                    in_code = False
                continue
            if in_code:
                code_chars += len(ln) + 1
        ratio = code_chars / max(1, len(text)) * 100
        lines = text.split("\n")
        longest = max((len(x) for x in lines), default=0)
        print(
            f"  {r['char_count']:>7,} 字  代码占 {ratio:5.1f}%  "
            f"行数 {len(lines):>5}  最长行 {longest:>6}  "
            f"{r['repo']}/{r['rel_path']}"
        )
        print(f"            breadcrumb: {r['breadcrumb'][:70]}")
        if longest > 800:
            sample = next(x for x in lines if len(x) == longest)
            print(f"            最长行开头: {sample[:110]}")
    print()

    # 谁在制造异常值
    print("异常值来源 TOP 10（按 > 限制 的块数）")
    counter: Counter[str] = Counter()
    for r in over:
        counter[f"{r['repo']}/{r['rel_path']}"] += 1
    for name, n in counter.most_common(10):
        print(f"  {n:>4} 块  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
