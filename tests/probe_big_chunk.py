"""针对单个巨型 chunk 的结构探针：看围栏在哪、为什么没被切开。

用法：  python tests/probe_big_chunk.py <chunks.jsonl> [阈值]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kb.ingest.chunker import FENCE_RE, _split_oversized_code_blocks  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = Path(sys.argv[1])
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    big = sorted((r for r in rows if r["char_count"] > threshold), key=lambda r: -r["char_count"])

    print(f"超过 {threshold} 字符的块：{len(big)} 个\n")
    for r in big[:3]:
        lines = r["text"].split("\n")
        print("=" * 72)
        print(f"{r['char_count']} 字  行数 {len(lines)}  {r['repo']}/{r['rel_path']}")
        print(f"breadcrumb: {r['breadcrumb']}")
        fences = [
            (i, ln) for i, ln in enumerate(lines) if FENCE_RE.match(ln)
        ]
        print(f"围栏位置（共 {len(fences)} 个）: {[i for i, _ in fences]}")
        for i, ln in fences[:6]:
            print(f"   L{i}: {ln[:60]!r}")
        print(f"首行: {lines[0][:80]!r}")
        print(f"末行: {lines[-1][:80]!r}")

        # 直接调用切分函数，看它认不认这是超长代码块
        chopped = _split_oversized_code_blocks(r["text"], 2400)
        print(f"再过一次 _split_oversized_code_blocks(2400) 后行数: {len(chopped.split(chr(10)))}")
        print(f"是否发生变化: {chopped != r['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
