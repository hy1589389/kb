"""调试用：把 chunks JSONL 以人类可读的形式打印出来。

用法：  python tests/dump_chunks.py tests/out/chunks_all.jsonl [--no-text]

（注意别把这个文件叫 inspect.py ——  它会遮住标准库的 inspect 模块，
 导致 dataclasses 直接报错。这坑踩过一次就够。）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/out/chunks_all.jsonl")
    show_text = "--no-text" not in sys.argv

    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        c = json.loads(line)
        print("=" * 72)
        print(f"#{i}  [{c['repo']}] {c['rel_path']}")
        print(f"    breadcrumb : {c['breadcrumb']}")
        print(
            f"    lines {c['line_start']}-{c['line_end']}"
            f"   chars={c['char_count']}   cjk={c['cjk_count']}"
            f"   part={c['part']}/{c['n_parts']}"
        )
        if c["merged_paths"]:
            print(f"    merged     : {c['merged_paths']}")
        if show_text:
            print("    ---- text ----")
            for ln in c["text"].split("\n"):
                print(f"    | {ln}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
