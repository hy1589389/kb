"""构建期自检：`data/index` 里是真内容，还是 LFS 指针。

为什么需要这个脚本：索引（139 MB）按 LFS 存在创空间仓库里。平台构建时偶尔
不会 smudge LFS，构建上下文里的 `chunks_meta.jsonl` 就只剩一行
`version https://git-lfs.github.com/spec/v1`，容器起来后在「加载索引」阶段抛

    [X] Expecting value: line 1 column 1 (char 0)

这个错误本身完全看不出是哪个文件、什么原因。所以在构建期先查一遍，不通过就
让 `fetch_index.py` 重拉（Dockerfile 里两条命令用 || 连着）。

用法：
    python tools/check_index.py data/index
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LFS_MAGIC = b"version https://git-lfs"
REQUIRED = (
    "index_meta.json",
    "dense_meta.json",
    "meta.json",
    "vocab.json",
    "chunks_meta.jsonl",
    "embeddings.npy",
    "postings.npz",
)


def _head(path: Path, size: int = 64) -> bytes:
    try:
        with path.open("rb") as fh:
            return fh.read(size)
    except OSError:
        return b""


def check(index_dir: Path) -> list[str]:
    """返回问题列表；空列表表示索引是好的。"""
    problems: list[str] = []
    for name in REQUIRED:
        path = index_dir / name
        if not path.is_file():
            problems.append(f"缺少 {name}")
            continue
        head = _head(path)
        if head.startswith(LFS_MAGIC):
            problems.append(f"{name} 是 LFS 指针（{path.stat().st_size} 字节），内容没拉下来")
            continue
        if not head:
            problems.append(f"{name} 是空文件")
            continue
        # 再按类型做个便宜的格式校验
        try:
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".jsonl":
                with path.open("r", encoding="utf-8") as fh:
                    first = fh.readline().strip()
                if first and not isinstance(json.loads(first), dict):
                    problems.append(f"{name} 第一行不是 JSON 对象")
            elif path.suffix == ".npy":
                if not head.startswith(b"\x93NUMPY"):
                    problems.append(f"{name} 不是 .npy（魔数不对）")
            elif path.suffix == ".npz":
                if not head.startswith(b"PK\x03\x04"):
                    problems.append(f"{name} 不是 .npz（不是 zip）")
        except (OSError, ValueError) as exc:
            problems.append(f"{name} 解析失败：{exc}")
    return problems


def main() -> int:
    index_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/index")
    problems = check(index_dir)
    if not problems:
        meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
        print(f"[OK] 索引自检通过：{index_dir}（{meta.get('n_chunks')} 块，dim={meta.get('dim')}）")
        return 0
    print(f"[X] 索引自检失败：{index_dir}")
    for item in problems:
        print(f"    - {item}")
    print("    如果是 LFS 指针，用 tools/fetch_index.py 从仓库的文件接口重拉：")
    print("      python tools/fetch_index.py data/index --repo <owner>/<space>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
