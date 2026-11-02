"""组装一个可以直接 push 到 Hugging Face Spaces 的目录。

用法：

    python tools/prepare_space.py G:\\hf-kb
    python tools/prepare_space.py G:\\hf-kb --index-dir data/index --force

它做的事：
  · 复制代码（src / tools / ingest.py / config.toml）和部署文件
  · 复制检索索引，**跳过 profile** —— 那是个人简历，不能公开
  · 生成带 HF front-matter 的 README.md（Spaces 靠这个识别 SDK 和端口）
  · 不复制 data/raw、models、.pylibs*、tests、results

复制完不会自动 push，git 相关操作留给人工确认。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

# 顶层要带过去的文件（目录单独处理）
FILES = [
    "ingest.py",
    "config.toml",
    "requirements-deploy.txt",
    "Dockerfile",
    ".dockerignore",
]

# 整个目录都要带
DIRS = ["src", "tools"]

# 索引里不能公开的部分
INDEX_EXCLUDE = {"profile"}

SPACE_README = """---
title: 本地知识库问答
emoji: 📚
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 本地知识库问答系统（在线演示）

把 5 个中文技术仓库（1,278 篇 / 1,150 万字）切成 18,276 个块，用 BM25 +
bge-m3 稠密向量双路召回、RRF 融合，检索结果交给 DeepSeek 生成带 `[n]` 引用的答案，
检索不到时明确拒答。

代码和完整说明：[github.com/hy1589389/kb](https://github.com/hy1589389/kb)

## 和本地版的差别

| | 本地 | 这里 |
| --- | --- | --- |
| 嵌入推理 | DirectML（4 GB 显卡），单次检索 153 ms | CPU（2 vCPU），单次约 1-2 秒 |
| 集合 | tech + profile（简历） | 只有 tech |
| 语料原文 | 完整，可跳转到源码行 | 不带，跳转按钮失效 |
| 问答 | 需要 API key | 需要配置 `DEEPSEEK_API_KEY` 环境变量 |

## 这里能试什么

- 搜「HashMap 什么时候转红黑树」「TCP 为什么不两次握手」
- 切到问答模式，看带引用的回答
- 问一个语料里没有的问题（比如「如何做法式红酒炖牛肉」），看它拒答
"""


def copy_index(src: Path, dst: Path) -> list[str]:
    """复制索引，跳过 INDEX_EXCLUDE 里的子目录。返回被跳过的名字。"""
    skipped: list[str] = []
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in INDEX_EXCLUDE:
            skipped.append(item.name)
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst / item.name)
    return skipped


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="组装 Hugging Face Spaces 目录")
    ap.add_argument("target", help="目标目录（会被清空重建）")
    ap.add_argument("--index-dir", default="data/index",
                    help="索引目录，相对项目根（默认 data/index）")
    ap.add_argument("--force", action="store_true",
                    help="目标目录非空时也继续（会先清空）")
    args = ap.parse_args(argv)

    target = Path(args.target).expanduser().resolve()
    index_src = (ROOT / args.index_dir).resolve()

    if not (ROOT / "src" / "kb").is_dir():
        print(f"[X] 找不到代码目录：{ROOT / 'src' / 'kb'}")
        return 2
    if not (index_src / "index_meta.json").exists():
        print(f"[X] 索引不完整：{index_src / 'index_meta.json'} 不存在")
        print("    先建索引：python ingest.py index --collection tech")
        return 2

    if target.exists():
        if any(target.iterdir()):
            if not args.force:
                print(f"[X] 目标目录非空：{target}")
                print("    确认可以清空后加 --force")
                return 2
            # 只清内容，保留 .git —— 目标通常是 clone 下来的 Space 仓库，
            # 整个删掉会把远端关联一起弄丢，还得重新 clone
            for child in target.iterdir():
                if child.name == ".git":
                    print("  保留 .git（Space 仓库的远端关联）")
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    target.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        src = ROOT / name
        if not src.exists():
            print(f"[!] 缺少 {name}，跳过")
            continue
        shutil.copy2(src, target / name)

    for name in DIRS:
        src = ROOT / name
        if not src.is_dir():
            print(f"[!] 缺少 {name}/，跳过")
            continue
        shutil.copytree(src, target / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    skipped = copy_index(index_src, target / "data" / "index")

    # 本地会注入项目根的 .pylibs；镜像里靠 pip 装，所以不能把它带过去
    (target / "README.md").write_text(SPACE_README, encoding="utf-8")

    print("=" * 60)
    print(f"  目标目录 : {target}")
    print(f"  索引     : {index_src}")
    if skipped:
        print(f"  已跳过   : {', '.join(skipped)}（不公开个人数据）")
    print("-" * 60)
    for name in FILES + DIRS + ["data/index", "README.md"]:
        p = target / name
        if p.exists():
            size = dir_size(p) if p.is_dir() else p.stat().st_size
            print(f"  {name:26} {human(size):>10}")
    print("-" * 60)
    print(f"  合计 {human(dir_size(target))}")
    print("=" * 60)
    print()
    print("下一步（在目标目录里执行）：")
    print("  git lfs install")
    print("  git lfs track '*.npy' '*.npz' '*.jsonl'")
    print("  git add .gitattributes")
    print("  git add -A && git commit -m 'deploy'")
    print("  git push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
