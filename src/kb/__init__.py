"""kb —— 本地知识库。

Phase 1: 语料入库（Markdown + PDF，多集合）
Phase 2: 混合检索 + 重排
Phase 3: 评测体系
Phase 4: 生成与引用溯源

## 依赖发现

本机 pip 装不了包，第三方依赖是手动解压的 wheel（参见 `embedder.py` 顶部注释）。
`kb` 包在 import 时把下面这些目录插到 `sys.path` 最前面，这样 `pymupdf` /
`tokenizers` 在任何模块里都能直接 `import`，不用每个文件各自注入：

    KB_PYLIBS               环境变量，指向任意解压目录（换机器 / CI 用）
    <项目根>/.pylibs        tokenizers、pymupdf 的正式位置
    G:\\kb\\.pylibs          本机共享的依赖目录（只读引用，模型和原始语料都在 G:\\kb）

这里**只改 sys.path，不 import 任何第三方模块**：缺包时让真正使用的那个
模块抛出它自己的 ImportError（`ingest/pdf.py` 会补一条可操作的中文提示），
不会在这里被静默吞掉。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.1.0"

# src/kb/__init__.py -> src/kb -> src -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PYLIBS_ENV = "KB_PYLIBS"
_PYLIBS_DEFAULTS: tuple[Path, ...] = (Path(r"G:\kb\.pylibs"),)


def pylibs_candidates() -> list[Path]:
    """按优先级返回候选依赖目录（不保证存在）。"""
    out: list[Path] = []
    env = (os.environ.get(_PYLIBS_ENV) or "").strip()
    if env:
        out.append(Path(env))
    out.append(PROJECT_ROOT / ".pylibs")
    out.extend(_PYLIBS_DEFAULTS)

    seen: set[str] = set()
    unique: list[Path] = []
    for path in out:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def ensure_pylibs() -> list[Path]:
    """把存在的依赖目录插到 `sys.path`，返回本次实际插入的目录（按优先级）。

    幂等：已经在 `sys.path` 里的目录不会重复插入。目录不存在时静默跳过 ——
    真正的「缺包」由使用方的 import 报错，不在这里判断。
    """
    inserted: list[Path] = []
    existing = {str(Path(p)) for p in sys.path if p}
    # 反序插入到 0，插完 sys.path 的顺序与 candidates 的优先级一致
    for path in reversed(pylibs_candidates()):
        try:
            if not path.is_dir():
                continue
        except OSError:
            continue
        text = str(path.resolve())
        if text in existing:
            continue
        sys.path.insert(0, text)
        existing.add(text)
        inserted.append(path)
    inserted.reverse()  # 恢复成候选顺序，方便打印和断言
    return inserted


PYLIBS_DIRS = ensure_pylibs()

__all__ = ["PYLIBS_DIRS", "PROJECT_ROOT", "ensure_pylibs", "pylibs_candidates",
           "__version__"]
