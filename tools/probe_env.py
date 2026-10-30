"""Phase 2 环境勘察：确认能装什么、跑得动什么。"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print(f"Python      : {sys.version.split()[0]}")
print(f"解释器路径  : {sys.executable}")
print()

print("=== 已安装的检索相关包 ===")
for m in [
    "numpy", "scipy", "torch", "transformers", "sentence_transformers",
    "FlagEmbedding", "onnxruntime", "lancedb", "chromadb", "qdrant_client",
    "jieba", "rank_bm25", "fastapi", "uvicorn", "httpx", "tqdm", "pydantic",
]:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        loc = getattr(mod, "__file__", "") or ""
        print(f"  {m:22} OK    {ver:>12}   {Path(loc).parent}")
    except Exception:
        print(f"  {m:22} -")
print()

print("=== 磁盘空间 ===")
for letter in ("C", "D", "G"):
    try:
        import ctypes

        free = ctypes.c_ulonglong(0)
        total = ctypes.c_ulonglong(0)
        ok = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(f"{letter}:\\"),
            None, ctypes.byref(total), ctypes.byref(free),
        )
        if ok:
            print(f"  {letter}:  可用 {free.value / 1024**3:8.1f} GB / 共 {total.value / 1024**3:8.1f} GB")
    except Exception as exc:  # noqa: BLE001
        print(f"  {letter}:  探测失败 {exc}")
print()

print("=== 关键外部工具 ===")
for tool in ("docker", "git", "nvidia-smi", "nvcc"):
    print(f"  {tool:12} {shutil.which(tool) or '-'}")
print()

print("=== torch 能否看到 CUDA ===")
try:
    import torch

    print(f"  torch {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
        print(f"  vram  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except Exception as exc:  # noqa: BLE001
    print(f"  torch 不可用: {type(exc).__name__}")
