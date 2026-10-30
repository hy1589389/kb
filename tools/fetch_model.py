"""从 hf-mirror 下载 ONNX 模型，支持断点续传。

这台机器上 huggingface.co 不通、hf-mirror.com 通；pip 装不了包但 urllib 正常。
bge-m3 的 fp32 权重有 2.1 GB，所以必须能续传，中断了不用从头来。

用法：
    python fetch_model.py <目标目录> [repo]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MIRROR = "https://hf-mirror.com"
UA = {"User-Agent": "dsh"}


def list_files(repo: str) -> list[str]:
    req = urllib.request.Request(f"{MIRROR}/api/models/{repo}", headers=UA)
    info = json.load(urllib.request.urlopen(req, timeout=60))
    return [s["rfilename"] for s in info.get("siblings", [])]


def head_size(url: str) -> int:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)


def download(repo: str, remote: str, dest: Path) -> bool:
    url = f"{MIRROR}/{repo}/resolve/main/{remote}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        total = head_size(url)
    except Exception as exc:  # noqa: BLE001
        print(f"  [X] {remote}: 取不到大小 ({type(exc).__name__})")
        return False

    have = dest.stat().st_size if dest.exists() else 0
    if have == total and total > 0:
        print(f"  [=] {remote}  已完成 ({total / 1024**2:.1f} MB)")
        return True

    headers = dict(UA)
    mode = "wb"
    if have > 0 and have < total:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        print(f"  [~] {remote}  续传 {have / 1024**2:.1f}/{total / 1024**2:.1f} MB")
    else:
        have = 0
        print(f"  [>] {remote}  {total / 1024**2:.1f} MB")

    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    last = t0
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, dest.open(mode) as fh:
            done = have
            while True:
                buf = resp.read(1024 * 256)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                now = time.time()
                if now - last >= 5:
                    pct = done / total * 100 if total else 0
                    speed = (done - have) / max(now - t0, 1e-6) / 1024**2
                    print(f"      {pct:5.1f}%  {done / 1024**2:7.1f} MB  {speed:5.1f} MB/s", flush=True)
                    last = now
    except Exception as exc:  # noqa: BLE001
        print(f"  [X] {remote} 中断: {type(exc).__name__}: {exc}")
        return False

    ok = dest.stat().st_size == total
    print(f"  [{'OK' if ok else '!!'}] {remote}  {dest.stat().st_size / 1024**2:.1f} MB")
    return ok


def main() -> int:
    dest_root = Path(sys.argv[1]).resolve()
    repo = sys.argv[2] if len(sys.argv) > 2 else "BAAI/bge-m3"
    name = repo.split("/")[-1]
    dest_dir = dest_root / name

    print(f"仓库   : {repo}")
    print(f"目标   : {dest_dir}")
    print()

    try:
        files = list_files(repo)
    except Exception as exc:  # noqa: BLE001
        print(f"[X] 取不到文件列表: {type(exc).__name__}: {exc}")
        return 1

    onnx = [f for f in files if f.startswith("onnx/") and "onnx" in f.lower()]
    print("仓库里的 ONNX 相关文件：")
    for f in onnx:
        print(f"  {f}")
    print()

    # 优先 fp16（体积减半，CPU 上精度损失可接受），否则 fp32
    has_fp16 = any("fp16" in f.lower() for f in onnx)
    if has_fp16:
        wanted = [f for f in onnx if "fp16" in f.lower() or f.endswith("tokenizer.json")
                  or f.endswith("tokenizer_config.json") or f.endswith("special_tokens_map.json")
                  or f.endswith("config.json")]
        print("使用 fp16 版本")
    else:
        wanted = [f for f in onnx if f.endswith((".onnx", ".onnx_data", ".json"))]
        print("没有 fp16，使用 fp32 版本")

    # 有些仓库（bge-reranker-base 就是这样）把 tokenizer 放在仓库根目录而不是 onnx/ 下。
    # 只按 onnx/ 前缀过滤会漏掉它们，结果是模型下全了却加载不了分词器。这里补齐。
    for extra in ("tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "config.json"):
        if extra in files and extra not in wanted:
            wanted.append(extra)

    print()
    ok_all = True
    for remote in wanted:
        dest = dest_dir / Path(remote).name
        if not download(repo, remote, dest):
            ok_all = False

    print()
    total = sum(p.stat().st_size for p in dest_dir.glob("*") if p.is_file())
    print(f"完成: {ok_all}   目录合计 {total / 1024**2:.1f} MB")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
