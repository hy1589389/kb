"""下载 ONNX 模型，支持断点续传。

这台机器上 huggingface.co 不通、hf-mirror.com 通；pip 装不了包但 urllib 正常。
bge-m3 的 fp32 权重有 2.1 GB，所以必须能续传，中断了不用从头来。

两个来源，用环境变量选：

    KB_MODEL_SOURCE=hf         （默认）走 HF 的 API，默认域名是 hf-mirror.com，
                               HF Spaces 构建时用 HF_ENDPOINT=https://huggingface.co
    KB_MODEL_SOURCE=modelscope 走魔搭的 API（modelscope.cn）。魔搭创空间的构建机
                               在阿里云，直连 modelscope.cn 比走镜像站更快也更稳。

用法：
    python fetch_model.py <目标目录> [repo]

    HF_ENDPOINT=https://huggingface.co python fetch_model.py models BAAI/bge-m3
    KB_MODEL_SOURCE=modelscope        python fetch_model.py models BAAI/bge-m3

两边下下来的文件是一样的（魔搭的 BAAI/bge-m3 同步了 onnx/model.onnx +
onnx/model.onnx_data），所以配置文件里的 model_dir 不用改。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 默认走镜像：开发机（国内）连不上 huggingface.co。
# 容器构建时用 HF_ENDPOINT=https://huggingface.co 直连 —— HF 自己的构建机在
# 内网，走官方域名反而更快，而且镜像站未必在它们的白名单里。
import os  # noqa: E402

MIRROR = (os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com").rstrip("/")
MS_BASE = (os.environ.get("MS_ENDPOINT") or "https://modelscope.cn").rstrip("/")
SOURCE = (os.environ.get("KB_MODEL_SOURCE") or "hf").strip().lower()
UA = {"User-Agent": "dsh"}


def resolve_url(repo: str, remote: str) -> str:
    """某个仓库文件的下载地址。两边的路径规则不同：HF 是 main，魔搭是 master。"""
    if SOURCE == "modelscope":
        return f"{MS_BASE}/models/{repo}/resolve/master/{remote}"
    return f"{MIRROR}/{repo}/resolve/main/{remote}"


def list_files(repo: str) -> dict[str, int]:
    """仓库里的文件 -> 大小（字节）。取不到大小的来源给 0，后面再 HEAD 一次。

    魔搭的 resolve 接口不返回 Content-Length（HEAD 也是空头），所以大小必须
    从列表接口拿，否则下载完没法校验，`total=0` 还会让校验永远判失败。
    """
    if SOURCE == "modelscope":
        url = (f"{MS_BASE}/api/v1/models/{repo}/repo/files"
               "?Revision=master&Recursive=True")
        req = urllib.request.Request(url, headers=UA)
        info = json.load(urllib.request.urlopen(req, timeout=60))
        if int(info.get("Code") or 0) != 200:
            raise RuntimeError(f"魔搭接口返回 {info.get('Code')}: {info.get('Message')}")
        rows = (info.get("Data") or {}).get("Files") or []
        out: dict[str, int] = {}
        for row in rows:
            path = str(row.get("Path") or row.get("Name") or "")
            size = int(row.get("Size") or 0)
            if path and size > 0:          # 目录项的 Size 是 0，跳过
                out[path] = size
        return out

    req = urllib.request.Request(f"{MIRROR}/api/models/{repo}", headers=UA)
    info = json.load(urllib.request.urlopen(req, timeout=60))
    return {str(item["rfilename"]): 0 for item in info.get("siblings", [])}


def head_size(url: str) -> int:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)


def download(repo: str, remote: str, dest: Path, size: int = 0) -> bool:
    url = resolve_url(repo, remote)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        total = size or head_size(url)
    except Exception as exc:  # noqa: BLE001
        print(f"  [X] {remote}: 取不到大小 ({type(exc).__name__})")
        return False
    if not total:
        print(f"  [X] {remote}: 拿不到文件大小，没法校验，放弃")
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
    if SOURCE not in ("hf", "modelscope"):
        print(f"[X] KB_MODEL_SOURCE 只能是 hf 或 modelscope，收到 {SOURCE!r}")
        return 2

    dest_root = Path(sys.argv[1]).resolve()
    repo = sys.argv[2] if len(sys.argv) > 2 else "BAAI/bge-m3"
    name = repo.split("/")[-1]
    dest_dir = dest_root / name

    print(f"来源   : {SOURCE}（{MS_BASE if SOURCE == 'modelscope' else MIRROR}）")
    print(f"仓库   : {repo}")
    print(f"目标   : {dest_dir}")
    print()

    try:
        files = list_files(repo)
    except Exception as exc:  # noqa: BLE001
        print(f"[X] 取不到文件列表: {type(exc).__name__}: {exc}")
        return 1
    if not files:
        print("[X] 文件列表是空的：仓库名写错了，或者这个来源上没有这个仓库")
        return 1

    names = list(files)
    onnx = [f for f in names if f.startswith("onnx/") and "onnx" in f.lower()]
    print(f"仓库里 {len(names)} 个文件，ONNX 相关 {len(onnx)} 个：")
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
        if not download(repo, remote, dest, files.get(remote, 0)):
            ok_all = False

    print()
    total = sum(p.stat().st_size for p in dest_dir.glob("*") if p.is_file())
    print(f"完成: {ok_all}   目录合计 {total / 1024**2:.1f} MB")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
