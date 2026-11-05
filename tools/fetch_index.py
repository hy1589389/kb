"""从创空间仓库的文件接口重拉检索索引（绕开构建上下文的 LFS 问题）。

背景：索引按 LFS 存在创空间仓库里，平台构建时偶尔不 smudge，构建上下文里只剩
`version https://git-lfs...` 的指针。`/api/v1/studio/<owner>/<name>/repo?FilePath=`
这个接口返回的是 **LFS 解析后的真实内容**（HTTP 206），所以在镜像里重拉一份即可。

用法（镜像构建时由 Dockerfile 调用，正常部署不需要手动跑）：

    python tools/fetch_index.py data/index --repo HY666YS/kb-qa
    python tools/fetch_index.py data/index --repo HY666YS/kb-qa --base https://modelscope.cn
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

UA = {"User-Agent": "kb-deploy/1.0"}


def list_index_files(base: str, repo: str, prefix: str = "data/index") -> list[dict]:
    """仓库里的 data/index 文件列表（跳过 profile）。"""
    url = (f"{base}/api/v1/studio/{repo}/repo/files"
           "?Revision=master&Recursive=True")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as res:
        data = json.loads(res.read().decode("utf-8"))
    rows = ((data.get("Data") or {}).get("Files")) or []
    out = []
    for row in rows:
        path = str(row.get("Path") or "")
        size = int(row.get("Size") or 0)
        if not path.startswith(prefix + "/") or size <= 0:
            continue
        if "/profile/" in path or path.endswith("/profile"):
            continue
        out.append({"path": path, "size": size})
    return out


def download_file(base: str, repo: str, remote: str, dest: Path) -> bool:
    """按路径下载单个文件；这个接口给的是 LFS 解析后的内容。"""
    url = (f"{base}/api/v1/studio/{repo}/repo"
           f"?FilePath={urllib.parse.quote(remote)}&Revision=master")
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as res, \
                dest.open("wb") as fh:
            total = int(res.headers.get("Content-Length") or 0)
            done = 0
            while True:
                buf = res.read(1024 * 256)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
    except Exception as exc:  # noqa: BLE001
        print(f"  [X] {remote}: {type(exc).__name__}: {exc}")
        return False
    cost = time.time() - t0
    print(f"  [OK] {remote}  {dest.stat().st_size:,} B  {cost:.1f}s")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="从创空间仓库重拉检索索引")
    ap.add_argument("dest", nargs="?", default="data/index", help="目标目录")
    ap.add_argument("--repo", required=True, help="创空间仓库，形如 owner/space")
    ap.add_argument("--base", default="https://modelscope.cn", help="接口根地址")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    dest_root = Path(args.dest)
    print(f"仓库   : {args.repo}")
    print(f"目标   : {dest_root.resolve()}")

    try:
        rows = list_index_files(base, args.repo)
    except Exception as exc:  # noqa: BLE001
        print(f"[X] 取不到文件列表：{type(exc).__name__}: {exc}")
        return 1
    if not rows:
        print("[X] 仓库里没有 data/index 下的文件")
        return 1

    print(f"共 {len(rows)} 个文件，开始下载：")
    ok = True
    for row in rows:
        name = Path(row["path"]).name
        if not download_file(base, args.repo, row["path"], dest_root / name):
            ok = False
        elif (dest_root / name).stat().st_size != row["size"]:
            print(f"  [!] {name} 大小对不上：{row['size']} != {(dest_root / name).stat().st_size}")
            ok = False

    total = sum(p.stat().st_size for p in dest_root.glob("*") if p.is_file())
    print(f"完成: {ok}   目录合计 {total / 1024**2:.1f} MB")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
