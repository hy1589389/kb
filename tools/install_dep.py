"""下 wheel 并解压到指定目录（这台机器 pip 装不了，只能手动）。

按平台和 ABI 挑选正确的 wheel：优先 cp313/cp312 精确匹配，其次 abi3（跨版本），
最后才退到 any。早期版本直接取 wheels[0]，结果在 Windows 上抓到了 macos 的包。

用法：python install_dep.py <包名> <目标目录> [--version X.Y.Z]
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UA = {"User-Agent": "dsh"}


def pick_wheel(urls: list[dict], want_version: str | None) -> dict | None:
    if want_version:
        urls = [u for u in urls if u["filename"].find(want_version) >= 0]
    wheels = [u for u in urls if u["packagetype"] == "bdist_wheel"]
    if not wheels:
        return None

    def score(w: dict) -> tuple:
        fn = w["filename"]
        return (
            0 if "win_amd64" in fn else 1,
            0 if "cp313" in fn else (1 if "abi3" in fn else 2),
            -w["size"],
        )

    return sorted(wheels, key=score)[0]


def main() -> int:
    pkg = sys.argv[1]
    target = Path(sys.argv[2]).resolve()
    want_version = None
    if "--version" in sys.argv:
        want_version = sys.argv[sys.argv.index("--version") + 1]

    meta = json.load(urllib.request.urlopen(
        urllib.request.Request(f"https://pypi.org/pypi/{pkg}/json", headers=UA), timeout=60))
    chosen = pick_wheel(meta["urls"], want_version)
    if chosen is None:
        print(f"[X] {pkg} 没有可用的 wheel")
        return 1

    print(f"包     : {pkg} {meta['info']['version']}")
    print(f"选择   : {chosen['filename']}  ({chosen['size'] / 1024**2:.1f} MB)")
    print(f"目标   : {target}")

    blob = urllib.request.urlopen(
        urllib.request.Request(chosen["url"], headers=UA), timeout=300).read()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = zf.namelist()
        zf.extractall(target)
    print(f"解压 {len(names)} 个文件")

    sys.path.insert(0, str(target))
    import importlib

    mod = importlib.import_module(pkg)
    print(f"导入成功: {pkg} {getattr(mod, '__version__', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
