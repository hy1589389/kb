"""把检索结果回链到语料源文件，并交给编辑器打开。

索引里的 chunk 只记 `repo` + `rel_path`（相对语料仓库根）和行号，真实文件在
`<raw_dir>/<repo>/<rel_path>`。本模块负责这段路径拼接、越界校验和编辑器调用。

三个函数是 Web 层（`/api/open`）和 CLI（`search --open N`）共用的：

    resolve_source_path(chunk, raw_dir)  ->  Path | None
    detect_editor()                      ->  str | None     （"code" / "notepad"）
    open_in_editor(path, line=1)         ->  (bool, str)

## 安全：不能让 `rel_path` 逃出 raw_dir

`rel_path` 来自语料文件，不是用户输入，但索引文件可以在本机被改。如果直接
`raw_dir / repo / rel_path` 再打开，一条 `rel_path = "../../../../Windows/System32/calc.exe"`
就能让 Web 接口变成任意文件启动器。所以这里：

1. 拒绝绝对路径和带盘符的路径（`C:foo` 这种盘符相对路径也算）；
2. `Path.resolve()` 消掉 `..` 和符号链接；
3. `is_relative_to(raw_dir)` 再判一次最终落点。

`..` 只在**没跑出去**时才允许（`docs/../README.md` 仍解析到 raw_dir 内部）。

## 为什么用 Popen 而不是 run

`notepad.exe <file>` 会一直跑到用户关窗，用 `subprocess.run` 会把调用方
（uvicorn 的请求线程 / CLI）一起挂住。`code -g file:line` 一般几百毫秒就返回，
但两者统一用 Popen，语义一致：**能拉起来就算成功**，不等待退出，也不看退出码。
代价是编辑器自己报的错（比如 `code` 参数写错）不会回传，只能从 `command`
字段看出到底执行了什么。
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "EDITOR_CANDIDATES",
    "LINE_AWARE_EDITORS",
    "detect_editor",
    "open_in_editor",
    "resolve_source_path",
]

# 回退顺序：code 能跳行，notepad 只能打开文件。
EDITOR_CANDIDATES: tuple[str, ...] = ("code", "notepad")

# 支持 `-g file:line` 的编辑器。不在这个集合里的只打开文件。
LINE_AWARE_EDITORS: tuple[str, ...] = ("code",)


def _field(chunk: Any, name: str, default: Any = None) -> Any:
    """同时支持 `SearchHit`（属性）和 dict（键）两种输入。"""
    if isinstance(chunk, dict):
        return chunk.get(name, default)
    return getattr(chunk, name, default)


@lru_cache(maxsize=64)
def _subdirs_of(base_text: str) -> tuple[Path, ...]:
    """`base` 的一级子目录，只算一次。

    旧索引（多集合改造前建的）里没有 `collection` 字段，只能靠「把 raw_dir 的
    一级子目录都当候选」来猜布局。每条检索结果都要猜一次，而一次 `/api/search`
    有 5~10 条结果 —— 实测单条 4.7 ms 里大头就是这里反复 `iterdir()` 加对不存在
    的路径做 `resolve()`。

    缓存的是**目录列表**，不是文件状态：文件后来新增/删除仍然被 `is_file()` 看到，
    只有「运行期间新增一级集合目录」需要重启才能发现。`maxsize=64` 是给测试留的
    余量，测试会来回换临时目录。
    """
    try:
        return tuple(sorted(p for p in Path(base_text).iterdir() if p.is_dir()))
    except OSError:
        return ()


@lru_cache(maxsize=64)
def _resolved_base(raw_text: str) -> Path:
    """`raw_dir` 的 resolve 结果。同一个服务的 raw_dir 是常量。"""
    return Path(raw_text).resolve()


@lru_cache(maxsize=512)
def _has_repo_dir(base_text: str, repo: str) -> bool:
    """`<base>/<repo>` 是不是目录。

    布局判据（见下面 `resolve_source_path` 的注释）对每条结果都要问一次，
    而 repo 的取值只有语料仓库那几个 —— 缓存后单个 chunk 的探测次数从
    「候选数 × 2」降到 2 次。
    """
    if not repo:
        return False
    try:
        return (Path(base_text) / repo).is_dir()
    except OSError:
        return False


def resolve_source_path(chunk: Any, raw_dir: Path | str) -> Path | None:
    """chunk（SearchHit 或 dict）-> 源文件的绝对路径；取不到返回 None。

    返回 None 的四种情况：没有 `rel_path`、`rel_path` 是绝对/带盘符路径、
    解析后落在 `raw_dir` 之外、文件不存在。调用方只需要区分「有/没有」，
    具体原因写进日志没有意义 —— 用户能做的都是换一个 chunk。
    """
    base = _resolved_base(str(raw_dir))

    rel_text = str(_field(chunk, "rel_path", "") or "").strip()
    if not rel_text:
        return None

    rel = Path(rel_text)
    # Windows 上 `C:foo` 不是绝对路径但有盘符，拼出来会跑出 base，一并拒绝。
    if rel.is_absolute() or rel.drive or rel.anchor:
        return None

    repo_text = str(_field(chunk, "repo", "") or "").strip()
    repo: Path | None = None
    if repo_text:
        repo = Path(repo_text)
        if repo.is_absolute() or repo.drive or repo.anchor:
            return None

    # 多集合之后有三种布局，都要支持：
    #   Markdown 集合：<raw_dir>/<collection>/<repo>/<rel_path>
    #   PDF 集合：    <raw_dir>/<collection>/<rel_path>（平铺，repo 只是文件名）
    #   旧单集合布局：<raw_dir>/<repo>/<rel_path>（没有 collection 字段时）
    #
    # 判据是「<基准目录>/<repo> 是不是目录」，而不是「哪个文件存在」——
    # 后者在集合根下恰好有同名文件时会拼错。
    bases: list[Path] = []
    collection = str(_field(chunk, "collection", "") or "").strip()
    if collection and _has_repo_dir(str(base), collection):
        bases.append(base / collection)
    elif not collection:
        # 旧索引（多集合改造前建的）里没有 collection 字段，而语料已经挪到了
        # <raw_dir>/<collection>/ 下。这里把一级子目录也当候选 —— 不这么做的话
        # Web 的 /api/open 直接查原始 meta 字典时会拼不出路径。
        bases.extend(_subdirs_of(str(base)))
    bases.append(base)

    for b in bases:
        use_repo = repo is not None and _has_repo_dir(str(b), repo_text)
        candidate = b / repo / rel if use_repo else b / rel
        # 大多数候选是「猜错布局」的空路径。先做一次最便宜的 is_file() 挡掉，
        # 只有真存在的才付 resolve() 的代价（Windows 上它会逐级碰文件系统）。
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        # resolve() 之后仍然要判：`..`、符号链接、盘符跳转都在这一步收敛。
        if not resolved.is_relative_to(base):
            continue
        if resolved.is_file():
            return resolved
    return None


def _find_executable(name: str | None) -> str | None:
    """按命令名或绝对路径找可执行文件；找不到返回 None。

    只走 `shutil.which`，不猜安装目录：本机 VS Code 在 PATH 上
    （`D:\\A\\Microsoft VS Code\\bin\\code.cmd`），猜路径的代码换个机器就失效，
    还会让测试没法用 mock 把它关掉。要指定特殊路径时用 `--editor`。
    """
    if not name:
        return None
    found = shutil.which(str(name))
    return found or None


def detect_editor() -> str | None:
    """返回可用的编辑器命令名（'code' / 'notepad'），都没有返回 None。"""
    for name in EDITOR_CANDIDATES:
        if _find_executable(name):
            return name
    return None


def _describe(argv: list[str], editor: str, line: int) -> str:
    """人能看懂、也能直接粘到终端里重跑的命令说明。"""
    parts: list[str] = []
    for i, arg in enumerate(argv):
        # 第 0 个是 exe 路径，其余参数带空格/冒号时加引号，方便复制
        parts.append(arg if i == 0 else f'"{arg}"' if " " in arg else arg)
    text = " ".join(parts)
    if editor.lower() not in LINE_AWARE_EDITORS:
        text += f"（{editor} 不支持跳行，只打开文件，请手动定位到第 {line} 行）"
    return text


def open_in_editor(path: Path, line: int = 1, *,
                   editor: str | None = None) -> tuple[bool, str]:
    """打开文件并跳到指定行。返回 (成功与否, 实际执行的命令说明)。

    `editor` 传 None 时自动探测；也可以传绝对路径（如 PowerShell 里
    `Get-Command code` 给出的 `...\\code.cmd`）。行号容错：0 / 负数 / 非数字
    一律当第 1 行，不抛异常 —— 这一层让命令行参数和索引脏数据都炸不掉服务。

    本函数不做根目录校验（它拿不到 `raw_dir`）：调用方必须先过
    `resolve_source_path`，不要把外部输入直接喂进来。
    """
    path = Path(path)
    if not path.is_file():
        return False, f"源文件不存在：{path}"

    name = (editor or detect_editor() or "").strip()
    if not name:
        return False, ("没有找到可用的编辑器：PATH 上既没有 code 也没有 notepad。"
                       "用 --editor 指定可执行文件路径。")

    exe = _find_executable(name)
    if exe is None:
        return False, f"编辑器不可用：找不到可执行文件「{name}」"

    try:
        line_no = max(1, int(line or 1))
    except (TypeError, ValueError):
        line_no = 1

    if Path(name).name.lower().startswith("code"):
        argv = [exe, "-g", f"{path}:{line_no}"]
    else:
        # notepad 没有跳行参数，只能把文件路径给它
        argv = [exe, str(path)]

    command = _describe(argv, name, line_no)
    try:
        subprocess.Popen(argv)  # noqa: S603 - 参数是列表，不经 shell
    except OSError as exc:
        return False, f"启动编辑器失败：{exc}（命令：{command}）"
    return True, command
