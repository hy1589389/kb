"""大模型客户端：DeepSeek（标准库 urllib）与测试替身。

## 为什么不引 requests

本机 pip 装不上任何包（解包阶段被沙箱拒绝），所以 HTTP 只用标准库
`urllib.request`。代价是要自己处理超时、非 2xx、JSON 解析和错误信息，
但这几件事加起来不到 40 行，换掉一个依赖是划算的。

## 密钥从哪来（按优先级）

    1. 环境变量            <env_name>，默认 DEEPSEEK_API_KEY
    2. 项目根的 .env        KEY=VALUE，# 注释，值可带引号
    3. Windows 注册表       系统级 + 用户级环境变量（仅 win32）

第 3 条不是补丁而是必需：本机 key 设在**系统作用域**，而当前进程是之前
启动的，`os.environ` 里没有它。重新开一个终端就能读到，但让代码自己去
注册表拿更省事，也避免「明明设了却报没找到」。非 Windows 平台直接跳过
这一步，用 `sys.platform` 判断。

## 安全约束（写在这里是为了以后不被人改坏）

  · key 只存在实例属性里，不进日志、不进异常消息、不进 `repr()`
  · 任何错误信息都过一遍 `_sanitize()`，把 key 替换成 `***`
  · 抛 HTTP 错误时只带状态码和服务端返回体的前 300 字符，
    **不带请求头**（请求头里有 Authorization）
  · `KeyNotFoundError` 的消息只说「没找到、怎么设」，不含任何 key 片段
"""

from __future__ import annotations

import json
import os
import socket
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

__all__ = [
    "LLMError",
    "KeyNotFoundError",
    "LLMClient",
    "DeepSeekClient",
    "FakeLLM",
    "load_api_key",
    "parse_dotenv",
    "DEFAULT_ENV_NAME",
    "DEFAULT_BASE_URL",
    "DEFAULT_ENV_FILE",
]

DEFAULT_ENV_NAME = "DEEPSEEK_API_KEY"
DEFAULT_BASE_URL = "https://api.deepseek.com/chat/completions"

# src/kb/generate/llm.py -> generate -> kb -> src -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# Windows 环境变量的两个作用域。系统级在前：这台机器的 key 就在系统级，
# 用户级同名变量一般不存在，但存在时按 Windows 的惯例让用户级覆盖系统级
# 更合理 —— 这里顺序反过来是因为「系统级里通常就是运维统一配的那份」，
# 而用户的 shell 里如果真设了，第 1 步（os.environ）就已经命中了。
_WINDOWS_ENV_KEYS = (
    ("HKEY_LOCAL_MACHINE", r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ("HKEY_CURRENT_USER", "Environment"),
)


class LLMError(Exception):
    """调用模型失败的统一异常。消息里不会出现 key。"""


class KeyNotFoundError(LLMError):
    """找不到 API key。消息只说明去哪儿设，不含 key 的任何片段。"""


# --------------------------------------------------------------------------
# 密钥读取
# --------------------------------------------------------------------------

def _clean(value: object) -> str:
    """去掉首尾空白和成对引号；空值返回空串。"""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return text


def parse_dotenv(text: str) -> dict[str, str]:
    """解析 `KEY=VALUE` 格式，支持 `#` 注释和引号。

    规则（故意宽松，只覆盖手写 .env 的常见写法）：

        # 整行注释
        KEY=VALUE
        KEY = "VALUE"        # 行尾注释
        export KEY=VALUE

    值里的 `#` 只有在前面有空白时才当注释，避免把 `pa#ss` 这种截断。
    """
    out: dict[str, str] = {}
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if not key:
            continue
        value = value.strip()
        if value[:1] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            if end != -1:
                value = value[1:end]
            else:  # 引号没闭合，按去掉首个引号处理
                value = value[1:]
        else:
            # 行尾注释：只有 ` #` 才算，`#` 紧贴值不算
            for i, ch in enumerate(value):
                if ch == "#" and i > 0 and value[i - 1] in " \t":
                    value = value[:i]
                    break
            value = value.strip()
        value = _clean(value)
        if value:
            out[key] = value
    return out


def _dotenv_candidates() -> list[Path]:
    """`.env` 的候选路径：项目根在前，当前工作目录在后。

    后者是为了「在别的目录里跑脚本」时不至于找不到；去重避免同一个文件
    被解析两次。
    """
    out = [DEFAULT_ENV_FILE]
    cwd_env = Path.cwd() / ".env"
    try:
        if cwd_env.resolve() != Path(DEFAULT_ENV_FILE).resolve():
            out.append(cwd_env)
    except OSError:
        pass
    return out


def _load_from_dotenv(env_name: str) -> str:
    for path in _dotenv_candidates():
        try:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            # 读不了就当没有：权限问题不该让整个 ask 流程崩在密钥读取上，
            # 最终会走到 KeyNotFoundError，消息里会提示怎么设置。
            continue
        value = _clean(parse_dotenv(text).get(env_name, ""))
        if value:
            return value
    return ""


def _read_windows_env(env_name: str) -> str:
    """从 Windows 注册表读系统 / 用户环境变量。非 Windows 返回空串。

    值类型可能是 `REG_EXPAND_SZ`（含 `%USERPROFILE%` 这类引用），需要展开；
    普通 `REG_SZ` 直接返回。
    """
    if sys.platform != "win32":
        return ""
    try:
        import winreg
    except ImportError:  # 理论上不会发生，win32 一定有
        return ""
    for hive_name, sub_key in _WINDOWS_ENV_KEYS:
        hive = getattr(winreg, hive_name, None)
        if hive is None:
            continue
        try:
            with winreg.OpenKey(hive, sub_key, 0, winreg.KEY_READ) as key:
                value, value_type = winreg.QueryValueEx(key, env_name)
        except OSError:
            # 没有这个值 / 没有读权限，继续试下一个作用域
            continue
        text = _clean(value)
        if not text:
            continue
        if value_type == getattr(winreg, "REG_EXPAND_SZ", None):
            text = _clean(os.path.expandvars(text))
        return text
    return ""


def load_api_key(env_name: str = DEFAULT_ENV_NAME) -> str:
    """按 环境变量 -> .env -> Windows 注册表 的顺序取 key。

    找不到抛 `KeyNotFoundError`，消息里只有「怎么设」，没有任何 key 片段。
    """
    value = _clean(os.environ.get(env_name, ""))
    if value:
        return value

    value = _load_from_dotenv(env_name)
    if value:
        return value

    value = _read_windows_env(env_name)
    if value:
        return value

    raise KeyNotFoundError(
        f"没有找到 API key（{env_name}）。已按顺序查过："
        f"环境变量、{DEFAULT_ENV_FILE}、Windows 注册表环境变量。\n"
        f"设置方法（任选一种）：\n"
        f"  1. 当前会话：$env:{env_name} = \"<你的 key>\"\n"
        f"  2. 项目根建 .env，写一行 {env_name}=<你的 key>\n"
        f"  3. 系统环境变量里设置 {env_name} 后重开终端\n"
        f"（报错信息不会显示 key 的内容）"
    )


# --------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------

class LLMClient(ABC):
    """模型客户端接口。只有一个方法：`complete`。"""

    model: str

    @abstractmethod
    def complete(self, messages: list[dict], *, temperature: float = 0.1,
                 max_tokens: int = 1024) -> tuple[str, dict]:
        """返回 `(文本, usage dict)`。失败抛 `LLMError`。"""

    def close(self) -> None:
        """释放资源。默认什么都不做。"""


class DeepSeekClient(LLMClient):
    """DeepSeek Chat Completions 客户端（标准库 urllib）。

    参数
    ----
    model        : 模型名，默认 `deepseek-chat`
    base_url     : 完整接口地址（含 /chat/completions）
    api_key      : 显式传入的 key；不传则按 `load_api_key` 的顺序找
    api_key_env  : 找 key 时用的环境变量名
    timeout      : 单次请求超时（秒）。显式设置，不依赖 urllib 的全局默认
    """

    def __init__(self, model: str = "deepseek-chat",
                 base_url: str = DEFAULT_BASE_URL, *,
                 api_key: str | None = None,
                 api_key_env: str = DEFAULT_ENV_NAME,
                 timeout: float = 60.0):
        if not model:
            raise LLMError("model 不能为空")
        if not base_url:
            raise LLMError("base_url 不能为空")
        self.model = str(model)
        self.base_url = str(base_url)
        self.timeout = float(timeout)
        self.api_key_env = api_key_env
        key = _clean(api_key) if api_key is not None else load_api_key(api_key_env)
        if not key:
            # 显式传了空 key，走和「找不到」同一条路，消息同样不含片段
            raise KeyNotFoundError(
                f"没有找到 API key（{api_key_env}）：显式传入的 api_key 为空。"
            )
        self._api_key = key

    # ------------------------------------------------------------ 表示
    def __repr__(self) -> str:
        # 只放模型和地址：key 永远不进 repr，否则日志里一行 print 就漏了
        return (f"DeepSeekClient(model={self.model!r}, "
                f"base_url={self.base_url!r})")

    __str__ = __repr__

    def _sanitize(self, text: object) -> str:
        """把 key 从任意文本里抹掉。所有对外错误消息都过这一层。"""
        out = "" if text is None else str(text)
        if self._api_key:
            out = out.replace(self._api_key, "***")
        return out

    # ------------------------------------------------------------ 请求
    def complete(self, messages: list[dict], *, temperature: float = 0.1,
                 max_tokens: int = 1024) -> tuple[str, dict]:
        if not messages:
            raise LLMError("messages 为空，拒绝发送请求")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib_request.Request(self.base_url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        # 这是唯一一处出现 key 的地方。请求对象不对外暴露，也不进日志。
        req.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            with urllib_request.urlopen(req, timeout=self.timeout) as resp:
                status = int(getattr(resp, "status", 200) or 200)
                raw = resp.read()
        except urllib_error.HTTPError as exc:
            # 只带状态码和服务端返回体，**不带请求头**
            raise LLMError(self._sanitize(
                f"DeepSeek 接口返回 HTTP {exc.code}：{self._body_snippet(exc)}"
                + ("（key 可能无效或已过期）" if exc.code in (401, 403) else "")
            )) from None
        except (urllib_error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise LLMError(self._sanitize(
                f"请求 DeepSeek 失败（timeout={self.timeout:g}s）：{reason}"
            )) from None

        if not 200 <= status < 300:
            raise LLMError(f"DeepSeek 接口返回 HTTP {status}（非 2xx）")

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMError(self._sanitize(
                f"模型返回的不是合法 JSON：{type(exc).__name__}"
            )) from None

        text = _extract_content(data)
        usage = data.get("usage") if isinstance(data, dict) else None
        return text, dict(usage) if isinstance(usage, dict) else {}

    @staticmethod
    def _body_snippet(exc: urllib_error.HTTPError, limit: int = 300) -> str:
        """读服务端返回体的前若干字符。读失败就只给状态行。"""
        try:
            body = exc.read()
        except Exception:  # noqa: BLE001 - 读错误体本身不该再抛
            return "(无响应体)"
        if not body:
            return "(无响应体)"
        try:
            text = body.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return "(响应体无法解码)"
        text = " ".join(text.split())
        return text[:limit] + ("…" if len(text) > limit else "")

    def close(self) -> None:
        # urllib 没有连接池，没有需要关闭的句柄。保留方法是为了统一接口，
        # 以后换成带 session 的实现时调用方不用改。
        return None


def _extract_content(data: object) -> str:
    """从 chat completions 响应里取 `choices[0].message.content`。"""
    if not isinstance(data, dict):
        raise LLMError("模型返回结构异常：顶层不是对象")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError("模型返回结构异常：没有 choices")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    if not isinstance(content, str):
        raise LLMError("模型返回结构异常：choices[0].message.content 不是字符串")
    return content.strip()


class FakeLLM(LLMClient):
    """测试用客户端。构造时给固定字符串或 `responder(messages)` 可调用对象。

    `responder` 可以返回字符串，也可以返回 `(字符串, usage dict)`。
    每次调用都记进 `self.calls`，测试靠它断言「有没有调模型」「提示词长
    什么样」「temperature 有没有传下去」。
    """

    def __init__(self, responder="", *, model: str = "fake-llm"):
        if not (isinstance(responder, str) or callable(responder)):
            raise TypeError("responder 必须是字符串或可调用对象")
        self._responder = responder
        self.model = model
        self.calls: list[dict] = []
        self.closed = False

    def complete(self, messages: list[dict], *, temperature: float = 0.1,
                 max_tokens: int = 1024) -> tuple[str, dict]:
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        result = self._responder(messages) if callable(self._responder) else self._responder
        if isinstance(result, tuple):
            text, usage = result
        else:
            text, usage = result, {}
        text = "" if text is None else str(text)
        if not isinstance(usage, dict) or not usage:
            # 假 usage：按 4 字符 ≈ 1 token 粗估。数值本身没有意义，
            # 但字段形状和真实响应一致，调用方不用为测试写分支。
            prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
            usage = {
                "prompt_tokens": prompt_chars // 4,
                "completion_tokens": len(text) // 4,
                "total_tokens": (prompt_chars + len(text)) // 4,
            }
        return text, dict(usage)

    def close(self) -> None:
        self.closed = True
