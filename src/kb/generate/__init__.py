"""Phase 4 生成层：检索 -> 大模型 -> 带引用的答案。

对外暴露：

    from kb.generate import Answerer, Answer, Citation
    from kb.generate import DeepSeekClient, FakeLLM, LLMError, KeyNotFoundError
    from kb.generate import load_api_key, build_messages, CITE_FORMAT

API key 的读取顺序是 环境变量 -> 项目根 .env -> Windows 注册表，实现和
安全约束见 `llm.py` 的模块注释。导入本包不会发起任何网络请求，也不会
在 import 阶段读 key —— key 只在构造 `DeepSeekClient` 时才去找。
"""

from __future__ import annotations

from .cite import count_uncited, extract_refs, invalid_refs, split_sentences
from .llm import (
    DeepSeekClient,
    FakeLLM,
    KeyNotFoundError,
    LLMClient,
    LLMError,
    load_api_key,
)
from .pipeline import Answerer
from .prompt import CITE_FORMAT, INSUFFICIENT, build_messages
from .types import Answer, Citation

__all__ = [
    "Answer",
    "Answerer",
    "CITE_FORMAT",
    "Citation",
    "DeepSeekClient",
    "FakeLLM",
    "INSUFFICIENT",
    "KeyNotFoundError",
    "LLMClient",
    "LLMError",
    "build_messages",
    "count_uncited",
    "extract_refs",
    "invalid_refs",
    "load_api_key",
    "split_sentences",
]
