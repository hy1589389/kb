"""配置加载：零依赖，用标准库 tomllib（Python 3.11+）读取 config.toml。

两个 dataclass 对应 TOML 里的两段互不相干的配置：

    IngestConfig     [path] / [chunk] / [filter]        Phase 1 语料入库
    RetrievalConfig  [retrieval]                        Phase 2 检索

`IngestConfig.load()` 会把 `[retrieval]` 一并解析挂在 `.retrieval` 上，
所以调用方读一次 TOML 就能拿到全部配置；`RetrievalConfig.load()` 单独
用也可以。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_SKIP_DIRS = [
    ".git",
    ".github",
    ".vuepress",
    ".idea",
    ".vscode",
    "node_modules",
    "assets",
    "images",
    "img",
    "static",
    "public",
    "media",
    "fonts",
    "dist",
    "build",
]

DEFAULT_SKIP_FILES = [
    "license",
    "license.md",
    "changelog",
    "changelog.md",
    "contributing.md",
    "code_of_conduct.md",
    "notice",
]

MARKDOWN_SUFFIXES = (".md", ".markdown", ".mdx")


@dataclass
class ChunkConfig:
    """切块参数。这几个数字后续要用评测集做对照实验，别拍脑袋定。"""

    max_chars: int = 1200  # 单块目标上限（字符）
    min_chars: int = 120  # 低于此值的块，尝试与相邻块合并
    drop_below: int = 40  # 合并后仍低于此值则丢弃（通常是残留占位）
    overlap: int = 120  # 超长章节二次切分时，块首补上的上文字符数
    code_block_max_chars: int = 2400  # 代码块超过此值就在空行处拆开（见 chunker 注释）


@dataclass
class RetrievalConfig:
    """Phase 2 检索参数。

    `IngestConfig` 与它是并列关系，不是包含关系 —— Phase 1 的代码只认
    `IngestConfig`，新增这一段不会影响 `python ingest.py ingest`。

    `IngestConfig.load()` 会顺手把 `[retrieval]` 段解析进来挂在
    `.retrieval` 上，所以 Phase 1 的调用方不用改；检索代码直接用
    `IngestConfig.load(p).retrieval` 即可，不必再读一遍 TOML。
    """

    index_dir: Path = Path("data/index")
    model_dir: Path = Path(r"G:\kb\models\bge-m3")
    reranker_dir: Path = Path(r"G:\kb\models\bge-reranker-v2-m3")
    embed_batch_size: int = 16
    embed_max_length: int = 512
    dense_top_k: int = 50
    sparse_top_k: int = 50
    fusion_top_k: int = 50
    final_top_k: int = 10
    rrf_k: int = 60
    use_reranker: bool = False

    @classmethod
    def load(cls, config_path: Path | None = None) -> "RetrievalConfig":
        cfg = cls()
        if config_path is None or not Path(config_path).exists():
            return cfg
        data = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        cfg.apply_section(data.get("retrieval", {}))
        return cfg

    def apply_section(self, section: dict) -> "RetrievalConfig":
        """把 TOML 的 [retrieval] 段套到默认值上。

        类型按**当前字段值的类型**推断，而不是用 typing 注解：本文件开了
        `from __future__ import annotations`，`dataclasses.fields()` 拿到的
        `f.type` 是字符串 `"Path"` / `"int"` / `"bool"`，`is` 比较不可靠。

        路径字段允许留空字符串：TOML 里写 `model_dir = ""` 表示「这台机器
        上没有这个模型」，此时置为 None，让调用方给出「没配 model_dir」的
        报错，而不是拿去拼出一个不存在的目录路径。
        """
        for f in fields(self):
            if f.name not in section:
                continue
            default = self.__dict__[f.name]
            raw = section[f.name]
            if isinstance(default, bool):
                value: object = bool(raw)
            elif isinstance(default, Path):
                text = str(raw).strip()
                if not text:
                    value = None
                else:
                    value = Path(text)
            elif isinstance(default, int):
                value = int(raw)
            else:
                value = raw
            self.__dict__[f.name] = value
        return self

    def to_retriever_cfg(self) -> dict:
        """转成 `Retriever(cfg=...)` 认的 dict。"""
        return {
            "dense_top_k": self.dense_top_k,
            "sparse_top_k": self.sparse_top_k,
            "fusion_top_k": self.fusion_top_k,
            "final_top_k": self.final_top_k,
            "rrf_k": self.rrf_k,
            "use_reranker": self.use_reranker,
            "embed_batch_size": self.embed_batch_size,
            "embed_max_length": self.embed_max_length,
            "model_dir": self.model_dir,
            "reranker_dir": self.reranker_dir,
        }

    def resolve(self, project_root: Path) -> "RetrievalConfig":
        if self.index_dir is not None and not self.index_dir.is_absolute():
            self.index_dir = (project_root / self.index_dir).resolve()
        return self


@dataclass
class GenerateConfig:
    """Phase 4 生成参数。对应 config.toml 的 `[generate]` 段。

    `api_key_env` 存的是**环境变量名**，不是 key 本身 —— 密钥只从环境变量 /
    .env / 注册表读，配置文件里永远不出现 key。

    `min_top_score` 填 0（或留空）表示关闭。注意它只在单路检索下有意义：
    混合检索的 score 是 RRF 融合分（约 0.016~0.033），和余弦相似度不在一个
    量纲上，拿绝对值当阈值会把所有查询都拒掉。`Answerer` 在 source 为
    hybrid / rerank 时会显式跳过这个阈值并打日志。
    """

    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/chat/completions"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 1024
    max_context_chars: int = 6000
    max_chunk_chars: int = 1500
    top_k: int = 5
    min_top_score: float = 0.0
    timeout: float = 60.0

    @classmethod
    def load(cls, config_path: Path | None = None) -> "GenerateConfig":
        cfg = cls()
        if config_path is None or not Path(config_path).exists():
            return cfg
        data = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        cfg.apply_section(data.get("generate", {}))
        return cfg

    def apply_section(self, section: dict) -> "GenerateConfig":
        """把 TOML 的 [generate] 段套到默认值上。

        类型推断方式与 `RetrievalConfig.apply_section` 相同（按当前字段值的
        类型），额外补了 float：`temperature` 默认 0.1 是 float，而 TOML 里
        写 `0` 会解析成 int，不转的话下游拿到的类型会飘。
        """
        for f in fields(self):
            if f.name not in section:
                continue
            default = self.__dict__[f.name]
            raw = section[f.name]
            if isinstance(default, bool):
                value: object = bool(raw)
            elif isinstance(default, Path):
                text = str(raw).strip()
                value = None if not text else Path(text)
            elif isinstance(default, int):
                value = int(raw)
            elif isinstance(default, float):
                value = float(raw)
            else:
                value = raw
            self.__dict__[f.name] = value
        return self

    def to_answerer_cfg(self) -> dict:
        """转成 `Answerer(cfg=...)` 认的 dict。

        不含 model / base_url / api_key_env：那三个是 `DeepSeekClient`
        的构造参数，Answerer 用不到。
        """
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_context_chars": self.max_context_chars,
            "max_chunk_chars": self.max_chunk_chars,
            "top_k": self.top_k,
            "min_top_score": self.min_top_score,
        }


@dataclass
class IngestConfig:
    raw_dir: Path = Path("data/raw")
    out_dir: Path = Path("data/processed")
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    skip_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_SKIP_DIRS))
    skip_files: list[str] = field(default_factory=lambda: list(DEFAULT_SKIP_FILES))
    # Phase 2 的检索配置挂在这里，方便一个入口拿到全部配置。
    # 默认值来自 RetrievalConfig()，load() 会用 TOML 的 [retrieval] 段覆盖。
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    # Phase 4 的生成配置，同理。新增这两个字段不影响 Phase 1 的行为：
    # 它们只在 load() 里被 TOML 覆盖，Phase 1 代码根本不读。
    generate: GenerateConfig = field(default_factory=GenerateConfig)

    @classmethod
    def load(cls, config_path: Path | None = None) -> "IngestConfig":
        cfg = cls()
        if config_path is None or not Path(config_path).exists():
            return cfg

        data = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        path_sec = data.get("path", {})
        if "raw_dir" in path_sec:
            cfg.raw_dir = Path(path_sec["raw_dir"])
        if "out_dir" in path_sec:
            cfg.out_dir = Path(path_sec["out_dir"])

        chunk_sec = data.get("chunk", {})
        for f in fields(ChunkConfig):
            if f.name in chunk_sec:
                setattr(cfg.chunk, f.name, int(chunk_sec[f.name]))

        filter_sec = data.get("filter", {})
        if "skip_dirs" in filter_sec:
            cfg.skip_dirs = [str(x) for x in filter_sec["skip_dirs"]]
        if "skip_files" in filter_sec:
            cfg.skip_files = [str(x) for x in filter_sec["skip_files"]]

        # Phase 2：同一份 TOML 里的 [retrieval] 段
        cfg.retrieval.apply_section(data.get("retrieval", {}))
        # Phase 4：同一份 TOML 里的 [generate] 段
        cfg.generate.apply_section(data.get("generate", {}))
        return cfg

    def resolve(self, project_root: Path) -> "IngestConfig":
        """把相对路径按项目根目录展开成绝对路径。"""
        if not self.raw_dir.is_absolute():
            self.raw_dir = (project_root / self.raw_dir).resolve()
        if not self.out_dir.is_absolute():
            self.out_dir = (project_root / self.out_dir).resolve()
        return self
