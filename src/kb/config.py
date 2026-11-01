"""配置加载：零依赖，用标准库 tomllib（Python 3.11+）读取 config.toml。

每个 dataclass 对应 TOML 里的一段，彼此互不相干：

    IngestConfig      [path] / [chunk] / [filter]        Phase 1 语料入库
    CollectionsConfig [collections] (+ 每集合一张表)      多集合隔离
    RetrievalConfig   [retrieval]                        Phase 2 检索
    GenerateConfig    [generate]                         Phase 4 生成
    WebConfig         [web]                              Phase 5 Web 界面

`IngestConfig.load()` 会把后面几段一并解析，挂在 `.collections` /
`.retrieval` / `.generate` / `.web` 上，所以调用方读一次 TOML 就能拿到全部
配置；各自的 `load()` 单独用也可以。
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
PDF_SUFFIXES = (".pdf",)

# 集合的解析器：只认 Markdown / 只认 PDF / 按扩展名分派
PARSER_MARKDOWN = "markdown"
PARSER_PDF = "pdf"
PARSER_AUTO = "auto"
PARSERS = (PARSER_MARKDOWN, PARSER_PDF, PARSER_AUTO)

# 生成策略：enabled = 正常走大模型；disabled = 只返回检索原文（隐私集合用）
GENERATION_ENABLED = "enabled"
GENERATION_DISABLED = "disabled"
GENERATIONS = (GENERATION_ENABLED, GENERATION_DISABLED)


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
class WebConfig:
    """Phase 5 Web 界面参数，对应 config.toml 的 `[web]` 段。

    `editor` 留空表示自动探测：先找 `code`（能跳行），再退到 `notepad`
    （只能打开文件）。填了值就固定用它，写命令名（`code`）或可执行文件
    绝对路径都可以。

    这里不放开 `raw_dir`：语料根目录只有 `[path].raw_dir` 一个来源，
    两处配置同一个目录迟早会不一致。要换目录用 `--raw-dir`。
    """

    host: str = "127.0.0.1"
    port: int = 8000
    editor: str = ""

    @classmethod
    def load(cls, config_path: Path | None = None) -> "WebConfig":
        cfg = cls()
        if config_path is None or not Path(config_path).exists():
            return cfg
        data = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        cfg.apply_section(data.get("web", {}))
        return cfg

    def apply_section(self, section: dict) -> "WebConfig":
        """把 TOML 的 [web] 段套到默认值上，类型推断方式同 RetrievalConfig。"""
        for f in fields(self):
            if f.name not in section:
                continue
            default = self.__dict__[f.name]
            raw = section[f.name]
            if isinstance(default, bool):
                value: object = bool(raw)
            elif isinstance(default, int):
                value = int(raw)
            elif isinstance(default, float):
                value = float(raw)
            else:
                value = str(raw)
            self.__dict__[f.name] = value
        return self


@dataclass
class CollectionConfig:
    """一个集合（collection）的配置，对应 `[collections.<name>]` 一张表。

    集合是**物理隔离**的单位：各自的 `raw_dir`、`index_dir`、解析器，互不
    读写。这样「上传简历」只会往 `profile/` 里写，重建索引也只重建
    `data/index/profile/`，不会碰到 18k 块的技术文档索引。

    `index_dir` 为 `data/index` 的那个集合就是历史包袱里的 tech：它的索引
    继续放在旧位置，不迁移、不重建。

    `parser` 取值：
        markdown 只认 .md/.markdown/.mdx
        pdf      只认 .pdf
        auto     按扩展名分派（论文、简历这种混合目录用）

    `generation` 取值 `enabled` / `disabled`。`disabled` 表示该集合的问答
    不调云端大模型，只返回检索到的原文 —— 简历里有手机号和邮箱，不能发出去。
    这个策略由 Phase 4/5 读取，Phase 1 只负责把它存进配置。

    `weight` 是跨集合 RRF 融合时这一路的权重（默认 1.0）。个人材料命中时
    通常希望排在技术文档后面，调小即可。
    """

    name: str = ""
    raw_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")
    parser: str = PARSER_MARKDOWN
    generation: str = GENERATION_ENABLED
    weight: float = 1.0

    def apply_section(self, section: dict) -> "CollectionConfig":
        """把 `[collections.<name>]` 套到默认值上，类型按当前字段值推断。"""
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
                value = str(raw)
            self.__dict__[f.name] = value

        self.parser = str(self.parser or PARSER_AUTO).strip().lower()
        if self.parser not in PARSERS:
            raise ValueError(
                f"集合 {self.name or '?'} 的 parser={self.parser!r} 不认识，"
                f"可选：{' / '.join(PARSERS)}"
            )
        self.generation = str(self.generation or GENERATION_ENABLED).strip().lower()
        if self.generation not in GENERATIONS:
            raise ValueError(
                f"集合 {self.name or '?'} 的 generation={self.generation!r} 不认识，"
                f"可选：{' / '.join(GENERATIONS)}"
            )
        return self

    def resolve(self, project_root: Path) -> "CollectionConfig":
        """把相对路径按项目根展开成绝对路径。"""
        for fname in ("raw_dir", "index_dir"):
            value = getattr(self, fname)
            if value is not None and not Path(value).is_absolute():
                setattr(self, fname, (Path(project_root) / Path(value)).resolve())
        return self

    def to_retriever_cfg(self) -> dict:
        """转成 `Retriever(cfg={"collections": ...})` 里按集合配权重的那部分。"""
        return {"weight": float(self.weight)}


def _legacy_collection() -> CollectionConfig:
    """没有 `[collections]` 段时的单集合默认值，与旧版行为一致。"""
    return CollectionConfig(
        name="tech",
        raw_dir=Path("data/raw"),
        index_dir=Path("data/index"),
        parser=PARSER_MARKDOWN,
    )


def _as_name_list(value: object) -> list[str]:
    """`default_search` 既接受 TOML 数组，也接受 "tech,profile" 这种逗号串。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_as_name_list(item))
        return out
    return [str(value).strip()]


@dataclass
class CollectionsConfig:
    """`[collections]` 段：默认查哪些集合 + 每个集合一张表。

    TOML 长这样：

        [collections]
        default_search = ["tech"]

        [collections.tech]
        raw_dir = "data/raw/tech"
        index_dir = "data/index"
        parser = "markdown"

        [collections.profile]
        raw_dir = "data/raw/profile"
        index_dir = "data/index/profile"
        parser = "auto"
        generation = "disabled"

    **向后兼容**：`config.toml` 里没有 `[collections]`（或只有空的
    `[collections]`）时，等价于只有一个 `tech` 集合，
    `raw_dir="data/raw"`、`index_dir="data/index"`、`parser="markdown"`，
    即旧版本的行为。
    """

    default_search: list[str] = field(default_factory=lambda: ["tech"])
    items: dict[str, CollectionConfig] = field(
        default_factory=lambda: {"tech": _legacy_collection()}
    )

    @classmethod
    def load(cls, config_path: Path | None = None) -> "CollectionsConfig":
        cfg = cls()
        if config_path is None or not Path(config_path).exists():
            return cfg
        data = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        cfg.apply_section(data.get("collections", {}))
        return cfg

    def apply_section(self, section: dict) -> "CollectionsConfig":
        section = dict(section or {})
        if not section:
            return self

        wanted: list[str] = []
        entries: dict[str, CollectionConfig] = {}
        for key, value in section.items():
            if key == "default_search":
                wanted = _as_name_list(value)
            elif isinstance(value, dict):
                coll = CollectionConfig(name=str(key))
                coll.apply_section(value)
                entries[str(key)] = coll
            else:
                raise ValueError(
                    f"[collections] 下无法识别的配置项：{key} = {value!r}。"
                    f"每集合的配置要写成 [collections.{key}] 一张表。"
                )

        if not entries:
            # 只有 default_search、没有任何集合表：当成旧版本的单 tech
            return self

        self.items = entries
        # 只保留真实存在的集合名；一个都对不上就退回「全部集合」
        names = [n for n in wanted if n in entries]
        self.default_search = names or list(entries)
        return self

    # ------------------------------------------------------------------ 读取

    def names(self) -> list[str]:
        return list(self.items)

    def get(self, name: str) -> CollectionConfig:
        try:
            return self.items[name]
        except KeyError:
            raise KeyError(
                f"配置里没有集合 {name!r}，现有：{', '.join(self.names()) or '(空)'}"
            ) from None

    def select(self, names: list[str] | None = None) -> list[CollectionConfig]:
        """按名字挑集合；留空用 `default_search`。名字不存在直接报错。"""
        chosen = [n for n in (names if names is not None else self.default_search) if n]
        if not chosen:
            chosen = self.names()
        missing = [n for n in chosen if n not in self.items]
        if missing:
            raise KeyError(
                f"配置里没有集合 {', '.join(repr(m) for m in missing)}，"
                f"现有：{', '.join(self.names()) or '(空)'}"
            )
        out: list[CollectionConfig] = []
        for name in chosen:
            if name not in {c.name for c in out}:
                out.append(self.items[name])
        return out

    def index_dirs(self, names: list[str] | None = None) -> dict[str, Path]:
        """`{集合名: 索引目录}`，给 `Retriever` 的多索引模式用。"""
        return {c.name: c.index_dir for c in self.select(names)}

    def weight_map(self, names: list[str] | None = None) -> dict[str, float]:
        return {c.name: float(c.weight) for c in self.select(names)}

    def resolve(self, project_root: Path) -> "CollectionsConfig":
        for coll in self.items.values():
            coll.resolve(project_root)
        return self


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
    # Phase 5 的 Web 配置，同理。
    web: WebConfig = field(default_factory=WebConfig)
    # 多集合配置。没有 [collections] 段时等于「只有一个 tech」，
    # raw_dir / index_dir 与上面的 raw_dir / 默认索引目录一致。
    collections: CollectionsConfig = field(default_factory=CollectionsConfig)

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
        # Phase 5：同一份 TOML 里的 [web] 段
        cfg.web.apply_section(data.get("web", {}))
        # 多集合：同一份 TOML 里的 [collections] 段（含每集合一张子表）
        cfg.collections.apply_section(data.get("collections", {}))
        return cfg

    def resolve(self, project_root: Path) -> "IngestConfig":
        """把相对路径按项目根目录展开成绝对路径。"""
        if not self.raw_dir.is_absolute():
            self.raw_dir = (project_root / self.raw_dir).resolve()
        if not self.out_dir.is_absolute():
            self.out_dir = (project_root / self.out_dir).resolve()
        self.collections.resolve(project_root)
        return self
