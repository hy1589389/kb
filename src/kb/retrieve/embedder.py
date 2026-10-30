"""嵌入器：零依赖的 HashingEmbedder（自检用）+ bge-m3 的 ONNX 版本。

## 两条实现的定位

`HashingEmbedder` —— 零依赖、确定性、跨进程一致。存在的意义是让整条
流水线（建索引 -> 稠密检索 -> BM25 -> RRF 融合 -> 重排接口）在 2 GB 的
bge-m3 下载完成之前就能跑通和写测试。它**不是**语义模型：用哈希把 token
映射到桶，只能做到「共词越多越相似」。它也不该出现在最终评测里。

`OnnxEmbedder` —— 真正干活的。onnxruntime CPU 后端加载 bge-m3，mean
pooling + L2 归一化。

## 环境依赖（重要，装不上 pip 的机器请照做）

CPU 上跑 onnxruntime 需要 `onnxruntime`（已装 1.27.0）和 `tokenizers`
（**未装**）。本机 pip 在解包阶段会被沙箱拒绝，所以 tokenizers 走手动
解压 wheel：

```powershell
# 1. 下载 wheel（cp310-abi3 = 兼容 CPython 3.10+，3.13 直接可用）
$ver = "0.23.2"
$whl = "tokenizers-$ver-cp310-abi3-win_amd64.whl"
Invoke-WebRequest "https://pypi.org/simple/tokenizers/" -OutFile index.html   # 或在浏览器里找下载页
# 实际下载地址形如：
#   https://files.pythonhosted.org/packages/.../tokenizers-0.23.2-cp310-abi3-win_amd64.whl

# 2. 解压到 work/.pylibs（wheel 就是 zip，改名即可，不需要 pip）
New-Item -ItemType Directory -Force .pylibs | Out-Null
Copy-Item $whl "$env:TEMP\tok.zip"
Expand-Archive "$env:TEMP\tok.zip" -DestinationPath .pylibs -Force

# 3. 验证
python -c "import sys; sys.path.insert(0, '.pylibs'); import tokenizers; print(tokenizers.__version__)"
```

`.pylibs` 的注入是**自动**的：`_ensure_tokenizers()` 会把项目根目录下的
`.pylibs` 插到 `sys.path`，所以你不需要设 PYTHONPATH。放在别处的话设
环境变量 `KB_PYLIBS=<目录>`。完整步骤也写在项目根的 `requirements.md`。

## bge-m3 模型文件

`G:/kb/models/bge-m3` 里应有：

    model.onnx            模型图（权重可能外置）
    model.onnx_data       外置权重（>2 GB 时 onnx 会拆出来，必须和 model.onnx 同目录）
    tokenizer.json        HF fast tokenizer
    config.json           hidden_size / max_position_embeddings
    special_tokens_map.json

`OnnxEmbedder` 在缺文件时抛 FileNotFoundError 并指出缺哪个、去哪儿补，
不会带着半个模型进入静默降级。

## 输入输出名自适应

不同导出脚本产出的 ONNX 图名字不一样（HF Optimum 导出是 `input_ids` /
`attention_mask` / `token_type_ids` + `last_hidden_state`；sentence-
transformers 导出还会多一个 `sentence_embedding`）。所以：

  · 输入：`session.get_inputs()` 拿真实名字，只喂模型声明要的。bge-m3
    的 XLMRoberta 没有 token_type_ids，就不传。
  · 输出：优先 `sentence_embedding`（已池化），其次
    `last_hidden_state`（自己做 mean pooling），再退到唯一输出。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .tokenize import tokenize

__all__ = ["Embedder", "HashingEmbedder", "OnnxEmbedder", "OnnxReranker"]

# 执行后端。默认那份 onnxruntime 只有 CPU；把 onnxruntime-directml 解压到
# <项目根>/.pylibs-dml（或 KB_DML_LIBS 指定的目录）就能跑在显卡上。
#
# 这不是可选的优化，是能不能用的分界。实测 bge-m3（fp32，568M 参数）：
#   CPU         0.8 块/秒  ->  全量 18,276 块要 6.6 小时
#   DirectML    7.6 块/秒  ->  40 分钟   （batch=4，GTX 1650 Ti 4GB）
# 每改一次切块参数就要重跑一遍索引，6.6 小时等于不能迭代。
#
# 注意 DirectML 与 CUDA 无关：它走 DirectX 12，Windows 上不用装 CUDA toolkit
# 就能用上较新的显卡。这比为了 onnxruntime-gpu 去凑 CUDA + cuDNN 划算得多。
_DML_LIBS_ENV = "KB_DML_LIBS"


def _ensure_onnxruntime(prefer_gpu: bool = True):
    """导入 onnxruntime，必要时优先用 DirectML 那份。

    必须在 `import onnxruntime` **之前**把 DML 目录插到 sys.path 最前面：
    已安装的 CPU 版会先被解析到，之后再插 sys.path 不会改已经加载的模块。
    """
    if prefer_gpu:
        candidates = []
        env = os.environ.get(_DML_LIBS_ENV)
        if env:
            candidates.append(Path(env))
        # embedder.py -> retrieve -> kb -> src -> 项目根
        candidates.append(Path(__file__).resolve().parents[3] / ".pylibs-dml")
        for cand in candidates:
            if (cand / "onnxruntime").is_dir():
                sys.path.insert(0, str(cand))
                break

    import onnxruntime as ort

    return ort


def _resolve_providers(ort, prefer_gpu: bool) -> list[str]:
    """按实际可用的后端给出 provider 列表。

    只显式声明真的存在的后端，否则 onnxruntime 会在会话创建时抛错。
    """
    available = ort.get_available_providers()
    if prefer_gpu and "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _ensure_tokenizers():
    """导入 `tokenizers`，失败时给出可操作的报错。

    先试直接导入；失败则把项目内 `.pylibs`（或 `KB_PYLIBS` 指定的目录）
    插到 sys.path 再试一次。这样手动解压的 wheel 无需改任何环境变量。
    """
    try:
        import tokenizers  # noqa: F401
        return tokenizers
    except ImportError:
        pass

    candidates = []
    env = os.environ.get("KB_PYLIBS")
    if env:
        candidates.append(Path(env))
    # embedder.py -> retrieve -> kb -> src -> 项目根
    candidates.append(Path(__file__).resolve().parents[3] / ".pylibs")

    for cand in candidates:
        if cand.is_dir():
            sys.path.insert(0, str(cand))
            try:
                import tokenizers  # noqa: F811
                return tokenizers
            except ImportError:
                sys.path.pop(0)

    raise RuntimeError(
        "缺少 tokenizers，无法加载 ONNX 模型的分词器。\n"
        "本机 pip 装不上，请手动解压 wheel（见 src/kb/retrieve/embedder.py 顶部注释"
        "或项目根 requirements.md）：\n"
        "  1) 下载 tokenizers-0.23.2-cp310-abi3-win_amd64.whl\n"
        "  2) 用 Expand-Archive 解压到  <项目根>\\.pylibs\n"
        "  3) 重跑；或设环境变量 KB_PYLIBS=<解压目录>\n"
        "只想先跑通流水线的话，用 --embedder hashing（零依赖）。"
    )


class Embedder(ABC):
    """嵌入器接口。所有实现必须输出**已 L2 归一化**的 float32 矩阵。"""

    dim: int

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """返回 (len(texts), dim) 的 float32，已 L2 归一化。"""

    def encode_one(self, text: str) -> np.ndarray:
        """单条便捷入口，返回 (dim,)。"""
        return self.encode([text])[0]


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """按行 L2 归一化。零向量保持零向量（不产生 NaN）。

    零向量在余弦检索里没有方向，分数恒为 0，不会误召回；用 max(norm, eps)
    而不是除以 0 是为了让空文本安静地沉底而不是把整个矩阵变成 NaN。
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


class HashingEmbedder(Embedder):
    """零依赖的确定性嵌入：token -> 稳定哈希 -> 桶 -> tf-idf 权重 -> L2。

    设计取舍：

    · **哈希函数用 `zlib.crc32` + `hashlib.blake2b` 的混合，不用内置
      `hash()`。** Python 内置 `hash()` 对 str 加了每进程随机盐
      （PYTHONHASHSEED），同一个词在两个进程里哈希值不同 —— 那意味着
      索引进程和查询进程的向量空间不同，检索结果完全是噪声，而且
      「进程 A 建的索引在进程 B 里查不到东西」这种 bug 极难定位。
      CRC32 只能出 32 位，用 blake2b(digest_size=8) 出 64 位，冲突概率
      在千万级词表下可忽略。

    · **分桶前加符号位。** 只用 `h % dim` 的话，所有 token 都往桶里加正
      数，向量全落在正卦限，任意两条文本的余弦都偏高（实测能到 0.5+），
      区分度被压缩。用哈希的一位做符号（hashing trick 的标准做法），
      期望上不同 token 的贡献相互抵消，随机文本对之间的余弦接近 0。

    · **权重用 sublinear tf × idf。** tf 用 `1 + log(tf)` 压掉重复词的
      线性增长（一块里「缓存」出现 30 次不该是出现 1 次的 30 倍权重）；
      idf 用 `log(1 + N/df)` 把「的」「是」这类字的权重压到接近 0。
      语料统计在 `build_idf()` 里从被索引的文本现算，不依赖外部词表。

    · **bigram 与 unigram 一起进桶**（来自 `.tokenize`）。这保证「深度
      学习」和「机器学习」在只有一个字不同时也能被区分。

    `dim` 默认 512。再小的桶冲突明显变多；再大对这个「共词相似度」模型
    没有实质提升（它不是语义模型，瓶颈不在维度）。
    """

    def __init__(self, dim: int = 512, use_idf: bool = True):
        if dim <= 0:
            raise ValueError(f"dim 必须 > 0，收到 {dim}")
        self.dim = int(dim)
        self.use_idf = use_idf
        self.df: dict[str, int] = {}
        self.n_docs = 0

    # ------------------------------------------------------------- 稳定哈希

    @staticmethod
    def _stable_hash(token: str) -> int:
        """跨进程稳定的 64 位哈希。

        blake2b 出 8 字节（64 位）保证低位分散度；再和 crc32 混合一次，
        让「只差一个字符」的 token（bigram 里大量存在，如「深度」/「度学」）
        在高位上也有差异，减少相邻 token 落同一个桶的概率。
        """
        h = zlib.crc32(token.encode("utf-8"))
        b = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return (int.from_bytes(b, "big") ^ (h * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF

    def _bucket(self, token: str) -> tuple[int, float]:
        """token -> (桶下标, 符号 ±1)。"""
        h = self._stable_hash(token)
        idx = h % self.dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        return idx, sign

    # ------------------------------------------------------------- 语料统计

    def build_idf(self, texts: list[str], sample: int | None = 20000) -> None:
        """从语料统计 df，供 idf 权重使用。

        `sample` 限样本量：18k 块全量统计约几秒，但自检场景不需要那么准，
        抽前 2 万条（或全部）足够。传 None 表示全量。
        """
        self.df = {}
        docs = texts if sample is None else texts[:sample]
        self.n_docs = len(docs)
        for text in docs:
            for token in set(tokenize(text or "")):
                self.df[token] = self.df.get(token, 0) + 1

    def _weight(self, token: str) -> float:
        if not self.use_idf or self.n_docs == 0:
            return 1.0
        df = self.df.get(token, 0)
        if df == 0:
            # 训练语料里没见过的词（查询里的新词）：给最高权重。这相当于
            # 认为它「非常稀有」，对自检场景是合理的；真实 IDF 需要
            # 全量语料统计，这也正是 HashingEmbedder 只用于自检的原因。
            return float(np.log(1.0 + self.n_docs / 0.5))
        return float(np.log(1.0 + self.n_docs / df))

    # ------------------------------------------------------------- 编码

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """`batch_size` 只用于保持接口一致；这里没有模型，逐条处理即可。"""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in tokenize(text or ""):
                counts[token] = counts.get(token, 0) + 1
            if not counts:
                continue  # 空文本 -> 零向量
            vec = out[i]
            for token, tf in counts.items():
                idx, sign = self._bucket(token)
                # sublinear tf：1 + log(tf)
                vec[idx] += sign * (1.0 + np.log(tf)) * self._weight(token)
        return _l2_normalize(out)


class OnnxEmbedder(Embedder):
    """bge-m3 的 ONNX 稠密嵌入（mean pooling + L2 归一化）。

    只依赖 onnxruntime + tokenizers，**不需要 torch / transformers /
    sentence-transformers**。这三个在本机装不上，而 ONNX 路径把「模型定义
    + 权重 + 分词」三个需要的东西都固化成了文件，运行时只要一个推理引擎。
    """

    def __init__(self, model_dir: str | Path, max_length: int = 512,
                 batch_size: int = 16, prefer_gpu: bool = True):
        self.model_dir = Path(model_dir)
        self.max_length = int(max_length)
        self.default_batch_size = int(batch_size)
        self.prefer_gpu = bool(prefer_gpu)
        self._check_files()
        self.tokenizer = self._load_tokenizer()
        self.session, self.input_names, self.output_names = self._load_session()
        self.dim = self._detect_dim()

    # --------------------------------------------------------------- 加载

    def _check_files(self) -> None:
        d = self.model_dir
        if not d.exists():
            raise FileNotFoundError(
                f"模型目录不存在：{d}\n"
                f"bge-m3 需要手动下载（本机 huggingface.co 不通，用 hf-mirror.com）：\n"
                f"  python fetch_model.py --dest {d}\n"
                f"或在浏览器打开 https://hf-mirror.com/BAAI/bge-m3/tree/main "
                f"下载 model.onnx / model.onnx_data / tokenizer.json / config.json / "
                f"special_tokens_map.json 放进该目录。\n"
                f"只想先跑通流水线的话，用 --embedder hashing（零依赖，不需要模型）。"
            )
        required = ["model.onnx", "tokenizer.json", "config.json"]
        missing = [name for name in required if not (d / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"模型目录 {d} 缺少文件：{', '.join(missing)}\n"
                f"模型可能还在下载中（对比 model.onnx_data 的大小是否还在增长）。\n"
                f"补齐方式见 fetch_model.py 或 requirements.md。"
            )

    def _load_tokenizer(self):
        _ensure_tokenizers()
        from tokenizers import Tokenizer

        path = self.model_dir / "tokenizer.json"
        try:
            tok = Tokenizer.from_file(str(path))
        except Exception as exc:  # tokenizers 的异常类型不稳定，统一包装
            raise RuntimeError(f"加载分词器失败：{path}\n原因：{exc}") from exc
        # 截断由我们显式控制（tokenizer.json 里可能写着 8192，CPU 上跑
        # 8192 个 token 一次前向要几十秒，检索场景没有意义）
        tok.enable_truncation(max_length=self.max_length)
        self._pad_id_cache = self._resolve_pad_id(tok)
        return tok

    def _resolve_pad_id(self, tok) -> int:
        """确定 padding 用的 id。

        tokenizer.json 里没配 padding 时 encode_batch 会**报错**而不是给
        不均等的长度（tokenizers >= 0.15 的行为）。所以这里显式打开
        padding，而不是依赖 tokenizer.json 的配置。
        """
        pad = None
        for name in ("<pad>", "[PAD]", "<|pad|>"):
            try:
                pad = tok.token_to_id(name)
            except Exception:
                pad = None
            if pad is not None:
                break
        if pad is None:
            try:
                cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
                pad = cfg.get("pad_token_id")
            except (OSError, ValueError):
                pad = None
        if pad is None:
            pad = 1  # xlm-roberta 的 <pad> id
        tok.enable_padding(pad_id=int(pad), pad_token="<pad>")
        return int(pad)

    def _load_session(self):
        ort = _ensure_onnxruntime(self.prefer_gpu)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 0  # 0 = 交给 onnxruntime 按物理核数决定
        providers = _resolve_providers(ort, self.prefer_gpu)
        try:
            session = ort.InferenceSession(
                str(self.model_dir / "model.onnx"),
                sess_options=opts,
                providers=providers,
            )
        except Exception as exc:
            raise RuntimeError(
                f"加载 ONNX 模型失败：{self.model_dir / 'model.onnx'}\n"
                f"原因：{exc}\n"
                f"如果报 'External initializer ... out of bounds'，说明 model.onnx_data "
                f"还没下载完或与 model.onnx 不匹配，等下载完成再试。"
            ) from exc

        input_names = [i.name for i in session.get_inputs()]
        output_names = [o.name for o in session.get_outputs()]
        return session, input_names, output_names

    def _detect_dim(self) -> int:
        """从 config.json 读 hidden_size；读不到就查输出张量的最后一维。"""
        try:
            cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
            for key in ("hidden_size", "d_model", "dim"):
                if key in cfg:
                    return int(cfg[key])
        except (OSError, ValueError):
            pass
        # 退路：查输出 shape。用 -1 表示动态维时取不到，这时才报错。
        for out in self.session.get_outputs():
            shape = out.shape
            if shape and isinstance(shape[-1], int) and shape[-1] > 1:
                return int(shape[-1])
        raise RuntimeError(
            f"无法确定嵌入维度：{self.model_dir / 'config.json'} 里没有 hidden_size，"
            f"ONNX 输出 shape 也是动态的（{out.shape}）。请检查模型导出是否完整。"
        )

    # --------------------------------------------------------------- 推理

    def _run_batch(self, texts: list[str]) -> np.ndarray:
        encs = self.tokenizer.encode_batch(texts)
        max_len = max((len(e.ids) for e in encs), default=1)
        max_len = max(1, min(max_len, self.max_length))

        input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
        for row, enc in enumerate(encs):
            ids = enc.ids[:max_len]
            input_ids[row, :len(ids)] = ids
            mask = (enc.attention_mask[:max_len] if enc.attention_mask
                    else [1] * len(ids))
            attention_mask[row, :len(ids)] = mask
        # 某些 tokenizer 会用 0 做 padding，但模型声明的 pad 是 1（xlm-roberta）。
        # tokenizer.json 里若没配 pad_token，encode_batch 出来是 0，必须补成
        # attention_mask 为 0 的位置对应的真实 pad id，否则 padding 位置会
        # 参与注意力并被当成真实 token（句首全是 <s>）。
        pad_id = self._pad_id()
        input_ids[attention_mask == 0] = pad_id

        feed: dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name == "input_ids":
                feed[name] = input_ids
            elif name == "attention_mask":
                feed[name] = attention_mask
            elif name == "token_type_ids":
                feed[name] = np.zeros_like(input_ids)
            # 其他未知输入（如 position_ids）交给模型默认值

        outputs = self.session.run(None, feed)
        named = dict(zip(self.output_names, outputs))
        return self._pool(named, attention_mask)

    def _pad_id(self) -> int:
        return getattr(self, "_pad_id_cache", 1)

    def _pool(self, named: dict[str, np.ndarray], attention_mask: np.ndarray) -> np.ndarray:
        """把模型输出池化成句向量。

        优先级：
          1. `sentence_embedding` —— 导出时已经池化过（通常是 CLS 或 mean），
             直接用它，不再自己池化，避免对已池化向量再求平均。
          2. `last_hidden_state` / `token_embeddings` —— 自己做 **mean
             pooling**（按 attention_mask 加权平均）。
          3. 唯一输出 —— 若三维就 mean pooling，二维就直接用。
        """
        vec = None
        for key in ("sentence_embedding", "pooler_output", "text_embeds"):
            if key in named:
                vec = named[key]
                break
        if vec is None:
            for key in ("last_hidden_state", "token_embeddings"):
                if key in named:
                    hidden = named[key]
                    vec = self._mean_pool(hidden, attention_mask)
                    break
        if vec is None:
            only = next(iter(named.values()))
            vec = (self._mean_pool(only, attention_mask)
                   if only.ndim == 3 else only)

        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[None, :]
        return _l2_normalize(vec)

    @staticmethod
    def _mean_pool(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """按 attention_mask 加权平均。

        **这是兜底路径，不是首选。** bge-m3 的 ONNX 导出带 `sentence_embedding`
        输出，实测它等于 CLS pooling 且已归一化（与手算 CLS 的余弦恒为 1.0，
        与 mean pooling 只有 0.79）。所以正常情况走 `_pool` 的第一优先级，
        直接取那个张量；只有模型没有这个输出时才退到这里做 mean pooling ——
        那说明导出方式和 bge 的标准做法不一致，效果会打折。

        不管走哪条路，padding 位置都必须被 mask 掉：否则 batch 内不同长度的
        样本互相污染，单条推理看不出来，一上 batch_size>1 结果就变。
        """
        mask = attention_mask.astype(np.float32)[:, :, None]  # (B, L, 1)
        summed = (hidden.astype(np.float32) * mask).sum(axis=1)
        counts = np.maximum(mask.sum(axis=1), 1e-6)  # (B, 1)
        return summed / counts

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        bs = int(batch_size or self.default_batch_size or 16)
        chunks = []
        for start in range(0, len(texts), bs):
            chunks.append(self._run_batch(texts[start:start + bs]))
        out = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        # 分 batch 各自归一化与整体归一化等价（归一化是逐行的），所以
        # 直接拼即可，这一步只是保证 float32 连续布局。
        return np.ascontiguousarray(out, dtype=np.float32)


class OnnxReranker:
    """bge-reranker 的 ONNX 交叉编码重排。

    与嵌入模型的关键差别：输入是 **(query, passage) 对**，模型内部让
    query 和 passage 的 token 互相做注意力，输出一个相关性 logit。所以
    它比「双塔 + 余弦」准得多，代价是不能预计算 —— 每次查询都要对候选集
    逐个前向。这就是它在流水线里只对 RRF 融合后的前几十条生效的原因。
    """

    def __init__(self, model_dir: str | Path, max_length: int = 512,
                 prefer_gpu: bool = True):
        self.model_dir = Path(model_dir)
        self.max_length = int(max_length)
        self.prefer_gpu = bool(prefer_gpu)
        self._check_files()
        self.tokenizer = self._load_tokenizer()
        self.session, self.input_names = self._load_session()

    def _check_files(self) -> None:
        d = self.model_dir
        if not d.exists():
            raise FileNotFoundError(
                f"重排模型目录不存在：{d}\n"
                f"bge-reranker 需要手动下载（用 hf-mirror.com）：\n"
                f"  python fetch_model.py --reranker --dest {d}\n"
                f"或不加重排：search 时不传 --rerank。"
            )
        missing = [n for n in ("model.onnx", "tokenizer.json") if not (d / n).exists()]
        if missing:
            raise FileNotFoundError(
                f"重排模型目录 {d} 缺少：{', '.join(missing)}"
            )

    def _load_tokenizer(self):
        _ensure_tokenizers()
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        tok.enable_truncation(max_length=self.max_length)
        return tok

    def _load_session(self):
        ort = _ensure_onnxruntime(self.prefer_gpu)

        try:
            session = ort.InferenceSession(
                str(self.model_dir / "model.onnx"),
                providers=_resolve_providers(ort, self.prefer_gpu),
            )
        except Exception as exc:
            raise RuntimeError(
                f"加载重排模型失败：{self.model_dir / 'model.onnx'}\n原因：{exc}"
            ) from exc
        return session, [i.name for i in session.get_inputs()]

    def score(self, query: str, passages: list[str], batch_size: int = 8) -> np.ndarray:
        """返回 (len(passages),) 的 float32 相关性分数，越大越相关。

        不保证落在 [0,1]：模型最后一层是回归 logit，是否过 sigmoid 取决于
        导出。这里返回原始 logit，因为排序不受单调变换影响。
        """
        if not passages:
            return np.zeros(0, dtype=np.float32)
        bs = int(batch_size or 8)
        out: list[np.ndarray] = []
        for start in range(0, len(passages), bs):
            batch = passages[start:start + bs]
            encs = self.tokenizer.encode_batch(
                [(query or "", p or "") for p in batch]
            )
            max_len = max(1, min(max(len(e.ids) for e in encs), self.max_length))
            input_ids = np.zeros((len(encs), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(encs), max_len), dtype=np.int64)
            for row, enc in enumerate(encs):
                ids = enc.ids[:max_len]
                input_ids[row, :len(ids)] = ids
                mask = enc.attention_mask[:max_len] if enc.attention_mask else [1] * len(ids)
                attention_mask[row, :len(ids)] = mask

            feed: dict[str, np.ndarray] = {}
            for name in self.input_names:
                if name == "input_ids":
                    feed[name] = input_ids
                elif name == "attention_mask":
                    feed[name] = attention_mask
                elif name == "token_type_ids":
                    feed[name] = np.zeros_like(input_ids)

            logits = self.session.run(None, feed)[0]
            logits = np.asarray(logits, dtype=np.float32).reshape(len(encs), -1)
            # 有的导出输出 (B, 1)，有的 (B, num_labels)；取第 0 列
            out.append(logits[:, 0])
        return np.concatenate(out, axis=0)
