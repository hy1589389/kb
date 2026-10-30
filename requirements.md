# Phase 2 检索层：依赖与安装

检索层的代码只依赖 numpy + 标准库。只有走 ONNX 真实模型那条路时，才需要
`onnxruntime` 和 `tokenizers` 两个包。

## 当前这台机器的状态

| 包 | 状态 |
| --- | --- |
| numpy 2.3.5 | 已装 |
| onnxruntime 1.27.0（仅 CPU） | 已装 |
| tqdm 4.67.1 | 已装（没有就用朴素打印，不报错） |
| tokenizers | **未装**，需要手动解压 wheel |
| torch / transformers / sentence-transformers | 不装，也不用 |

本机 pip 在解包阶段会被沙箱拒绝，所以 `pip install` 走不通。下面的步骤
不经过 pip，直接把 wheel 当 zip 解开。

## 手动安装 tokenizers

```powershell
# 1. 下载 wheel（cp310-abi3 表示兼容 CPython 3.10 及以上，3.13 可用）
#    文件名固定是 tokenizers-0.23.2-cp310-abi3-win_amd64.whl，约 2.7 MB
#    从 pypi 的下载页找，或者用 hf-mirror（huggingface.co 在这台机器上不通）
Invoke-WebRequest `
  "https://files.pythonhosted.org/packages/.../tokenizers-0.23.2-cp310-abi3-win_amd64.whl" `
  -OutFile "$env:TEMP\tokenizers.whl"

# 2. 解压到工作副本的 .pylibs（wheel 就是 zip，改扩展名即可，不需要 pip）
New-Item -ItemType Directory -Force .pylibs | Out-Null
Copy-Item "$env:TEMP\tokenizers.whl" "$env:TEMP\tokenizers.zip"
Expand-Archive "$env:TEMP\tokenizers.zip" -DestinationPath .pylibs -Force

# 3. 验证
python -c "import sys; sys.path.insert(0, '.pylibs'); import tokenizers; print(tokenizers.__version__)"
```

`.pylibs` 的位置会被自动找到：`src/kb/retrieve/embedder.py` 里的
`_ensure_tokenizers()` 在 `import tokenizers` 失败时，会把
`<项目根>/.pylibs` 插到 `sys.path` 再试一次。放在别处就设环境变量
`KB_PYLIBS=<解压目录>`。

## 模型文件

ONNX 路径需要两个模型目录（自己下，代码不联网）：

```
G:\kb\models\bge-m3\                    稠密嵌入，约 2.2 GB
    model.onnx          模型图（几百 KB，权重外置）
    model.onnx_data     外置权重（必须和 model.onnx 同目录）
    tokenizer.json      HF fast tokenizer
    config.json         hidden_size=1024，决定向量维度
    special_tokens_map.json

G:\kb\models\bge-reranker-v2-m3\        交叉编码重排，可选
    model.onnx
    tokenizer.json
```

下载源用 `hf-mirror.com`（`huggingface.co` 不通）。

缺文件时的报错是分级的：

- 目录不存在 → 提示跑 `fetch_model.py`，并列出需要哪几个文件
- 目录在但缺 `model.onnx` / `tokenizer.json` / `config.json` → 列出缺哪个，
  并提示「模型可能还在下载中，对比 `model.onnx_data` 的大小是否还在增长」
- `model.onnx` 与 `model.onnx_data` 不匹配（下载没完就试着加载）→
  报 `External initializer ... out of bounds`，并指出这是数据文件不完整

没装 tokenizers 时不会静默降级，报错里直接给出上面这段安装步骤，并提示
可以先用 `--embedder hashing` 跑通流水线。

## 不需要模型就能跑的两条路

```powershell
# 单元测试（68 个用例，零依赖）
python tests\test_retrieve.py

# 真实语料自检：用 HashingEmbedder 建受限索引并跑 5 个查询
python tests\smoke_retrieve.py --limit 3000
```

这两条都只用 numpy，任何时候都能跑。

## 建索引与查询

```powershell
# 用真实模型建全量索引（18,276 块，CPU 上约 40~70 分钟，16 核）
python ingest.py index --chunks "G:\kb\data\processed\chunks_all.jsonl" `
                       --index-dir data\index --embedder onnx --batch-size 16

# 先用 hashing 建个小索引，验证 CLI 和索引格式
python ingest.py index --chunks "G:\kb\data\processed\chunks_all.jsonl" `
                       --index-dir data\index-hashing --embedder hashing --limit 3000

# 查询
python ingest.py search "TCP 三次握手为什么需要三次" --index-dir data\index-hashing --top-k 5
python ingest.py search "缓存穿透" --index-dir data\index-hashing --no-dense   # 只走 BM25
python ingest.py search "缓存穿透" --index-dir data\index-hashing --json       # 机器可读
```

预期输出（`--limit 800` 的 hashing 索引，`--no-dense`）：

```
查询: TCP 三次握手
索引: ...\cli-index   返回 2 条   模式: sparse
====================================================================
[ 1]   8.7753  HTTP > 五、具体应用 > 连接管理 > 1. 短连接与长连接
     CS-Notes/notes/HTTP.md:369-376
     当浏览器访问一个包含多张图片的 HTML 页面时，除了请求访问的 HTML 页面资源...
```

## 索引目录的内容

| 文件 | 说明 |
| --- | --- |
| `embeddings.npy` | (N, dim) float32，已 L2 归一化，支持 mmap |
| `dense_meta.json` | N 与 dim |
| `postings.npz` | BM25 倒排表：`term_ids`/`doc_ids`/`tfs`/`seg_starts`/`doc_lengths`/`idf` |
| `vocab.json` | token → term_id |
| `meta.json` | BM25 的 N / avgdl / k1 / b |
| `chunks_meta.jsonl` | 第 i 行对应第 i 个向量，含正文与定位信息（不含 `embed_text`） |
| `index_meta.json` | 建索引时的 embedder 类型、维度、参数、耗时 |

用 `--embedder hashing` 建的索引只能用 `HashingEmbedder` 查询，反过来也一样。
两种 embedder 的向量空间无关，混用不会抛异常只会返回垃圾结果，所以
`Retriever` 会检查 `index_meta.json` 里记录的类型并在不匹配时直接报错。
