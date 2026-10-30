# kb

本地知识库。把散落各处的 Markdown 文档收进来，切成带标题路径和原文行号的块，
再用「BM25 + 稠密向量」的混合检索找出来，答案带引用来源。

目前完成到检索和评测，生成部分还没写。

## 目录结构

```
kb/
├── config.toml              # 切块参数、检索参数、过滤规则
├── ingest.py                # 入口
├── data/
│   ├── raw/                 # 语料，git clone 到这里
│   ├── processed/           # 切块结果
│   └── index/               # 检索索引（生成物）
├── models/                  # ONNX 模型（生成物，2.18 GB）
├── src/kb/
│   ├── config.py
│   ├── cli.py
│   ├── ingest/              # 第一层：清洗、切块、落盘
│   │   ├── clean.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   ├── retrieve/            # 第二层：检索
│   │   ├── tokenize.py      # 中文 bigram 分词
│   │   ├── sparse.py        # BM25
│   │   ├── dense.py         # 暴力向量检索
│   │   ├── hybrid.py        # RRF 融合
│   │   ├── embedder.py      # ONNX 嵌入 / 重排
│   │   └── pipeline.py
│   └── eval/                # 评测指标与运行器
├── eval/
│   ├── questions.jsonl      # 评测集
│   └── run_eval.py
├── scripts/bootstrap.ps1    # 一键拉语料并入库
├── tests/                   # 回归测试
└── docs/
    ├── design-notes.md      # 切块与清洗的设计说明
    ├── phase2-notes.md      # 检索层的环境与选型记录
    └── eval-set.md          # 评测集构成说明
```

## 环境要求

Python 3.11 以上（用到 `tomllib`）。

**入库（第一层）零依赖**，标准库就够跑。
检索（第二层）需要 `onnxruntime` 和 `tokenizers` —— 本机 pip 装不上，见
[docs/phase2-notes.md](docs/phase2-notes.md) 里的手动安装步骤（两个 wheel 合计约 27 MB）。

不需要 torch，不需要 CUDA。

## 快速开始

### 1. 准备语料

任意 Markdown 仓库都行，clone 到 `data/raw/` 下：

```powershell
git clone --depth 1 https://github.com/Snailclimb/JavaGuide.git data/raw/JavaGuide
```

或者一次拉一批，脚本里列了 5 个中文仓库：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
```

### 2. 入库

```powershell
python ingest.py ingest
```

会打印每个仓库的文件数和切块数，结果写到 `data/processed/`。

### 3. 建检索索引

```powershell
python ingest.py index --embedder onnx --model-dir models\bge-m3 --batch-size 4
```

不装模型也能先跑通流水线（用零依赖的哈希嵌入，只用于自检，没有语义能力）：

```powershell
python ingest.py index --embedder hashing
```

### 4. 检索

```powershell
python ingest.py search "HashMap 什么时候把链表转成红黑树"
python ingest.py search "TCP 为什么三次握手" --no-dense --top-k 5
python ingest.py search "LoRA 微调怎么配" --json
```

### 5. 评测

```powershell
python eval\run_eval.py --index-dir data\index --top-k 10
python eval\run_eval.py --no-dense            # 只跑 BM25，用于对照
python eval\run_eval.py --compare results\a.json results\b.json
```

### 6. 测试

```powershell
python tests\test_chunker.py      # 切块器
python tests\test_retrieve.py     # 检索层
python tests\test_eval.py         # 评测逻辑
python -m unittest discover -s tests
```

## 配置

改 `config.toml`，不用动代码。

切块参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `max_chars` | 1200 | 正文块的目标上限 |
| `min_chars` | 120 | 小于这个值的相邻块会被合并 |
| `drop_below` | 40 | 合并后仍小于这个值的块丢弃 |
| `overlap` | 120 | 超长章节二次切分时，块首补上的上文 |
| `code_block_max_chars` | 2400 | 代码块超过这个值才拆，优先在空行处拆 |

检索参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `index_dir` | `data/index` | 索引目录 |
| `model_dir` | `models/bge-m3` | 嵌入模型 |
| `dense_top_k` / `sparse_top_k` | 50 | 两路各自召回多少 |
| `fusion_top_k` | 30 | 融合后进入重排的候选数 |
| `final_top_k` | 10 | 最终返回条数 |
| `rrf_k` | 60 | RRF 的平滑常数 |
| `embed_max_length` | 512 | 截断长度 |

命令行参数覆盖配置：

```powershell
python ingest.py ingest --max-chars 800 --overlap 80
python ingest.py search "xxx" --top-k 20 --no-sparse
```

## 检索怎么做的

**两路召回 + RRF 融合。**

- 稠密那一路用 `bge-m3`，把 `embed_text`（标题路径 + 正文）编码成 1024 维向量。
- 稀疏那一路是 BM25。中文用**字符 unigram + bigram** 切词，不引入分词器 ——
  bigram 对中文检索的效果已经够用，而 jieba 对本机来说是白增加一个依赖。
- 两路各取 50 条，用 RRF 融合后取前 30。

**为什么用 RRF 而不是加权分数**：稠密余弦相似度和 BM25 分数不在一个量纲上，
归一化不可靠（BM25 的绝对值随查询词频变化）。RRF 只用名次，所以免标定：

```
score(d) = Σ  weight_i / (k + rank_i(d))
```

**为什么不用向量库**：18,276 × 1024 的 float32 只有 75 MB，一次全量余弦就是一次矩阵乘法，
CPU 上几十毫秒。这个规模上引入 Qdrant 之类的组件只增加故障面。块数到百万级再换。

**为什么直接取 `sentence_embedding` 输出**：bge-m3 的 ONNX 导出已经做完了池化和归一化，
实测它等于 CLS pooling（与手算 CLS 的余弦恒为 1.0，与 mean pooling 只有 0.79）。
自己再做一遍 mean pooling 会掉点。

## 检索效果

40 条真实问题，gold 是逐条在语料里核对过的 `chunk_id`（平均 1.85 个/题），top_k=10，
同一份索引跑三种配置：

| 配置 | R@1 | R@3 | R@5 | R@10 | MRR@10 | 未召回 | 40 条耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 只 BM25 | 47.5% | 65.0% | 72.5% | 87.5% | 0.596 | 5 | 0.1s |
| 只稠密 bge-m3 | 52.5% | 75.0% | 85.0% | 90.0% | 0.648 | 4 | 11.7s |
| 混合（默认） | 52.5% | 80.0% | 85.0% | 95.0% | 0.667 | 2 | 11.8s |

混合在 R@5 上和只稠密持平，但把「一条都没召回」的问题从 4 条降到 2 条，
R@10 从 90% 提到 95%，MRR 从 0.648 到 0.667。

按题型看两路的互补关系（R@5）：

| 题型 | 只 BM25 | 只稠密 | 混合 |
| --- | --- | --- | --- |
| 术语精确匹配 | 83.3% | 100% | 100% |
| 同义改写 | 85.7% | 71.4% | 85.7% |
| 中英混杂 | 71.4% | 85.7% | 71.4% |
| 代码类 | 50.0% | 100% | 87.5% |
| 跨节综合 | 71.4% | 71.4% | 85.7% |
| 干扰项 | 80.0% | 80.0% | 80.0% |

按难度（R@5）：简单 100% / 77.8% / 100%，中等 83.3% / 72.2% / 77.8%，
困难 76.9% / 69.2% / 84.6%（只稠密 / 只 BM25 / 混合）。

**混合不是在所有题型上都更好。** 代码类 87.5% 低于只稠密的 100%，中英混杂 71.4%
低于 85.7%。原因是 RRF 等权融合，BM25 在代码查询上召回的错误块把正确块挤了下去。
下一步该试加权 RRF，或者给 BM25 一路加分数阈值。

复现：

```powershell
python eval\run_eval.py --index-dir data\index --out results\run_hybrid.json
python eval\run_eval.py --index-dir data\index --no-dense --out results\run_sparse_only.json
python eval\run_eval.py --index-dir data\index --no-sparse --out results\run_dense_only.json
python eval\run_eval.py --compare results\run_sparse_only.json results\run_hybrid.json
```

评测集的构成、每道题考什么、gold 怎么标的，见 [docs/eval-set.md](docs/eval-set.md)。

## 输出

`data/processed/` 下会生成：

- `chunks_all.jsonl` —— 所有块
- `chunks_<repo>.jsonl` —— 按仓库分片，方便以后做增量索引
- `manifest.jsonl` —— 文档清单
- `stats.json` —— 统计报告

每个块是一个 JSON 对象：

| 字段 | 说明 |
| --- | --- |
| `chunk_id` / `doc_id` | 块 ID 和文档 ID |
| `repo` / `rel_path` | 来源仓库和相对路径 |
| `title` | 文档的 H1 |
| `section_path` | 标题路径数组 |
| `breadcrumb` | 标题路径字符串，如 `A > B > C` |
| `chunk_index` | 在文档内的序号 |
| `part` / `n_parts` | `n_parts > 1` 表示这一节被拆成了多块 |
| `merged_paths` | 这块合并了哪些小节 |
| `line_start` / `line_end` | 原文行号 |
| `char_count` / `cjk_count` / `est_tokens` | 长度统计 |
| `text` | 正文 |
| `embed_text` | 送去嵌入的内容，标题路径拼在正文前面 |

## 语料规模

当前 `data/raw/` 下 5 个仓库的入库结果：

```
文档     1,278 篇
切块     18,276 个
字符     11,504,287（其中中文 3,215,198）
耗时     17.1s
块长     p50 529 / p90 1163 / p95 1308 / max 2415
```

BM25 索引：212,481 个词，4,053,133 条 posting，平均块长 370 token。

## 实现说明

几条设计上的硬约束，展开在 [docs/design-notes.md](docs/design-notes.md)：

- 清洗不改行数。删内容一律用等量换行替换，这样行号才能和原文对上。
- 切分点不落在代码块内部。所以含代码的块可能超过 `max_chars`，上限是 `code_block_max_chars`。
- 小块合并后，标题路径取各方的最长公共前缀，而不是沿用第一块的标题。
- 超过 200 字符的连续无空白串替换成占位符，主要是 base64 图片。

## 已知问题

- 只处理 Markdown，PDF 和网页还没做。
- 生成和引用溯源还没写，目前只到检索。
- `scripts/*.ps1` 必须保持纯 ASCII。Windows PowerShell 5.1 对无 BOM 的 `.ps1` 按系统 ANSI
  代码页解码，中文注释会把换行吞掉，脚本直接解析失败。
- 本机 pip 装不上包（解包阶段被沙箱拒绝），依赖靠下 wheel 手动解压，
  见 [docs/phase2-notes.md](docs/phase2-notes.md)。
- `est_tokens` 是粗估，中文按 1 字 1 token、英文按 3.5 字符 1 token 算。

## TODO

- [x] 语料入库
- [x] 混合检索：BM25 + bge-m3 稠密向量 + RRF 融合
- [x] 评测集与 Recall@k / MRR
- [ ] 重排（bge-reranker，代码已写，权重未下）
- [ ] 生成与引用溯源
- [ ] 增量索引
- [ ] PDF、网页、代码仓库解析
- [ ] 封装成 MCP Server

## 协议

代码 MIT。`data/raw/` 下的语料来自各开源仓库，仅供本地使用，已加进 `.gitignore`。
