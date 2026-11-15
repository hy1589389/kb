# kb

本地知识库。把散落各处的 Markdown 文档收进来，切成带标题路径和原文行号的块，
再用「BM25 + 稠密向量」的混合检索找出来，答案带引用来源。

目前完成到生成：检索 → 大模型 → 带引用的答案。拒答和引用校验在后置步骤里做。
另有一个本地 Web 界面，可以在浏览器里检索和问答，并把结果直接跳回编辑器。

## 两种用法

| 形态 | 用途 | 谁在用 |
| --- | --- | --- |
| 本地 `python ingest.py web` | 日常使用：查文档、面试前用「今日复习」过一遍评测里没检索到的题、结果按行号跳回原文 | 我自己 |
| 在线 [hy666ys-kb-qa.ms.show](https://hy666ys-kb-qa.ms.show) | 只读演示：证明这条链路能真跑，点开就能问，答案带引用 | 看简历的人 |

在线那份不是长期服务，也不指望有人天天用。语料是 JavaGuide、CS-Notes 这类公开技术笔记，
模型本来就熟悉这些内容，检索带来的增量主要是「第几段、第几行」，不是「知不知道」。
要让知识库真正有用，语料得换成模型没见过的东西：自己的项目代码、面试复盘、内部文档。

魔搭免费档会休眠，冷启动约 40 秒加载索引和模型。面试前先点开预热一次。

## 目录结构

```
kb/
├── config.toml              # 切块参数、检索参数、集合定义
├── ingest.py                # 入口
├── data/
│   ├── raw/                 # 语料，按集合分目录
│   │   ├── tech/            # 技术文档（JavaGuide / CS-Notes / ...）
│   │   ├── papers/          # 论文 PDF
│   │   └── profile/         # 简历等个人材料
│   ├── processed/           # 切块结果（每个集合一份）
│   └── index/               # 检索索引：tech 用本目录，其它集合用子目录
├── models/                  # ONNX 模型（生成物，2.18 GB）
├── src/kb/
│   ├── config.py
│   ├── cli.py
│   ├── ingest/              # 第一层：清洗、切块、落盘
│   │   ├── clean.py
│   │   ├── chunker.py
│   │   ├── pdf.py           # PDF 解析（坐标排序 + 字号识别标题）
│   │   ├── quality.py       # 乱码率 / 空白块 / 平均块长
│   │   └── pipeline.py
│   ├── retrieve/            # 第二层：检索
│   │   ├── tokenize.py      # 中文 bigram 分词
│   │   ├── sparse.py        # BM25
│   │   ├── dense.py         # 暴力向量检索
│   │   ├── hybrid.py        # RRF 融合
│   │   ├── embedder.py      # ONNX 嵌入 / 重排
│   │   └── pipeline.py      # Retriever，支持多索引
│   ├── eval/                # 第三层：评测指标与运行器
│   ├── generate/            # 第四层：生成与引用校验
│   │   ├── prompt.py        # 提示词与上下文截断
│   │   ├── llm.py           # DeepSeek 客户端 + 密钥读取
│   │   ├── cite.py          # [n] 校验与切句
│   │   ├── pipeline.py      # Answerer / 两层拒答
│   │   └── types.py         # Citation / Answer
│   └── web/                 # 第五层：Web 界面与编辑器回链
│       ├── app.py           # create_app() + 路由
│       ├── editor.py        # 源文件解析 + code / notepad
│       └── static/index.html  # 单文件前端，无 CDN
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
Web 层（第五层）需要 `fastapi` 和 `uvicorn`，本机已有（fastapi 0.139.2 / uvicorn 0.51.0）。

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

### 5. 问答（检索 + 生成）

```powershell
python ingest.py ask "HashMap 什么时候把链表转成红黑树"
python ingest.py ask "HashMap 什么时候把链表转成红黑树" --top-k 3 --json
python ingest.py ask "LoRA 微调怎么配" --no-rerank --show-hits
```

先打答案，再打引用列表：

```
当链表长度大于阈值（默认为 8）时，会执行链表转红黑树的逻辑 [1][2][3]。

引用（5 条）：
[1] JavaGuide/docs/java/collection/java-collection-questions-02.md:222-297  Map（重要） > HashMap 的底层实现
...
```

模型需要 API key，按这个顺序找，找到即止：

1. 环境变量 `DEEPSEEK_API_KEY`
2. 项目根 `.env`（`DEEPSEEK_API_KEY=...`，支持 `#` 注释和引号，文件不入库）
3. Windows 注册表里的系统 / 用户环境变量

都没有就报错退出，不会自己建 `.env`。key 不写进源码、日志、报错和 `--json` 输出。

资料不足时拒答，正文是「知识库中没有找到足够的信息来回答这个问题。」，
`refused=true`，`refuse_reason` 写明是「检索无结果」还是「资料不足」。

### 6. 在编辑器里打开结果

`search --open N` 在打印完结果后，把第 N 条（从 1 开始）对应的源文件在编辑器里打开，
并用 `code -g "<绝对路径>:<行号>"` 跳到 `line_start`：

```powershell
python ingest.py search "RAG 是什么" --open 1
```

```
查询: RAG 是什么
索引: G:\kb\data\index   返回 3 条   模式: hybrid
====================================================================
[ 1]   0.0311  RAG 基础
     JavaGuide/docs/ai/interview-questions/rag-interview-questions.md:18-36
     相关内容：[《RAG 基础概念：检索、生成与工程取舍》]...
--------------------------------------------------------------------
提示：用 --open N 打开第 N 条结果的源文件（N 从 1 开始，例如 --open 1）
[open] D:\A\Microsoft VS Code\bin\code.CMD -g G:\kb\data\raw\JavaGuide\docs\ai\interview-questions\rag-interview-questions.md:18
       文件：G:\kb\data\raw\JavaGuide\docs\ai\interview-questions\rag-interview-questions.md
```

源文件路径按 `<raw_dir>/<repo>/<rel_path>` 拼，`raw_dir` 默认取 `[path].raw_dir`。
语料不在项目目录下时用 `--raw-dir`：

```powershell
python ingest.py search "RAG 是什么" --raw-dir G:\kb\data\raw --open 1
```

编辑器按 `code` → `notepad` 的顺序探测。`code` 能跳行；`notepad` 没有跳行参数，
只打开文件，输出里会注明「不支持跳行」。可以用 `[web].editor` 或 `--editor` 固定。

N 越界、源文件不存在、编辑器都不可用时只打印 `[X] ...` 并返回 1，不抛栈。
`--open` 和 `--json` 一起用时，打开说明写 stderr，stdout 的 JSON 不受影响。

### 7. 本地 Web 界面

```powershell
python ingest.py web                 # http://127.0.0.1:8000，自动开浏览器
python ingest.py web --port 8123
python ingest.py web --no-browser
```

索引和模型不在项目目录下时（比如用 `G:\kb` 那一套）：

```powershell
python ingest.py web --index-dir G:\kb\data\index --model-dir G:\kb\models\bge-m3 --raw-dir G:\kb\data\raw --no-browser
```

启动日志：

```
====================================================================
  Phase 5 · 本地 Web 界面
====================================================================
  地址     : http://127.0.0.1:8123
  索引     : G:\kb\data\index  (18,276 块)
  模型     : G:\kb\models\bge-m3
  语料     : G:\kb\data\raw
  编辑器   : code
--------------------------------------------------------------------
  正在加载索引（一次性，之后每个请求复用）...
  索引就绪 : 0.4s
  模型就绪 : 2.7s（预热检索）
  问答     : 已启用（deepseek-chat）
--------------------------------------------------------------------
  复习题库 : 6 道失败题（results/run_hybrid.json）
  打开 http://127.0.0.1:8123    （Ctrl+C 停止服务）
====================================================================
```

服务是长驻的，Ctrl+C 退出。索引和模型各加载一次（`Retriever` 构造 + 一次预热检索），
之后每个请求复用同一个实例；18,276 块的索引上单次混合检索约 50 ms。
没有 API key 也能启动，只是 `/api/ask` 返回 503，检索和 `--open` 照常。

界面：顶部输入框 + 检索 / 问答切换，还有一个「智能改写」开关；检索结果是卡片，显示排名、
分数、标题路径、`repo/rel_path:行号`、4 行摘要，两个按钮「展开全文」「在 VS Code 打开」；
问答显示答案和引用，正文里的 `[1]` 是可点的角标，点了滚到对应引用并高亮；拒答单独配色并
显示原因；底部状态条是块数、embedder、是否就绪和上次查询耗时，右侧是「今日复习」
「统计」两个入口。

路由：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 返回 `static/index.html` |
| GET | `/api/search?q=&top_k=10&mode=hybrid` | 结果列表，`mode` 可选 `hybrid` / `dense` / `sparse`；`rewrite=1` 先改写查询再融合 |
| POST | `/api/ask` | body `{"q": "...", "top_k": 5}`，返回 `Answer.to_dict()` |
| GET | `/api/open?chunk_id=...` | 在编辑器打开该块的源文件；未知 id 返回 404 |
| GET | `/api/health` | 就绪状态、块数、embedder、模型目录、编辑器 |
| GET | `/api/review/today?day=2026-11-02` | 今日复习：一道评测失败题 + 参考原文；`day` 可选，用来指定日期 |
| GET | `/metrics` | Prometheus 文本：请求计数、延迟分位、空召回、拒答、token 数、块数 |
| GET | `/api/metrics` | 同一批数据的 JSON：延迟分位 + 计数器 + 最近 64 条请求摘要 |

Python 侧入口是 `kb.web.app.create_app(retriever, answerer=None, *, raw_dir, editor=None, review_file=None)`。
`retriever` / `answerer` 由调用方构造好再传进去，`create_app` 自己不做任何加载 ——
这就是「只加载一次」的实现方式。

#### 今日复习

右下角「今日复习」按日期出一道评测里没检索到的题：先自己答，点「我答完了，看参考」
再展开知识库里的原文，需要的话再让模型基于这些段落作答。

题库来自 `[web].review_file` 指向的评测结果（默认 `results/run_hybrid.json`），
取 `per_query` 里 `rank` 缺失或大于 5 的条目，也就是前 5 条没命中的题。当前 40 条
评测里有 6 条（q008 / q012 / q016 / q019 / q024 / q036）。按东八区的日期取模
`date.toordinal() % 6` 轮换：同一天永远是同一道，6 天走完一圈。没有用户系统也没有
定时任务，进页面时算一次日期就有答案。

日期用固定 `+08:00`，不取进程本地时区：本地跑是 +08 没区别，但 HF Space 的容器默认
UTC，用 `date.today()` 会在北京时间早上 8 点换题。中国没有夏令时，固定偏移和
`Asia/Shanghai` 等价，还省掉 Windows 上要额外装 tzdata 的麻烦。

参考原文优先用 `gold` 字段（评测标注的正确块）。这些块已经在 `retriever.meta` 里，
按 `chunk_id` 直接取，不额外编码查询；18,276 块的索引上单次请求只剩查字典和一次
日期取模。只有索引重建、`chunk_id` 对不上时才会退回一次实时检索，此时响应里的
`chunks_source` 是 `search`，界面上会写明这是现检索结果而不是标注答案。

题库文件缺失、格式不对、或者一条失败题都没有时，接口返回 200 + `available=false`
+ `reason`，页面显示一句原因，其它功能不受影响。

答题草稿只放在页面内存里，刷新就没了 —— 没有用户系统，不做本地存储。

#### 模型精排

检索是两路召回、按名次融合，所以「词面凑巧重合」的结果会混进前几。搜 `token怎么控制`
时第 3 条是一本工具书的「适合谁看」——它的正文里有「怎么控制提交粒度」，短文档还被
长度归一化抬高。

搜索框旁边的「模型精排」开关：把前 15 条结果的标题和片段列给 DeepSeek，让它挑出真正
相关的并按相关度排序，其余丢掉；模型说「都不相关」就返回空列表。

同一句查询实测：

```
原顺序：Token预算 / 采样解码 / 适合谁看 / Fencing Token / 投机解码 / 全栈路线 / LLM运行机制 / ...
精排后：Token预算 / Token预算公式 / LLM运行机制
```

40 条评测集上（同一 top-5 口径）：

| | R@5 | Hit@1 | MRR |
| --- | --- | --- | --- |
| 不开精排 | 85.0% | 52.5% | 0.652 |
| 模型精排 | 85.0% | **67.5%** | **0.766** |

召回没变，第一名就对的题多了 15 个百分点。代价是一次 chat 调用（约 1,200 token、1~2 秒），
所以默认关闭；没配模型、调用失败、模型没给出可用判断时一律退回原顺序，响应里 `reranked`
是 false。

为什么不用 `bge-reranker-base`：同一份评测集上它把 R@5 从 85.0% 压到 80.0%、MRR 从
0.667 掉到 0.580，对中文技术文本区分力不足。查询改写也试过，模型给出的词太泛
（`token怎么控制` → `Token 控制 方法 实现`），融合后反而更差。
### 8. 评测

```powershell
python eval\run_eval.py --index-dir data\index --top-k 10
python eval\run_eval.py --no-dense            # 只跑 BM25，用于对照
python eval\run_eval.py --compare results\a.json results\b.json
```

### 9. 测试

```powershell
python tests\test_chunker.py      # 切块器
python tests\test_pdf.py          # PDF 解析
python tests\test_retrieve.py     # 检索层
python tests\test_eval.py         # 评测逻辑
python tests\test_generate.py     # 生成层（不联网）
python tests\test_collections.py  # 多集合与入库
python tests\test_web.py          # Web 层与编辑器回链（不加载模型）
python -m unittest discover -s tests
```

## 多集合

技术文档、论文、简历分开存，各自的索引物理隔离。传一份简历不会影响技术文档的检索结果，
重建简历索引也不会碰那 18,276 个块。

仓库里不带任何个人材料：`data/` 整个目录都在 `.gitignore` 里，部署脚本还会跳过
`data/index/profile`。`profile` 集合默认是空的，按下面这段配置自己放材料就能重建。

三个集合写在 `config.toml`：

```toml
[collections]
default_search = ["tech"]        # search 不写 --collection 时查哪些

[collections.tech]
raw_dir = "data/raw/tech"
index_dir = "data/index"         # 沿用原有索引位置
parser = "markdown"

[collections.papers]
raw_dir = "data/raw/papers"
index_dir = "data/index/papers"
parser = "auto"

[collections.profile]
raw_dir = "data/raw/profile"
index_dir = "data/index/profile"
parser = "auto"
generation = "disabled"          # 简历不走云端生成，只返回原文
```

加一份文档：

```powershell
copy 我的论文.pdf data\raw\papers\
python ingest.py ingest --collection papers --dry-run    # 先看解析结果
python ingest.py ingest --collection papers
python ingest.py index  --collection papers
python ingest.py search "注意力机制" --collection papers
```

`--dry-run` 会打印每个文件解析出几个块、平均多长、前两块长什么样，以及质量警告
（平均块长 < 100 字、乱码率 > 15%、空白块 > 30% 会标 `low_quality`，默认不参与检索）。

同一份文件重复入库不会重复计数：用 sha256 记指纹，内容没变就复用上次的块，
`chunk_id` 不会漂移。

删掉一份文档：

```powershell
python ingest.py drop --collection papers --path 我的论文.pdf
```

`drop` 只动目标集合的索引，并且会二次确认。

### PDF 解析

PyMuPDF 取文本块，按 `(y, x)` 坐标排序恢复阅读顺序 —— 不少导出工具的文本层顺序是乱的，
直接 `get_text()` 拿到的内容前后颠倒。标题用字号中位数识别（大于 1.15 倍算标题），
跨页重复率超过 60% 且位于页面顶部或底部 15% 的行当页眉页脚剔除。

引用位置对 PDF 用页码（`P3`），对 Markdown 用行号（`L222-297`）。

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

生成参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `model` / `base_url` | `deepseek-chat` | 模型与接口地址 |
| `api_key_env` | `DEEPSEEK_API_KEY` | 环境变量名，不是 key |
| `temperature` / `max_tokens` | 0.1 / 1024 | 采样参数 |
| `max_context_chars` | 6000 | 送进提示词的正文总字数上限，超了按排名丢后面的块 |
| `max_chunk_chars` | 1500 | 单块正文截断长度 |
| `top_k` | 5 | 送进提示词的块数 |
| `min_top_score` | 0（关闭） | 只在单路检索下有意义，混合检索是 RRF 分，见下 |
| `timeout` | 60 | 单次请求超时（秒） |

`min_top_score` 不能拿去卡混合检索：RRF 融合分只有名次含义（约 0.016~0.033），
和余弦相似度不在一个量纲上。代码在 `source=hybrid` / `rerank` 时显式跳过它并打日志，
否则会把所有查询都拒掉。

Web 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `host` | `127.0.0.1` | 监听地址，`0.0.0.0` 表示局域网可访问 |
| `port` | 8000 | 端口 |
| `editor` | 空 | 空 = 自动探测 `code` → `notepad`；也可写命令名或 `code.cmd` 的绝对路径 |
| `review_file` | `results/run_hybrid.json` | 「今日复习」的题库，一次评测的结果 JSON（相对项目根）；留空关掉入口 |

Web 没有单独的语料根目录配置，`/api/open` 和 `search --open` 都用 `[path].raw_dir`。
两处配同一个目录迟早会不一致，临时换目录用 `--raw-dir`。

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

### 重排：接了，但效果反而降了

加了 `bge-reranker-base`（XLM-R，1077 MB，跑在 DirectML 上）之后：

| 配置 | R@1 | R@3 | R@5 | R@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- |
| 混合 | 52.5% | 80.0% | 85.0% | 95.0% | 0.667 |
| 混合 + 重排 | 40.0% | 75.0% | 80.0% | 87.5% | 0.580 |

**所以默认不开重排**（`--rerank` 是可选参数）。原因查了两条：

1. **评测集的 gold 标窄了。** 5 个失败里 2 个其实答对了，只是答案落在另一篇文档。
   最典型的是问 vLLM 的三个参数，`self-llm` 里至少两份部署文档都解释了，但 gold 只标了
   DeepSeek 那篇；重排把 Hunyuan 那篇排上来就判失败。这是 chunk 级精确匹配的固有偏差，
   不是重排的错。
2. **扣掉假失败后仍有 3 个真失败**，R@1 也明显下降。重排模型对这类中文技术文本的区分力
   不够，而且 `max_length=512` 对平均 370 token、最长 1200+ token 的块会截断。

排查过程中排除了两个怀疑：分词器的 pair 模板是对的（XLM-R 的 `<s> A </s></s> B </s>`），
`<pad>` 虽然应该是 1 而代码填了 0，但 `attention_mask` 已经屏蔽，改成 1 分数完全一样。

**结论**：要提升的话，换 `bge-reranker-v2-m3`（多语言专用）比调参更有希望；
或者先把评测改成端到端判断答案正确性，消掉 gold 标窄带来的假失败。

复现：

```powershell
python eval\run_eval.py --index-dir data\index --out results\run_hybrid.json
python eval\run_eval.py --index-dir data\index --no-dense --out results\run_sparse_only.json
python eval\run_eval.py --index-dir data\index --no-sparse --out results\run_dense_only.json
python eval\run_eval.py --index-dir data\index --rerank --out results\run_hybrid_rerank.json
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

## 部署

镜像按「检索本地、生成走 API」的方式打，模型在构建期下好，索引随镜像走。

| 目标 | 说明 |
| --- | --- |
| 魔搭创空间 | 免费 2 vCPU / 8 GB，手机号注册，国内直连。步骤见 [deploy-modelscope.md](deploy-modelscope.md) |
| Hugging Face Spaces | 同档免费硬件，构建机走官方域名下模型。步骤见 [deploy-hf.md](deploy-hf.md) |

两个目标共用一套代码，差别只有模型的下载源和目标平台的构建配置文件：

```powershell
python tools\prepare_space.py G:\kb-ms  --platform modelscope   # 产出可直接 push 的目录
python tools\prepare_space.py G:\kb-hf  --platform hf
```

`tools/fetch_model.py` 用 `KB_MODEL_SOURCE=modelscope` 从 modelscope.cn 下权重，
默认 `hf` 走 hf-mirror / HF 官方。

## 已知问题

- 只处理 Markdown 和 PDF，网页、代码仓库还没做。
- PDF 只能按块级坐标恢复阅读顺序，同一行内的文字顺序救不回来（`insert_text` 在同一基线多次写入会被合并成一个 block）。
- 生成只用一路 DeepSeek，没有本地模型兜底；接口超时或限流时整轮失败，没有重试。
- 引用校验只检查编号是否存在、句子有没有标引用，不检查引用内容是否真的支持该论断。
- `scripts/*.ps1` 必须保持纯 ASCII。Windows PowerShell 5.1 对无 BOM 的 `.ps1` 按系统 ANSI
  代码页解码，中文注释会把换行吞掉，脚本直接解析失败。
- 本机 pip 装不上包（解包阶段被沙箱拒绝），依赖靠下 wheel 手动解压，
  见 [docs/phase2-notes.md](docs/phase2-notes.md)。
- `est_tokens` 是粗估，中文按 1 字 1 token、英文按 3.5 字符 1 token 算。
- 多集合的跨集合 RRF 权重没标定过，`profile` 目前是 1.0；想让个人材料排在技术文档后面就调小。

## TODO

- [x] 语料入库
- [x] 混合检索：BM25 + bge-m3 稠密向量 + RRF 融合
- [x] 评测集与 Recall@k / MRR
- [x] 重排验证（`bge-reranker-base` 实测效果反而低 5 个点，默认关闭，原因见「检索效果」）
- [x] 生成与引用溯源（`ask`，DeepSeek + 两层拒答 + `[n]` 编号校验 + 同位置引用归并）
- [x] 多集合隔离（tech / papers / profile）+ PDF 解析 + 指纹幂等
- [x] Web 界面 + 「今日复习」入口（按日期从评测失败案例出题）
- [x] 部署打包：Dockerfile + 两个平台的组装脚本（魔搭创空间 / HF Spaces）
- [ ] 实际上线：HF 账号注册被 418 拦，改走魔搭创空间，尚未推送
- [ ] 端到端评测：让模型判断答案正确性，消掉 gold 标窄导致的假失败
- [ ] 换 `bge-reranker-v2-m3` 再试重排
- [ ] 引用内容级校验（现在只查编号范围，不查论断是否被引用块支持）
- [ ] 增量索引（现在每个集合全量重建；profile 只有几块，几秒就完，暂时不需要）
- [ ] 网页、代码仓库解析
- [ ] 封装成 MCP Server

## 协议

代码 MIT。`data/raw/` 下的语料来自各开源仓库，仅供本地使用，已加进 `.gitignore`。
