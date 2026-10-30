# 切块与清洗设计说明

README 保持简短，设计上的取舍和开发过程记录在这里。

## 为什么按标题切，不按字符数切

技术文档的 Markdown 本身就是有结构的：标题层级就是语义边界，代码块是不可分割的单元。

按固定字符数硬切会同时犯两个错。一是把一段代码劈成两半，检索命中的是半截代码；二是把标题和它下面的正文分到不同的块里，检索结果失去语境。

现在的做法是按 ATX 标题（`#` 到 `######`）切分成 Section，记录完整标题路径。嵌入时把路径拼在正文前面，这样"问题里有关键词、正文里恰好没有"的情况也能召回。

## 三条硬约束

### 1. 清洗必须保持行数不变

所有删除操作（front matter、HTML 注释、徽章、只含图片的行）都替换成等量的换行，而不是直接删掉。

这不是洁癖。代码块的起止标记、每个块的行号，都依赖"清洗后第 i 行等于原文第 i 行"。开发早期版本在清洗时顺手压缩了空行，结果标记数组和文本错位，`###` 级标题被当成正文、整节内容被静默丢弃，而且不报错。这类 bug 从输出上完全看不出来。

`tests/test_chunker.py` 里的 `test_front_matter_and_comments_removed_but_lines_preserved` 和 `test_line_numbers_point_at_real_content` 守着这条。

### 2. 切分点不落在代码块内部

切分只能发生在代码块之外。代价是含超长代码块的块会超出 `max_chars`。

但完全不切也不行。真实语料里 self-llm 有 56KB 的 base64 图片示例、7KB 的单行 repr dump，这些既超出嵌入模型的上下文（会被静默截断），也不可能被检索命中。所以设了 `code_block_max_chars`：超过它才拆，拆的时候优先找空行（代码块内的空行通常就是函数边界），找不到空行就退到按行切、再退到按字符切，每段都补全围栏。

### 3. 不静默丢内容

每个块记录 `line_start` 和 `line_end`，可以回链到原文。

小到不成句的章节不删除，而是"尾巴吸收"合并进相邻的块。`drop_below` 只在合并之后仍小于阈值时才丢，并且数量会记进 `stats.json`。丢了就再也检索不到了，宁可让前一块稍杂一点。

## 超长无空白串的压缩

连续 200 字符以上的 `[A-Za-z0-9+/=_-]` 会被替换成 `[已省略 N 字符的长串数据]`。

触发这条规则的主要是 base64 图片。self-llm 的 Gemma3 教程里有 `data:image/png;base64,...` 的示例，单个块 56KB、100% 是 base64。这段内容对检索零贡献，却在总字符里占了 61 万。压缩后同样的块降到几百字符，正常标识符不受影响。

## 开发记录

按发现顺序记录。这些都是写测试和拿真实语料跑出来的，不是提前设计好的。

序号 4 到 9 都是尺寸异常值，通过 `tests/analyze_chunks.py` 定位。

| # | 现象 | 原因 | 处理 |
| --- | --- | --- | --- |
| 1 | `###` 级标题被当成正文，整节内容消失 | 清洗压缩空行导致行号错位 | 改成等量换行替换 |
| 2 | 一句话的小节凭空消失 | `drop_below` 直接删了 | 加"尾巴吸收"，挂到相邻块上 |
| 3 | 合并三节内容却标着第一节的标题 | 沿用前一块的 breadcrumb | 改取最深公共前缀 |
| 4 | 出现 3400 字的块 | 没有空行的长段落，按行切切不动 | 按句读边界切，再退到字符级 |
| 5 | 代码块围栏不成对 | 按行硬切切进了代码块内部 | 切分点限制在代码块外 |
| 6 | 含空行的代码块被拦腰截断 | 正则 split 空行不认代码块 | 段落切分改为 fence-aware |
| 7 | 单个块 56,768 字符 | base64 图片数据 | 压缩超长无空白串 |
| 8 | 单个块 7,180 字符 | repr dump，一整行 7172 字符 | 代码块内单行超长时退到字符级切分 |
| 9 | 单个块 5,354 字符 | 三个代码块紧挨着、中间无空行，被当成一段 | 开启围栏也算段落边界 |

修完之后块长上限由参数界定：正文块不超过 `max_chars`，含代码的块不超过 `code_block_max_chars`。最大块从 56,768 降到 2,415。

## 测试覆盖

`tests/test_chunker.py` 共 14 个用例，分三组：

- `TestStructure` —— 结构解析。深层标题、代码块内的假标题、front matter 与行号对齐、徽章图片剔除
- `TestChunking` —— 切分行为。超长段落、代码块保护、base64 压缩、相邻代码块、小块合并、尾巴吸收
- `TestNoContentLoss` —— 不变量。清洗后有内容的每一行都必须出现在某个块里；块的行号区间内必须能找到它的首行内容

第三组是最重要的两个断言，上面 1、2 两条约束就靠它们兜底。

```powershell
python tests\test_chunker.py
```

## 环境上的一点坑

`scripts/*.ps1` 必须保持纯 ASCII。

本机是 Windows PowerShell 5.1，对无 BOM 的 `.ps1` 按系统 ANSI 代码页（中文 Windows 上是 936）解码。UTF-8 的中文注释被按 GBK 解码后，多字节字符会吞掉后面的换行，脚本行号整体错位，报出来是一堆莫名其妙的 "Unexpected token"。

Python 源码不受影响，Python 3 固定按 UTF-8 解码源文件，所以面向用户的中文输出都放在 Python 层。

---

# Phase 2 检索层设计说明

## 中文检索不分词

用字符 unigram + bigram，不上 jieba 这类分词器。

分词器的错误在检索里不可逆：把「深度学习」切成「深度/学习」，而查询是「深度学习框架」，这条链路上没有任何补救手段。词典还永远滞后于术语（模型名、缩写、错误码天天在变）。unigram 保证任何单字都不会漏召回，bigram 提供短语区分度（「深度学习」和「机器学习」靠 bigram 分开），两者的 IDF 差异让常见字自动沉底，不需要停用词表。

另一层原因是装不上。本机 pip 解包阶段被沙箱拒绝，为一个稳定的 token 序列引入 30 MB 词典不划算。

## BM25 而不是 TF-IDF

语料里的块长从 40 字符到 2415 字符，差 60 倍。没有长度归一化，长块因为「词多所以命中词多」而系统性排前面。b=0.75 让长块里的词频收益递减，这是 BM25 在这份语料上稳定压过 TF-IDF 余弦的原因。`tests/test_retrieve.py` 的 `test_length_normalization_active` 用两篇同样词频、不同长度的文档守着这一条：b=0.75 时短文档得分更高，b=0 时两者相等。

IDF 用 `ln(1 + (N-df+0.5)/(df+0.5))`，那个 +1 是必需的。去掉它，出现在超过一半文档里的词（中文 unigram 里的「的」「是」）IDF 变负，负分词的贡献会把排序搅乱。

## 倒排表的持久化

三个等长的 numpy 平行数组加两个 JSON：

```
postings.npz   term_ids(int32) / doc_ids(int32) / tfs(int16) / seg_starts(int64)
               / doc_lengths(int32) / idf(float32)
vocab.json     term -> term_id
meta.json      N / avgdl / k1 / b
```

不用 pickle：反序列化会执行任意代码，索引文件来自别处就是 RCE；而且 pickle 存的是内存镜像，跨 numpy 版本会失效。`np.load(..., allow_pickle=False)` 是硬要求。

不用 JSON 存倒排表：18k 文档的 posting 是千万级，JSON 文本化后体积和解析时间都不可接受。

建索引结束时按 `term_ids` 做 argsort，于是每个 term 的 posting 落在连续区间里，`seg_starts` 记录区间边界。查询时 vocab 查表拿 term_id，直接切片拿 posting，不需要扫全表。真正耗时的只有按命中的 posting 累加分数，候选集是「命中至少一个查询词的文档数」。

## 稠密检索不用向量库

18,276 × 1024 float32 = 74.9 MB，一次 sgemv 是 1.9×10^7 次乘加，CPU 上十几毫秒。Qdrant 起容器、一次 HTTP 往返本身就要几毫秒，还要维护额外的数据副本和 ID 映射。向量库解决的是十亿级向量的近似最近邻，这个规模上它只是负担。

换方案的门槛是百万块（4 GB 矩阵），那时再谈 IVF/HNSW。

加载用 `np.load(mmap_mode="r")`：启动零拷贝，多个进程共享物理页。它省的是内存和启动时间，不是查询时间 —— 暴力检索的定义就是每次扫全矩阵。

取 top-k 用 `argpartition`（O(N) 选择）再对 k 个元素排序，比全量 `argsort` 省掉约 40%。`k >= N` 时退化为全排序，代码里用 `min(top_k, n)` 兜住。

## RRF 而不是加权分数

余弦相似度落在 0.3~0.8，BM25 分数在中文 bigram 语料上常见 5~20，直接加权求和的结果由 BM25 的尺度决定，稠密那一路等于没参与。

归一化补救不了：min-max 依赖当前候选集，同一个文档在 top_k=50 和 top_k=10 下归一分不同；单个离群高分把其余结果压平；z-score 的分布随查询变化。要标定就得为每种查询类型定参数，评测集换个分布就得重标。

RRF 只用名次：`Σ w_i/(k+rank_i)`，k 默认 60。权重调的是路与路之间的信任度，不是分数的尺度。某路没召回该文档就不贡献分数。

按分数降序、分数相同按 doc_index 升序，保证评测可复现。

## ONNX 而不是 torch

bge-m3 需要模型定义、权重、分词三样东西。ONNX 把前两样固化成文件，分词靠 `tokenizer.json`，运行时只要一个推理引擎（onnxruntime 已装），省掉 torch + transformers + sentence-transformers 这 2 GB 依赖。CPU 后端只影响速度不影响结果。

三个自适应点，都是导出脚本差异逼出来的：

- 输入名用 `session.get_inputs()` 拿。bge-m3 的 XLMRoberta 没有 `token_type_ids`，就按实际声明喂，多喂一个不存在的输入直接报错
- 输出名可能是 `last_hidden_state`，也可能是已池化的 `sentence_embedding`。已池化的不再池化，否则对均值再求均值
- 维度从 `config.json` 的 `hidden_size` 读，读不到才查输出张量形状

mean pooling 按 attention_mask 加权，不用 CLS：bge-m3 的训练目标就是 mean pooling，CLS 位置没有对比学习优化过。mask 必须盖上 padding，否则 batch 内不同长度的样本互相污染 —— 单条推理看不出来，batch_size>1 就变结果。

`enable_padding` 显式打开。tokenizer.json 里没配 padding 时 `encode_batch` 会直接报错而不是返回不等长数组。

## HashingEmbedder 是自检工具

存在的意义是让整条链路在 2 GB 模型下载完成前能跑通和写测试。它不是语义模型，数字不能当指标。

哈希用 blake2b(8 字节) 异或 crc32。不能用内置 `hash()`：它对 str 加了每进程随机盐，索引进程和查询进程的向量空间不同，结果是「A 建的索引 B 查不到」，而且不报错。`test_deterministic_across_processes` 用两个不同的 PYTHONHASHSEED 跑子进程钉死这条。

桶下标之外再用哈希的一位做符号位（hashing trick）。只用 `h % dim` 的话所有 token 都往正数上加，向量全落在正卦限，任意两条文本的余弦都能到 0.5 以上，区分度被压掉。

权重是 sublinear tf (`1+log(tf)`) × idf (`log(1+N/df)`)。

## 实测数字

| 项目 | 数值 |
| --- | --- |
| 单元测试 | 68 个用例，0.6 s，全绿 |
| 自检语料 | 前 3,000 块 / 174 万字符 / 580 字符每块 |
| BM25 索引 | 74,610 term / 561,617 posting / avgdl 308.7 |
| 建索引耗时（hashing, 3000 块） | 6.7 s（含 6.0 s 编码） |
| HashingEmbedder 编码吞吐 | 约 650 chunk/s（单进程，512 维） |
| 单次查询 | 2~4 ms（3000 块，dense + BM25 + RRF） |
| 向量矩阵 | 3000 × 512 = 5.9 MB（全量 18k × 1024 将是 74.9 MB） |

hashing 自检的 top-5 结果里 BM25 那一路明显更准（「TCP 三次握手」召回 `CS-Notes/notes/计算机网络 - 传输层.md` 的三次握手小节，「Redis 持久化 AOF 和 RDB」召回 `Redis.md` 的持久化小节）。稠密那一路因为是共词匹配，会出现「线程池」命中讲解 RAG rerank 的文档这类噪声。这也说明单看融合后的分数没有意义，要等真实模型。

## 测试覆盖

`tests/test_retrieve.py` 68 个用例：

- `TestTokenizer`（11）—— bigram 数量、中英混排不产生跨语种假 bigram、CJK 标点不参与 bigram、全角折叠、零宽字符
- `TestBM25`（11）—— 排序合理、稀有词排第一、IDF 恒正且稀有词更大、常见字贡献低、长度归一化开关（b=0.75 vs b=0）、存盘加载结果逐条一致、落盘不是 pickle、倒排区间首尾相接
- `TestDenseIndex`（9）—— 与教科书式余弦实现对照、k>=N 边界、归一化与未归一化查询等价、维度不匹配报错、mmap 加载一致
- `TestRRF`（9）—— 已知名次的精确分数、k 的影响、权重生效（含权重 0 关掉一路）、同路重复只算一次、并列按 doc_index 稳定排序
- `TestHashingEmbedder`（7）—— 同实例确定性、跨进程确定性、模长≈1 且空文本为零向量不产生 NaN、相似文本余弦更高、不相似文本余弦低（符号位回归）
- `TestSearchHit`（4）—— to_dict/from_dict 往返、缺省字段容错
- `TestRetrieverEndToEnd`（17）—— 小语料上建索引、稀疏/稠密/混合三路召回、embedder 类型与维度不匹配报错、缺索引目录的报错可操作、rerank 无模型时降级

几条是回归性质的，对应真会静默出错的地方：跨进程哈希、IDF 符号、倒排区间、embedder 类型混用（不报错只返回垃圾）。

```powershell
python tests\test_retrieve.py
python tests\smoke_retrieve.py --limit 3000
```

