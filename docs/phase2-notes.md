# Phase 2 环境与选型记录

记录检索层落地过程中被这台机器的真实条件逼出来的决策。每条都有实测数据支撑。

---

## 一、环境事实（和原计划的前提不同）

| 项 | 实际情况 | 影响 |
| --- | --- | --- |
| 显卡 | **GTX 1650 Ti，4096 MiB** | 不是原以为的 ≥8GB。bge-m3 + 重排 + 生成模型无法同时常驻 |
| Docker | **未安装** | 排除 Qdrant 一类需要容器的向量库 |
| torch / transformers / sentence-transformers | **未安装，且装不上** | pip 在本机解包阶段被沙箱拒绝，只能手动解压 wheel |
| onnxruntime | **已装 1.27.0（仅 CPU 后端）** | 走 ONNX 路线，不需要 torch |
| `huggingface.co` | **不通** | 模型从 `hf-mirror.com` 下 |
| `pypi.org` / 清华源 | 通 | 依赖可以下 wheel 手动装 |
| Python | 3.13.9（Anaconda） | — |
| 逻辑核心 | 16 | — |
| G: 可用空间 | 74 GB | — |

**必要条件**：`onnxruntime` + `tokenizers`（2.7 MB wheel，手动解压到 `.pylibs`），
不需要 torch，不需要 CUDA toolkit。

---

## 二、向量库：不用，直接暴力检索

18,276 个块 × 1024 维 float32 = **75 MB**。一次全量余弦相似度就是一次矩阵乘法，
CPU 上几十毫秒。这个规模上引入向量库（Qdrant / Milvus / Chroma）只增加组件和故障面，
不带来任何收益。

`DenseIndex` 因此是 `embeddings.npy` + `meta.jsonl`，检索时 `emb @ q` 加 `argpartition` 取 top-k，
支持 `mmap_mode="r"` 降低内存占用。

**什么时候该换向量库**：块数到百万级、或者需要按元数据做强过滤再检索时。
这条判断可以直接在面试里讲。

---

## 三、嵌入模型：bge-m3，但要靠 DirectML 才跑得动

### 速度实测（真实语料块，字符 p50 ≈ 550，max_length=512）

| 模型 | 参数量 | 体积 | 速度 | 全量 18,276 块 | 领域区分度 |
| --- | --- | --- | --- | --- | --- |
| bge-m3（CPU） | 568M | 2.18 GB fp32 | 0.8 块/秒 | **398 分钟** | 5/5 |
| bge-m3（DirectML，batch=4） | 568M | 2.18 GB fp32 | **7.6 块/秒** | **40 分钟** | 5/5 |
| bge-base-zh int8（CPU） | 102M | 98 MB | 4.7 块/秒 | 65 分钟 | 3/5 |
| bge-small-zh int8（CPU） | 24M | 23 MB | 23.2 块/秒 | 13 分钟 | 1/5 |

「领域区分度」是 5 个领域（HashMap / TCP / LoRA / Redis / JVM）各 3 条语料、
5 个查询，同领域 3 条全部排进 top-3 记 1 分。

### 结论

bge-m3 的质量明显最好（5/5），小模型掉得厉害（bge-small 只有 1/5）。
但 CPU 上 6.6 小时不可用 —— **每改一次切块参数就要重建索引**，6.6 小时等于不能迭代。

解法是 **DirectML**：把 `onnxruntime-directml` 解压到 `.pylibs-dml`，
代码在 `import onnxruntime` 之前把它插到 `sys.path` 最前面，就会用上显卡。
DirectML 走 DirectX 12，**和 CUDA 无关**，Windows 上不用装 CUDA toolkit 就能用较新的显卡。

关键验证：**DirectML 与 CPU 的逐条余弦 = 1.000000**，GPU 路径没有数值偏差。

### 备选方案（保留在配置里）

`Xenova/bge-small-zh-v1.5`、`Xenova/bge-base-zh-v1.5` 有官方量化 ONNX 导出
（int8 / fp16 / q4），下下来就能用，不需要 torch 也不需要导出工具。
质量不足时再换回 bge-m3。

---

## 四、池化：直接取 `sentence_embedding`，不要自己池化

这是本轮最值得记的一条。我最初在实现规格里写了「mean pooling」，**这是错的**。

bge-m3 的 ONNX 导出有两个输出：

```
token_embeddings     (B, L, 1024)      每个 token 的隐状态
sentence_embedding   (B, 1024)         已池化、已归一化
```

实测 `sentence_embedding` 与手算 CLS pooling 的余弦 **恒为 1.0**，与 mean pooling 只有 **0.79**。
也就是说 bge 用的是 **CLS pooling**，而且导出时已经把归一化做完了。

所以正确做法是**直接取 `sentence_embedding`**，既不用池化也不用归一化。
代码里 `_pool()` 的优先级是 `sentence_embedding` > `pooler_output` > `text_embeds`，
取不到才退回自己 mean pooling —— 那条路是给非标准导出兜底的，效果会打折。

**怎么发现这个问题的**：一开始用 mean pooling 跑领域区分度只拿到 1/5~2/5，
换成 `sentence_embedding` 之后是 5/5。没有靠猜，是拿两个张量互相算余弦确认的。

---

## 五、踩过的坑

1. **`pip install` 在这台机器上不可用**：不是网络问题 —— wheel 下载正常，
   是解包阶段写 `.whl.metadata` 被沙箱拒绝。所有依赖改用「下 wheel + `zipfile` 解压」。
   顺带发现 `tempfile` 的默认目录（`D:\系统缓存\...`）也不可写，pytest 的 `tmp_path` 同样失效。

2. **无 BOM 的 `.ps1` 会被按 ANSI 解码**：Windows PowerShell 5.1（本机不是 pwsh 7）
   按系统代码页 936 解码脚本，中文注释后的换行被多字节前导字节吞掉，
   行号整体错位，报出来是一堆 "Unexpected token"。所以 `scripts/*.ps1` 一律保持纯 ASCII。

3. **`alembic.ini` 不能写中文注释**：Alembic 按系统 locale 编码读它，中文 Windows 下直接
   `UnicodeDecodeError`。

4. **pip 选的 wheel 可能是别的平台**：早期版本的安装脚本直接取 `urls[0]`，
   在 Windows 上抓到了 macos 的包。现在按 `win_amd64` + `cp313`/`abi3` 打分挑选。

5. **多个进程同时写同一个工作副本会互相踩**：本次有两个子代理并行工作在同一目录，
   其中一个报告 `src/kb/retrieve/` 被覆盖过又删过。最终状态经逐文件核验是完整的，
   但并行任务应当各自用独立目录。

---

## 六、复现步骤

```powershell
# 1. tokenizers（ONNX 分词器，2.7 MB）
python install_dep.py tokenizers <项目根>\.pylibs

# 2. onnxruntime-directml（用上显卡，23.9 MB）
python install_dep.py onnxruntime-directml <项目根>\.pylibs-dml

# 3. bge-m3 的 ONNX（2.18 GB，从 hf-mirror）
python fetch_model.py <项目根>\models BAAI/bge-m3

# 4. 建索引（约 40-50 分钟，会用到 DirectML）
python ingest.py index --embedder onnx --model-dir <项目根>\models\bge-m3 `
    --index-dir <项目根>\data\index --batch-size 4

# 5. 评测
python eval/run_eval.py --index-dir <项目根>\data\index --top-k 10
```

如果 `.pylibs-dml` 不存在，代码会自动退回 CPU，不会报错 —— 只是慢 9.5 倍。
用 `KB_DML_LIBS=<目录>` 可以指定别的位置。
