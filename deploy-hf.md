# 部署到 Hugging Face Spaces

目标是拿到一个能点的公网地址。免费，不需要域名备案。

**耗时**：注册 5 分钟 + 推送 10 分钟 + 首次构建 10 分钟。

---

## 0. 前提

```powershell
git lfs version     # 需要 git-lfs，本机已有
```

没有的话：`winget install GitHub.GitLFS` 或去 https://git-lfs.com 下载。

---

## 1. 注册并创建 Space

1. 注册：https://huggingface.co/join （邮箱验证）
2. 创建：https://huggingface.co/new-space

填这四项，其余默认：

| 字段 | 值 |
| --- | --- |
| Space name | `kb` |
| License | `mit` |
| SDK | **Docker** |
| Hardware | **CPU basic · 2 vCPU · 16 GB · FREE** |

Visibility 选 **Public**（要给人看）。

创建完会显示一个空仓库地址，形如：
`https://huggingface.co/spaces/你的用户名/kb`

---

## 2. 把 Space 仓库 clone 下来

```powershell
cd G:\
git clone https://huggingface.co/spaces/你的用户名/kb hf-kb
```

需要登录时：用户名 + **Access Token**（不是密码）。Token 在
https://huggingface.co/settings/tokens 创建，权限选 `write`。

---

## 3. 组装部署目录

在项目根目录执行（脚本从当前项目读代码和索引，往 clone 下来的目录写）：

```powershell
cd G:\kb
python tools\prepare_space.py G:\hf-kb --force
```

预期输出：

```
  ���标目录 : G:\hf-kb
  索引     : G:\kb\data\index
  已跳过   : profile（不公开个人数据）
------------------------------------------------------------
  ingest.py                       387 B
  config.toml                    6.5 KB
  requirements-deploy.txt         680 B
  Dockerfile                      1.9 KB
  .dockerignore                   342 B
  src                          460.6 KB
  tools                         24.6 KB
  data/index                   139.1 MB
  README.md                      1.1 KB
------------------------------------------------------------
  合计 139.6 MB
```

**`已跳过 profile` 这行必须出现** —— 那是你的简历索引，不能公开。
脚本只清目标目录里的内容，**保留 `.git`**，所以不会丢远端关联。

---

## 4. 用 LFS 提交索引

`data/index` 里最大的文件 71 MB，超过 GitHub/HF 的普通文件限制，必须走 LFS。

```powershell
cd G:\hf-kb
git lfs install
git lfs track "*.npy" "*.npz" "*.jsonl"
git add .gitattributes
git add -A
git commit -m "部署：本地知识库问答（Docker + 139 MB 索引）"
git push
```

推送体积约 140 MB，网速慢的话会等几分钟。

---

## 5. 看构建日志

推上去后 HF 会自动开始构建，页面顶部变成 `Building`。

**第一次构建约 5-10 分钟**，其中下载模型（2.1 GB）占大头。构建机在 HF 内网，
直连 `huggingface.co`，不用镜像。

构建成功后状态变 `Running`，访问：

```
https://你的用户名-kb.hf.space
```

---

## 6. 配置 API key（问答需要）

检索不需要 key，但问答要。**不要写进代码或提交到仓库**：

Space 页面 → **Settings** → **Variables and secrets** → **New secret**

| Name | Value |
| --- | --- |
| `DEEPSEEK_API_KEY` | `sk-...` |

保存后 Space 会自动重启，重启完问答就能用了。

---

## 和本地版的差别

| | 本地 | 线上 |
| --- | --- | --- |
| 嵌入推理 | DirectML，4 GB 显卡，单次 153 ms | CPU，2 vCPU，单次约 1-2 秒 |
| 集合 | tech + profile | 只有 tech |
| 语料原文 | 完整，可跳转到源码行 | 不带，跳转按钮失效 |
| 冷启动 | 每次 60-70 秒（CLI） | 首次请求等模型加载，之后常驻 |

**这不是缩水版**：索引是完整的 18,276 块，检索逻辑完全一样，只是推理设备不同。

---

## 可能出的问题

**构建失败：`fetch_model.py` 下载超时**

构建机网络偶发不稳。改 `Dockerfile` 里的 `ARG HF_ENDPOINT`，或在 Space 的
Settings 里加重试。也可以先把模型上传到一个 HF Dataset，构建时从那里拉。

**构建成功但页面 500**

多半是索引没传全。在 Space 的 `Files` 标签页看 `data/index/` 下是不是有 7 个文件
（`embeddings.npy` / `postings.npz` / `chunks_meta.jsonl` / `vocab.json` /
`index_meta.json` / `meta.json` / `dense_meta.json`）。

**检索很慢（每次 5 秒以上）**

免费层 2 vCPU 跑 bge-m3 就是这个速度。想快只能换更小的模型
（`bge-small-zh-int8`，但要重建索引）。

**问答返回 503**

没配 `DEEPSEEK_API_KEY`。检索不受影响。

---

## 想自己本地验证镜像

本机没装 Docker，所以这一步我没法替你验。装了之后：

```powershell
cd G:\hf-kb
docker build -t kb:latest .
docker run --rm -p 7860:7860 -e DEEPSEEK_API_KEY=sk-xxx kb:latest
```

然后开 http://127.0.0.1:7860 看。
