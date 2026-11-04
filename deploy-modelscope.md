# 部署到魔搭创空间

把现有的 Web 界面原样搬到 ModelScope 创空间：免费 2 vCPU / 8 GB、闲置休眠，
和 HF Spaces 同一档。镜像只有一处不同 —— 模型从 `modelscope.cn` 下，不走 hf-mirror。

## 前提

- 魔搭账号（手机号注册，不需要信用卡）
- 本地能跑 `python ingest.py web`
- 空闲磁盘 4 GB：模型不进仓库，但要放 139 MB 的索引和组装目录
- 不需要能访问 `huggingface.co`；构建期从魔搭下载权重

## 1. 组装推送目录

```powershell
cd G:\kb
python tools\prepare_space.py G:\kb-ms --platform modelscope --force
```

预期输出（合计 139.7 MB，模型不在里面）：

```
  目标目录 : G:\kb-ms
  索引     : G:\kb\data\index
------------------------------------------------------------
  ingest.py                       387 B
  config.toml                    6.7 KB
  requirements-deploy.txt         680 B
  Dockerfile                      2.0 KB
  .dockerignore                   524 B
  results/run_hybrid.json        39.8 KB
  docker.yaml                     388 B
  src                           486.6 KB
  tools                          31.9 KB
  data/index                    139.1 MB
  README.md                       1.5 KB
------------------------------------------------------------
  合计 139.7 MB
```

如果本地还有 `data/index/profile`（个人材料），这里会多一行
`已跳过 : profile（不公开个人数据）`，组装时自动剔除。当前仓库里没有个人数据，
所以看不到这行。

`docker.yaml` 声明用 Docker 构建，目标目录里的 `Dockerfile` 是
`Dockerfile.modelscope` 那份（`KB_MODEL_SOURCE=modelscope`）。

## 2. 建创空间

1. 登录 https://modelscope.cn ，进「创空间」→ 新建
2. **先把账号的个人信息补完（验证邮箱）**：平台会提示「个人信息完善后方可使用」，
   没补的话建完也可能不让跑
3. 表单建议值：

| 字段 | 填什么 |
| --- | --- |
| 英文名称 | `kb-qa`（会进 URL，小写字母/数字/连字符） |
| 中文名称 | 本地知识库问答 |
| 是否公开 | **公开**（面试官要能直接点开，非公开需要登录） |
| License | Apache-2.0 即可（代码本身是 MIT） |
| 描述 | 见下 |

描述可以直接用：

```
把 5 个中文技术仓库（1278 篇 / 1150 万字）切成 18276 个块，BM25 + bge-m3 双路召回、
RRF 融合，再交给 DeepSeek 生成带引用的答案，检索不到时明确拒答。单次检索约 1 秒。
```

4. 建好后在仓库页复制 clone 地址，形如
   `https://www.modelscope.cn/studios/<用户名>/<名字>.git`

这一版创建页没有「运行时/SDK」选项，构建方式由仓库里的 `docker.yaml` 决定，
推上去之后平台自动按 Docker 构建。

## 3. 推送

```powershell
cd G:\
git clone https://www.modelscope.cn/studios/<用户名>/<名字>.git kb-ms-repo

# 新仓库里通常带一个模板（app.py / requirements.txt / README.md），
# 不清掉会被一起打进镜像，还可能干扰平台的运行时判断
cd G:\kb-ms-repo
Get-ChildItem -Force | Where-Object { $_.Name -ne '.git' } | Remove-Item -Recurse -Force

robocopy G:\kb-ms G:\kb-ms-repo /E
git lfs install
git lfs track "*.npy" "*.npz" "*.jsonl"
git add .gitattributes
git add -A
git commit -m "deploy: 本地知识库问答"
git push origin master
```

清空时保留 `.git`，远端关联才不会丢。默认分支按页面提示，有的是 `master`
有的是 `main`；`git push` 报错时看一下 `git branch` 再推。

推完确认一遍仓库根目录只有这些：

```
docker.yaml  Dockerfile  .dockerignore  config.toml  ingest.py
requirements-deploy.txt  README.md  src/  tools/  data/index/  results/
```

`data/index` 里应该是 7 个文件（`embeddings.npy` 71.4 MB 最大），没有 `profile/`。

## 4. 部署设置（push 不会自动构建）

推完刷新创空间页面，第 1 步能看到文件列表，但**`git push` 本身不触发构建**：
接口里还是 `AppPort=0`、`IsPublished=false`、`Status=Empty`，必须把第 2、3 步点完。

第 2 步「部署设置」：

| 项 | 值 |
| --- | --- |
| 端口 / AppPort | **7860**，不要留空 —— 平台的转发端口要和镜像 `EXPOSE` 的一致 |
| 实例规格 | 免费 CPU 档（页面写 2 vCPU / 8 GB） |
| 环境变量 | `DEEPSEEK_API_KEY=<你的 key>`；不配也能跑，只是 `/api/ask` 返回 503 |

8 GB 够用：实测服务起来后峰值 2,582 MB（模型 2.18 GB + 索引 139 MB + Python）。
`DEEPSEEK_API_KEY` 放**密文变量**，别放明文变量。

第 3 步「确认并部署」点下去，页面变成构建中。

## 5. 构建与验证

构建日志能逐行看：

- `pip install`：约 1~2 分钟
- 模型下载：2.1 GB，阿里内网 10~60 MB/s，约 1~4 分钟
- 索引随镜像打包：139 MB，秒级
- 首字节等待：镜像起来后 `ingest.py web` 还要加载模型，**40~90 秒**

部署成功后的公开地址形如 `https://<用户名>-<空间名>.ms.show`（页面「访问」按钮
给的链接为准），再查一次接口：

```powershell
curl https://<用户名>-<空间名>.ms.show/api/health
```

应该看到 `"chunks": 18276`、`"ready": true`。

## 6. 配置 API key

问答需要 `DEEPSEEK_API_KEY`。第 2 步的部署设置里能加环境变量就加；加不了的话
`/api/ask` 会返回 503「问答未启用」，检索和「今日复习」照常可用。

## 和 HF Spaces 的差别

| | Hugging Face | 魔搭 |
| --- | --- | --- |
| 免费规格 | 2 vCPU / 8 GB（页面为准） | 2 vCPU / 16 GB |
| 注册 | 邮箱，需要能访问站点（本机卡 418） | 手机号，国内直连 |
| 模型源 | `huggingface.co`（构建机内网） | `modelscope.cn`（构建机内网） |
| 构建配置 | README 的 YAML front-matter | 根目录 `docker.yaml` |
| Dockerfile | `Dockerfile` | `Dockerfile.modelscope`（取名不同，内容只差模型源） |

## 可能踩的坑

| 现象 | 原因 / 处理 |
| --- | --- |
| 「今日复习」提示题库读不出来 | 镜像里没带 `results/`（`Dockerfile` 漏了 `COPY results`）。2026-11-04 已修，重新部署即可 |
| 每条结果下面一行红字「源文件未找到」 | 镜像里 `mkdir data/raw` 造了个空目录，`/api/health` 误报 `raw_dir_exists=true`。已修：不再建这个目录，前端按 false 隐藏提示和「在 VS Code 打开」 |
| 日志 `trailing characters at line ...`（`tokenizer.json`） | 模型文件名冲突导致的拼接损坏，`fetch_model.py` 已修（2026-11-04，按 basename 去重 + 下载后校验 JSON）。重新部署即可 |
| push 被拒，提示文件太大 | 索引单文件最大 71.4 MB（`embeddings.npy`）。确认 git lfs 已生效（`git lfs ls-files`）；仍然被拒就把 `data/index` 拆成两个包，构建时再合并 |
| 构建日志里模型下载超时 | 换回 hf-mirror：`docker.yaml` 仍指向 Dockerfile，把 `Dockerfile` 里的 `KB_MODEL_SOURCE=modelscope` 改成 `HF_ENDPOINT=https://hf-mirror.com`，重新推送 |
| 构建成功但页面 502 | 看日志里 uvicorn 是不是监听在 7860；`docker.yaml` 的 `dockerfile` 路径要和实际文件名一致 |
| 页面第一次打开要等一分钟 | 免费档行为：容器起来后还要加载 2.1 GB 模型，日志里会打「索引就绪 / 模型就绪」 |
| 隔几天打开又要等 | 闲置休眠，访问会唤醒，不是挂了 |

## 已验证 / 未验证

已验证（2026-11-02，本机）：

- 组装目录能直接当产物跑：`/api/health` 返回 18,276 块、只有 tech 集合、模型加载成功；
  `/api/review/today` 正常出题（说明题库文件确实被打进目录）；`/api/search` 能检索
- 2026-11-03 已 push 成功：仓库里 69 个条目，`data/index` 7 个文件、
  `embeddings.npy` 74,858,624 字节，`docker.yaml` / `.gitattributes` 都在
- 魔搭文件接口能列出 30 个文件，`onnx/model.onnx_data` 是 2,266,820,608 字节，
  和本地那份一致；单文件下载、截断到 200 字节后的续传都验过
- 组装脚本会跳过 `data/index/profile`（个人材料）。本地已经删掉简历数据和索引，
  所以现在组装时看不到这行提示

未验证：

- **Docker 构建本身**：这台机器没装 Docker。`pip install` 的 Linux wheel、2.1 GB
  权重的完整下载、镜像体积，都要等创空间第一次的构建日志
- 首次访问的实际等待时间（按本机 44 秒模型加载 + 容器启动估，1 分钟左右）

## 回滚

整个流程只写 `G:\kb-ms`（组装目录）和远端创空间仓库，项目本身不动。
不想用了就在创空间里删除，本地删掉组装目录即可。
