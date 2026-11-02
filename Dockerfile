# 本地知识库问答系统的部署镜像
#
# 目标是 Hugging Face Spaces（Docker SDK），也能在任意装了 Docker 的机器上跑：
#
#   docker build -t kb:latest .
#   docker run --rm -p 7860:7860 kb:latest
#
# 目录约定（构建时必须存在）：
#   src/                     代码
#   tools/fetch_model.py     模型下载脚本
#   data/index/              检索索引，tech 这一套（约 139 MB，走 git lfs）
#   config.toml              配置
#
# 不放进镜像的：data/raw（语料原文，几百 MB）、data/index/profile（个人简历）、
# models/（构建期重新下载）、.pylibs*（本机手动解的 wheel）

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf-cache \
    PORT=7860

WORKDIR /app

# 依赖单独一层：只改代码时不会重装
COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY src ./src
COPY tools ./tools
COPY ingest.py config.toml ./

# 模型在构建期下好，别让第一个访客等 2.2 GB 下载。
# HF Spaces 的构建机连官方域名更快；本地构建可覆盖：
#   docker build --build-arg HF_ENDPOINT=https://hf-mirror.com -t kb .
ARG HF_ENDPOINT=https://huggingface.co
RUN HF_ENDPOINT=${HF_ENDPOINT} python tools/fetch_model.py models BAAI/bge-m3

# 索引跟着镜像走。139 MB，靠 git lfs 提交到 Space 仓库。
COPY data/index ./data/index

# 语料根目录不存在也能起服务，只是「在编辑器打开源码」按钮会失效
# （容器里本来也没有编辑器）。create_app 的 /api/health 会如实报告
# raw_dir_exists=false。
RUN mkdir -p data/raw

EXPOSE 7860

# --no-browser 必须带：容器里没有图形界面，webbrowser.open 只会报错
CMD ["sh", "-c", "python ingest.py web --no-browser --host 0.0.0.0 --port ${PORT:-7860}"]
