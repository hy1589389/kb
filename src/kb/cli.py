"""kb 命令行入口：出问题只看这一个文件就能定位流程。

子命令：
    ingest   遍历语料目录，切块并输出 JSONL + 统计      （Phase 1）
    index    建稠密 + 稀疏检索索引                        （Phase 2）
    search   查询索引，人类可读或 --json 输出              （Phase 2）
    ask      检索 + 大模型生成带引用的答案                 （Phase 4）
    web      启动本地 Web 界面（检索 / 问答 / 打开源文件）  （Phase 5）
    drop     按文件或 doc_id 从一个集合里删文档并重建索引   （Phase 1+2）

多集合：`ingest` / `index` / `search` / `drop` 都支持 `--collection`，取值是
逗号分隔的集合名（`tech` / `papers` / `profile`，见 config.toml 的
`[collections]` 段）。不写 `--collection` 时：

    ingest  入所有集合
    index   保持旧行为：从 chunks_all.jsonl 建单个索引到 [retrieval].index_dir
    search  查 [collections].default_search 里的集合
    drop    必须显式指定（删除是破坏性操作）
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

from .config import PDF_SUFFIXES, IngestConfig
from .ingest.pipeline import build_stats, write_outputs

# src/kb/cli.py -> src/kb -> src -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"


def _fmt(n: int) -> str:
    return f"{n:,}"


# --------------------------------------------------------------------------- #
# 集合选择
# --------------------------------------------------------------------------- #
def _split_names(value: str | list[str] | None) -> list[str]:
    """`--collection tech,profile` / `["tech"]` / None -> 名字列表。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    out: list[str] = []
    for item in value:
        out.extend(_split_names(item))
    return out


def _select_collections(cfg: IngestConfig, spec, *,
                        default: list[str] | None = None
                        ) -> list:
    """按 `--collection` 选集合；没给时用 default / default_search。"""
    names = _split_names(spec)
    if not names:
        names = default if default is not None else list(cfg.collections.default_search)
    return cfg.collections.select(names)


def _print_collection_head(cfg: IngestConfig, collections: list) -> None:
    for coll in collections:
        print(f"  集合 {coll.name:<8} parser={coll.parser:<8} "
              f"generation={coll.generation}")
        print(f"       raw_dir  : {coll.raw_dir}")
        print(f"       index_dir: {coll.index_dir}")


def cmd_ingest(args: argparse.Namespace) -> int:
    # 注意：这里必须回退到项目根的 config.toml（_load_ingest_config 就是这么做的）。
    # 旧版本写的是 IngestConfig.load(None)，等于只用代码默认值，config.toml 里的
    # [path] / [chunk] / [collections] 全被忽略 —— 不修的话 --collection 永远找不到集合。
    cfg = _load_ingest_config(args)

    if args.out:
        cfg.out_dir = Path(args.out)
    if args.max_chars is not None:
        cfg.chunk.max_chars = args.max_chars
    if args.min_chars is not None:
        cfg.chunk.min_chars = args.min_chars
    if args.overlap is not None:
        cfg.chunk.overlap = args.overlap
    cfg.resolve(PROJECT_ROOT)

    try:
        # 不写 --collection 时入「所有集合」；`ingest` 是构建入口，
        # 用 default_search 筛会让人以为简历没入库。
        selected = _select_collections(cfg, args.collection,
                                       default=cfg.collections.names())
    except KeyError as exc:
        print(f"[X] {exc.args[0] if exc.args else exc}")
        return 2

    # --raw 覆盖所选集合的 raw_dir（旧的单集合用法要能继续跑）
    if args.raw:
        for coll in selected:
            coll.raw_dir = Path(args.raw).resolve()

    dry_run = bool(getattr(args, "dry_run", False))

    print("=" * 68)
    print("  Phase 1 · 语料入库" + ("（dry-run，不写任何文件）" if dry_run else ""))
    print("=" * 68)
    _print_collection_head(cfg, selected)
    print(f"  输出目录 : {cfg.out_dir}")
    print(f"  切块参数 : max={cfg.chunk.max_chars} min={cfg.chunk.min_chars} "
          f"overlap={cfg.chunk.overlap} drop_below={cfg.chunk.drop_below}")
    print("-" * 68)

    import time

    from .ingest.pipeline import ingest_collection, print_reports

    any_docs = False
    exit_code = 0
    for coll in selected:
        print(f"[集合 {coll.name}]")
        if not coll.raw_dir.exists():
            print(f"  [X] 原始语料目录不存在：{coll.raw_dir}")
            exit_code = exit_code or 2
            continue

        t0 = time.perf_counter()
        try:
            result = ingest_collection(cfg, coll, dry_run=dry_run)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            # 例如没装 PyMuPDF（PdfUnavailableError 是 RuntimeError）：给出
            # 一句能照着做的提示，别让整个 ingest 拖着栈退出。
            print(f"  [X] 集合 {coll.name} 入库失败：{exc}")
            exit_code = 2
            continue
        elapsed = time.perf_counter() - t0

        if dry_run:
            print_reports(result)

        if not result.docs:
            print(f"  [!] 集合 {coll.name} 没有解析出任何文档")
            continue
        any_docs = True

        stats = build_stats(result, cfg, elapsed)
        written = write_outputs(result, cfg, stats) if not dry_run else {}

        t = stats["totals"]
        cs = stats["chunk_size"]
        files = stats["files"]
        print("-" * 68)
        print(f"  入库结果（{coll.name}）")
        print("-" * 68)
        print(f"  仓库        : {t['repos']}")
        print(f"  文档        : {_fmt(t['docs'])} 篇")
        print(f"  切块        : {_fmt(t['chunks'])} 个")
        print(f"  总字符      : {_fmt(t['chars'])}  （其中中文 {_fmt(t['cjk_chars'])}）")
        print(f"  估算 token  : {_fmt(t['est_tokens'])}")
        print(f"  块长分布    : min {cs['min']} / p50 {cs['p50']} / p90 {cs['p90']} "
              f"/ p95 {cs['p95']} / max {cs['max']}")
        print(f"  文件        : 新增 {files['added']}  变更 {files['updated']}  "
              f"未变更 {files['unchanged']}  移除 {files['removed']}  "
              f"跳过 {files['skipped']}")
        print(f"  耗时        : {stats['elapsed_sec']}s")
        if stats["skipped_summary"]:
            print(f"  跳过原因    : {stats['skipped_summary']}")
        if stats["low_quality_docs"]:
            print(f"  [!] 低质量文档 {len(stats['low_quality_docs'])} 篇"
                  f"（已标记 low_quality，检索默认排除）")
            for doc in stats["low_quality_docs"][:5]:
                print(f"      - {doc['rel_path']}")
        if dry_run:
            print("  dry-run：未写任何文件")
        else:
            print("-" * 68)
            print(f"  chunks    -> {written['chunks']}")
            print(f"  manifest  -> {written['manifest']}")
            print(f"  指纹      -> {written['fingerprints']}")
            print(f"  stats     -> {written['stats']}")
            if "chunks_all" in written:
                print(f"  全量 JSONL-> {written['chunks_all']}"
                      f"   （tech 沿用旧入口，勿删）")
        print("=" * 68)

    if not any_docs:
        print("[X] 没有解析出任何文档，请检查语料目录与 parser 配置。")
        return 1 if exit_code == 0 else exit_code
    return exit_code


def _load_retrieval_config(args: argparse.Namespace):
    """读 config.toml 的 [retrieval] 段，再被命令行参数覆盖。

    命令行优先级高于配置文件，配置文件的优先级高于代码里的默认值 ——
    这样「跑一次实验」不需要改文件，「长期设置」不需要每次敲参数。
    """
    cfg_path = Path(args.config) if getattr(args, "config", None) else DEFAULT_CONFIG
    ingest_cfg = IngestConfig.load(cfg_path)
    rc = ingest_cfg.retrieval
    rc.resolve(PROJECT_ROOT)

    if getattr(args, "index_dir", None):
        rc.index_dir = Path(args.index_dir).resolve()
    if getattr(args, "model_dir", None):
        rc.model_dir = Path(args.model_dir)
    if getattr(args, "reranker_dir", None):
        rc.reranker_dir = Path(args.reranker_dir)
    if getattr(args, "batch_size", None):
        rc.embed_batch_size = int(args.batch_size)
    if getattr(args, "max_length", None):
        rc.embed_max_length = int(args.max_length)
    return rc


def _load_ingest_config(args: argparse.Namespace) -> IngestConfig:
    """读整份 config.toml（[path] / [web] 等）。

    `[retrieval]` 段仍走 `_load_retrieval_config`，那边有命令行覆盖逻辑；
    这里只服务「语料根目录」和 Web 配置这类不需要命令行逐项覆盖的字段。
    """
    cfg_path = Path(args.config) if getattr(args, "config", None) else DEFAULT_CONFIG
    return IngestConfig.load(cfg_path)


def _resolve_raw_dir(args: argparse.Namespace) -> Path:
    """语料根目录：`--raw-dir` 优先，否则用 `[path].raw_dir`（相对项目根展开）。

    `<raw_dir>/<repo>/<rel_path>` 就是检索结果对应的真实文件。
    """
    if getattr(args, "raw_dir", None):
        return Path(args.raw_dir).resolve()
    raw = _load_ingest_config(args).raw_dir
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def _chunks_path_for(cfg: IngestConfig, collection: str) -> Path:
    return Path(cfg.out_dir) / f"chunks_{collection}.jsonl"


def _embedder_kind_of_index(index_dir: Path, fallback: str = "onnx") -> str:
    """按 `index_meta.json` 里记录的类型决定用哪个 embedder。

    重建索引时必须和建索引时一致：hashing 索引换成 onnx 查询，向量空间不同，
    不报错但全是噪声（`Retriever` 有一道断言，这里提前选对省一次失败）。
    """
    try:
        info = json.loads((Path(index_dir) / "index_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
    recorded = str(info.get("embedder") or "").lower()
    if "hashing" in recorded:
        return "hashing"
    return fallback


def _build_one_index(args: argparse.Namespace, rc, chunks_path: Path,
                     index_dir: Path, *, embedder_kind: str,
                     label: str = "") -> int:
    """建一个索引目录并打印结果，返回退出码。`label` 是集合名前缀。"""
    from .retrieve.pipeline import _make_embedder, build_index, load_chunks, \
        write_limited_chunks

    embedder_kind = embedder_kind or "onnx"
    limit = getattr(args, "limit", None)
    print(f"  语料     : {chunks_path}")
    print(f"  索引目录 : {index_dir}")
    print(f"  embedder : {embedder_kind}"
          + (f"  (model_dir={rc.model_dir})" if embedder_kind == "onnx" else ""))
    print(f"  批大小   : {rc.embed_batch_size}   截断长度: {rc.embed_max_length}"
          + (f"   limit={limit}" if limit else ""))
    print("-" * 68)

    try:
        if embedder_kind == "hashing":
            # HashingEmbedder 的 idf 要从语料现算，所以必须先读一遍文本。
            # 用 --limit 时只读前 N 条，否则 44 MB 全读进来只为统计 df。
            # 建索引那一步会再读一遍同样的文件（idf 必须先于编码算完，
            # 顺序无法合并），多读一次约 1 秒，可接受。
            chunks = load_chunks(chunks_path, limit=limit)
            embedder = _make_embedder(
                "hashing", None, rc.embed_max_length, rc.embed_batch_size,
                texts_for_idf=[c.get("embed_text") or c.get("text") or "" for c in chunks],
            )
            del chunks
        else:
            embedder = _make_embedder("onnx", rc.model_dir, rc.embed_max_length,
                                      rc.embed_batch_size)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[X] {label}{exc}")
        return 2

    # hashing 且限量的场景下，build_index 也要限量：把前 N 条写成一个临时
    # JSONL，避免为了自检把 18k 条全嵌一遍（也无从「只嵌前 N 条」）。
    # 临时文件放在索引目录内，结束就删，不会污染语料目录。
    effective_chunks = chunks_path
    tmp_path = None
    if limit:
        tmp_path = index_dir / "_limit_chunks.jsonl"
        written = write_limited_chunks(chunks_path, tmp_path, limit)
        effective_chunks = tmp_path
        print(f"  自检模式 : 只处理前 {written:,} 条")

    try:
        info = build_index(
            effective_chunks, index_dir, embedder,
            batch_size=rc.embed_batch_size,
            max_length=rc.embed_max_length,
            progress=True,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[X] {label}建索引失败：{exc}")
        return 2
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    print("-" * 68)
    print(f"  块数        : {info['n_chunks']:,}")
    print(f"  嵌入维度    : {info['dim']}")
    print(f"  词表        : {info['vocab_size']:,} term / {info['n_postings']:,} posting")
    print(f"  平均块长    : {info['avgdl']} token")
    print(f"  耗时        : {info['elapsed_sec']}s")
    print(f"  索引目录    : {index_dir}")
    print("-" * 68)
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    rc = _load_retrieval_config(args)
    names = _split_names(getattr(args, "collection", None))

    print("=" * 68)
    print("  Phase 2 · 建检索索引")
    print("=" * 68)

    if not names:
        # 旧行为：单个索引，语料默认 chunks_all.jsonl，目录取 [retrieval].index_dir
        chunks_path = Path(args.chunks).resolve() if args.chunks else (
            PROJECT_ROOT / "data" / "processed" / "chunks_all.jsonl"
        )
        code = _build_one_index(args, rc, chunks_path, rc.index_dir,
                                embedder_kind=args.embedder)
        if code == 0:
            print(f"  试查        : python ingest.py search \"三次握手\" "
                  f"--index-dir \"{rc.index_dir}\"")
        print("=" * 68)
        return code

    cfg = _load_ingest_config(args)
    if getattr(args, "out", None):
        cfg.out_dir = Path(args.out)
    cfg.resolve(PROJECT_ROOT)
    try:
        selected = _select_collections(cfg, names)
    except KeyError as exc:
        print(f"[X] {exc.args[0] if exc.args else exc}")
        return 2
    if args.index_dir:
        print("[!] --index-dir 只在单索引模式下生效，已忽略")

    code = 0
    for coll in selected:
        print(f"[集合 {coll.name}]")
        chunks_path = _chunks_path_for(cfg, coll.name)
        if not chunks_path.exists():
            print(f"  [X] 没有 {chunks_path.name}，先入库："
                  f"python ingest.py ingest --collection {coll.name}")
            code = 2
            continue
        kind = args.embedder or _embedder_kind_of_index(coll.index_dir)
        code = _build_one_index(args, rc, chunks_path, coll.index_dir,
                                embedder_kind=kind,
                                label=f"[{coll.name}] ") or code
        print("=" * 68)
    return code


def _resolve_search_indexes(args: argparse.Namespace, cfg: IngestConfig):
    """决定这次查哪些索引，返回 `(index_arg, labels, selected, rc_weights)`。

    · 给了 `--index-dir` 且没给 `--collection`：旧的单索引模式，直接用一个目录。
    · 否则按 `--collection` / `[collections].default_search` 挑集合，只保留
      已经建过索引的那些（没建索引的集合打印提示后跳过，不整体失败）。
    """
    names = _split_names(getattr(args, "collection", None))
    if getattr(args, "index_dir", None) and not names:
        path = Path(args.index_dir).resolve()
        return path, [str(path)], [], {}

    selected = _select_collections(cfg, names)
    index_map: dict[str, Path] = {}
    missing: list[str] = []
    for coll in selected:
        if (coll.index_dir / "index_meta.json").exists():
            index_map[coll.name] = coll.index_dir
        else:
            missing.append(coll.name)
    for name in missing:
        print(f"[!] 集合 {name} 还没建索引，已跳过"
              f"（python ingest.py index --collection {name}）")
    if not index_map:
        hint = selected[0].name if selected else "profile"
        raise FileNotFoundError(
            f"选中的集合都没有索引（{', '.join(c.name for c in selected) or '空'}）。"
            "先入库再建索引：\n"
            f"    python ingest.py ingest --collection {hint}\n"
            f"    python ingest.py index  --collection {hint}"
        )
    labels = [f"{name} ({path})" for name, path in index_map.items()]
    weights = {c.name: float(c.weight) for c in selected if c.name in index_map}
    return index_map, labels, selected, weights


def _read_manifest(cfg: IngestConfig, collection: str) -> list[dict]:
    path = Path(cfg.out_dir) / f"manifest_{collection}.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _raw_rel_from_manifest(row: dict) -> str:
    """manifest 行 -> 相对 raw_dir 的路径（与 ingest 的指纹 key 一致）。"""
    rel = str(row.get("rel_path", "") or "")
    repo = str(row.get("repo", "") or "")
    if Path(rel).suffix.lower() in PDF_SUFFIXES:
        return rel
    return f"{repo}/{rel}" if repo else rel


def _match_manifest(manifest: list[dict], spec: str, raw_dir: Path) -> tuple[str, list[str]]:
    """把 `--path` 解析成 raw_dir 相对路径。

    支持三种写法：相对 raw_dir 的路径、`<repo>/<rel_path>`、只写文件名。
    返回 `(命中的路径, 候选列表)`；候选多于一个时由调用方报歧义。
    """
    spec_norm = spec.replace("\\", "/").strip()
    while spec_norm.startswith("./"):
        spec_norm = spec_norm[2:]
    wanted_name = Path(spec_norm).name

    exact: list[str] = []
    by_name: list[str] = []
    for row in manifest:
        rel = str(row.get("rel_path", "") or "")
        repo = str(row.get("repo", "") or "")
        raw_rel = _raw_rel_from_manifest(row)
        forms = {rel, raw_rel, f"{repo}/{rel}" if repo else rel}
        if spec_norm in forms:
            exact.append(raw_rel)
        elif wanted_name and wanted_name in {Path(rel).name, Path(raw_rel).name}:
            by_name.append(raw_rel)

    candidates: list[str] = []
    for item in exact or by_name:
        if item not in candidates:
            candidates.append(item)
    if candidates:
        return candidates[0], candidates

    # manifest 里没有（还没入过库的文件）：当成 raw_dir 下的相对路径
    direct = Path(spec_norm)
    if not direct.is_absolute():
        direct = raw_dir / direct
    direct = direct.resolve()
    try:
        return direct.relative_to(raw_dir.resolve()).as_posix(), []
    except ValueError:
        return "", []


def cmd_drop(args: argparse.Namespace) -> int:
    """从一个集合里删掉一份文档，然后重建**这个集合**的入库结果和索引。

    为什么用「删文件 + 重跑集合」而不是从索引矩阵里删行：
      · 个人资料集合只有几十块，全量重跑 2 秒；矩阵删行要重写 embeddings /
        postings / meta 三个文件，出错概率高得多；
      · 重跑走的是同一套指纹逻辑，chunks / manifest / 指纹三者不会不同步。

    tech 的索引（`data/index`）只在该集合就是 `--collection tech` 时才会被重建，
    删 profile 永远碰不到它。
    """
    import shutil
    import time

    from .ingest.pipeline import build_stats, ingest_collection, legacy_index_dir, \
        write_outputs

    cfg = _load_ingest_config(args)
    if getattr(args, "out", None):
        cfg.out_dir = Path(args.out)
    cfg.resolve(PROJECT_ROOT)

    names = _split_names(args.collection)
    if not names:
        print("[X] drop 需要显式指定 --collection（删除是破坏性操作）")
        return 2
    try:
        selected = _select_collections(cfg, names)
    except KeyError as exc:
        print(f"[X] {exc.args[0] if exc.args else exc}")
        return 2
    if len(selected) != 1:
        print("[X] drop 一次只能处理一个集合，收到：" + ", ".join(names))
        return 2
    coll = selected[0]
    raw_dir = Path(coll.raw_dir).resolve()

    if not args.path and not args.doc_id:
        print("[X] 需要 --path 或 --doc-id 指定要删的文档")
        return 2

    manifest = _read_manifest(cfg, coll.name)
    if args.doc_id:
        matches = [row for row in manifest if str(row.get("doc_id", "")) == args.doc_id]
        if not matches:
            print(f"[X] 集合 {coll.name} 的 manifest 里没有 doc_id={args.doc_id}")
            return 2
        candidates = [_raw_rel_from_manifest(row) for row in matches]
    else:
        _, candidates = _match_manifest(manifest, args.path, raw_dir)

    if not candidates:
        print(f"[X] 集合 {coll.name} 里找不到 {args.path!r}（raw_dir={raw_dir}）")
        print("    提示：--path 可以写相对 raw_dir 的路径、<repo>/<rel_path>，或只写文件名")
        return 2
    if len(candidates) > 1:
        print(f"[X] {args.path!r} 匹配到多个文件，请写得更具体：")
        for item in candidates:
            print(f"      {item}")
        return 2

    raw_rel = candidates[0]
    target = (raw_dir / raw_rel).resolve()
    if not target.is_relative_to(raw_dir):
        print(f"[X] 目标文件不在 raw_dir 里，拒绝删除：{target}")
        return 2
    if not target.is_file():
        print(f"[X] 文件不存在：{target}")
        return 2

    index_dir = Path(coll.index_dir).resolve()
    is_legacy = index_dir == legacy_index_dir(cfg)

    print("=" * 68)
    print("  drop · 删除文档并重建集合")
    print("=" * 68)
    print(f"  集合     : {coll.name}")
    print(f"  文件     : {target}")
    print(f"  索引目录 : {index_dir}"
          + ("   （data/index，tech 的旧索引）" if is_legacy else ""))
    print("-" * 68)

    if not args.yes:
        try:
            answer = input(f"确认从集合 {coll.name} 删除这份文档？[y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("已取消，未改任何文件。")
            return 1

    try:
        target.unlink()
    except PermissionError:
        # 从别处复制过来的文件常带 Windows 的 ReadOnly 属性，直接 unlink 会
        # 报「拒绝访问」。先清掉只读位再删一次；还失败就当作被占用，给提示。
        try:
            os.chmod(target, stat.S_IWRITE)
            target.unlink()
        except OSError as exc:
            print(f"[X] 删除失败（文件只读或被占用）：{exc}")
            return 2
    except OSError as exc:
        print(f"[X] 删除失败：{exc}")
        return 2
    print(f"[删除] {target}")

    t0 = time.perf_counter()
    result = ingest_collection(cfg, coll, verbose=True)
    elapsed = time.perf_counter() - t0
    stats = build_stats(result, cfg, elapsed)
    write_outputs(result, cfg, stats)
    removed = raw_rel in result.removed or not target.exists()
    print(f"  剩余文档 : {len(result.docs)} 篇 / {len(result.chunks)} 块"
          f"（本次删除的块已同步移除：{'是' if removed else '否'}）")

    if not result.chunks:
        if is_legacy:
            print("[!] 集合已空，但它用的是 data/index，索引不会自动删除。")
            return 0
        shutil.rmtree(index_dir, ignore_errors=True)
        print(f"  集合已空，已删除索引目录 {index_dir}")
        print("=" * 68)
        return 0

    rc = _load_retrieval_config(args)
    kind = getattr(args, "embedder", None) or _embedder_kind_of_index(index_dir)
    print("-" * 68)
    print("  重建该集合的索引")
    print("-" * 68)
    code = _build_one_index(args, rc, _chunks_path_for(cfg, coll.name), index_dir,
                            embedder_kind=kind, label=f"[{coll.name}] ")
    print("=" * 68)
    return code


def cmd_search(args: argparse.Namespace) -> int:
    from .retrieve.embedder import HashingEmbedder
    from .retrieve.pipeline import Retriever

    rc = _load_retrieval_config(args)
    ingest_cfg = _load_ingest_config(args)
    if args.out:
        ingest_cfg.out_dir = Path(args.out)
    ingest_cfg.resolve(PROJECT_ROOT)

    # 命令行 --rerank 优先于配置；配置里 use_reranker=false 时不重排
    want_rerank = bool(args.use_rerank or rc.use_reranker)
    want_hybrid = not args.no_dense and not args.no_sparse
    mode = "hybrid" if want_hybrid else ("dense" if not args.no_dense else "sparse")
    include_low = bool(getattr(args, "include_low_quality", False))

    try:
        index_arg, labels, _selected, weights = _resolve_search_indexes(args, ingest_cfg)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[X] {exc.args[0] if exc.args else exc}")
        return 2

    cfg = rc.to_retriever_cfg()
    if args.use_rerank:
        cfg["use_reranker"] = True
    if isinstance(index_arg, dict):
        cfg["collections"] = weights

    # 判断索引用的是哪种 embedder：hashing 索引不能用 onnx 查询（向量空间不同），
    # 反之亦然。Retriever 里也有一道检查，这里提前决定省一次加载。
    embedder = None
    dirs = index_arg.values() if isinstance(index_arg, dict) else [index_arg]
    for directory in dirs:
        try:
            info = json.loads((directory / "index_meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"[X] {directory / 'index_meta.json'} 读不出来，索引可能不完整，请重建。")
            return 2
        if embedder is None and "hashing" in str(info.get("embedder", "")).lower():
            embedder = HashingEmbedder(dim=int(info.get("dim", 512)))

    try:
        retriever = Retriever(index_arg, embedder=embedder, cfg=cfg)
        hits = retriever.search(
            args.query, top_k=args.top_k,
            use_dense=not args.no_dense,
            use_sparse=not args.no_sparse,
            use_rerank=want_rerank,
            collections=list(index_arg) if isinstance(index_arg, dict) else None,
            include_low_quality=include_low,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[X] {exc}")
        return 2

    if args.json:
        payload = {
            "query": args.query,
            "index_dir": str(index_arg) if not isinstance(index_arg, dict) else "",
            "collections": list(index_arg) if isinstance(index_arg, dict) else [],
            "include_low_quality": include_low,
            "n_hits": len(hits),
            "hits": [h.to_dict() for h in hits],
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        if args.open_index is not None:
            # stdout 已经是一份 JSON，打开编辑器的说明只能走 stderr，
            # 否则管道下游会拿到半截 JSON。
            print("[!] --json 模式下 --open 的输出改走 stderr", file=sys.stderr)
            return _open_hit(args, hits, out=sys.stderr)
        return 0

    if not hits:
        print(f"没有命中：{args.query}")
        print("（检查 --no-dense / --no-sparse 是否把两路都关了，或者换个查询词）")
        if not include_low:
            print("（低质量块已按默认排除；要复查可加 --include-low-quality）")
        if args.open_index is not None:
            print(f"[X] --open {args.open_index} 没有可打开的结果：本次 0 条")
            return 1
        return 0

    print(f"查询: {args.query}")
    print(f"集合: {', '.join(labels)}")
    print(f"返回 {len(hits)} 条   模式: {mode}"
          f"{' + rerank' if want_rerank else ''}"
          f"{'' if include_low else '   低质量块: 已排除'}")
    print("=" * 68)
    for hit in hits:
        tag = f"[{hit.collection}] " if hit.collection else ""
        print(f"[{hit.rank:>2}] {hit.score:>8.4f}  {tag}{hit.breadcrumb or '(无标题)'}")
        print(f"     {hit.location()}")
        print(f"     {_summarize(hit.text)}")
        print("-" * 68)
    print(f"提示：用 --open N 打开第 N 条结果的源文件（N 从 1 开始，例如 --open 1）")

    if args.open_index is not None:
        return _open_hit(args, hits)
    return 0


def _open_hit(args: argparse.Namespace, hits: list, *,
              out=None) -> int:
    """`search --open N`：用编辑器打开第 N 条结果的源文件并跳到行号。

    返回 0 表示「结果存在且编辑器拉起来了」，1 表示 N 越界 / 找不到源文件 /
    没有可用编辑器。这些都不是异常情况，所以只打印说明、不抛栈。
    """
    from .web.editor import open_in_editor, resolve_source_path

    stream = out if out is not None else sys.stdout
    try:
        n = int(args.open_index)
    except (TypeError, ValueError):
        print(f"[X] --open 需要一个从 1 开始的整数，收到：{args.open_index!r}", file=stream)
        return 1

    if n < 1 or n > len(hits):
        print(f"[X] --open {n} 超出范围：本次只有 {len(hits)} 条结果（N 从 1 开始）",
              file=stream)
        return 1

    hit = hits[n - 1]
    raw_dir = _resolve_raw_dir(args)
    path = resolve_source_path(hit, raw_dir)
    if path is None:
        print(f"[X] 找不到第 {n} 条结果对应的源文件：{hit.location()}", file=stream)
        print(f"    拼接规则是 <raw_dir>/<repo>/<rel_path>，当前 raw_dir = {raw_dir}",
              file=stream)
        if not raw_dir.exists():
            print("    raw_dir 不存在：用 --raw-dir 指向语料仓库的上一级目录"
                  "（含 JavaGuide、CS-Notes 等子目录的那一层）", file=stream)
        return 1

    ok, command = open_in_editor(path, hit.line_start or 1)
    if ok:
        print(f"[open] {command}", file=stream)
        print(f"       文件：{path}", file=stream)
    else:
        print(f"[X] 打开失败：{command}", file=stream)
    return 0 if ok else 1


def _summarize(text: str, width: int = 160) -> str:
    """把正文压成单行摘要：去掉换行和多余空白，超长截断加省略号。"""
    flat = " ".join((text or "").split())
    if len(flat) <= width:
        return flat
    return flat[:width] + f"... [共 {len(text)} 字符]"


def _load_generate_config(args: argparse.Namespace):
    """读 config.toml 的 [generate] 段，再被命令行参数覆盖。

    和 `_load_retrieval_config` 同一套优先级：命令行 > 配置文件 > 代码默认。
    """
    cfg_path = Path(args.config) if getattr(args, "config", None) else DEFAULT_CONFIG
    gc = IngestConfig.load(cfg_path).generate
    if getattr(args, "model", None):
        gc.model = args.model
    if getattr(args, "temperature", None) is not None:
        gc.temperature = float(args.temperature)
    if getattr(args, "max_context_chars", None):
        gc.max_context_chars = int(args.max_context_chars)
    if getattr(args, "top_k", None):
        gc.top_k = int(args.top_k)
    return gc


def cmd_ask(args: argparse.Namespace) -> int:
    """检索 -> 大模型 -> 带引用的答案。

    输出里不打印 key，`DeepSeekClient` 的异常消息也经过脱敏；找不到 key 时
    只提示设置方法。
    """
    from .generate.llm import DeepSeekClient, KeyNotFoundError, LLMError
    from .generate.pipeline import Answerer
    from .retrieve.embedder import HashingEmbedder
    from .retrieve.pipeline import Retriever

    rc = _load_retrieval_config(args)
    gc = _load_generate_config(args)
    index_dir = rc.index_dir
    if not index_dir.exists():
        print(f"[X] 索引目录不存在：{index_dir}")
        print("    先建索引：python ingest.py index --chunks <chunks.jsonl>")
        return 2

    try:
        index_info = json.loads(
            (index_dir / "index_meta.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        print(f"[X] {index_dir / 'index_meta.json'} 读不出来，索引可能不完整，请重建。")
        return 2

    embedder = None
    if "hashing" in str(index_info.get("embedder", "")).lower():
        embedder = HashingEmbedder(dim=int(index_info.get("dim", 512)))

    # --no-rerank 优先：配置里开了 use_reranker 也能被它关掉
    want_rerank = (bool(args.use_rerank) or bool(rc.use_reranker)) \
        and not args.no_rerank

    try:
        retriever = Retriever(index_dir, embedder=embedder,
                              cfg=rc.to_retriever_cfg())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[X] {exc}")
        return 2

    try:
        llm = DeepSeekClient(
            model=gc.model,
            base_url=gc.base_url,
            api_key_env=gc.api_key_env,
            timeout=gc.timeout,
        )
    except KeyNotFoundError as exc:
        print(f"[X] {exc}")
        return 3
    except LLMError as exc:
        print(f"[X] 初始化模型客户端失败：{exc}")
        return 2

    try:
        answerer = Answerer(retriever, llm, cfg=gc.to_answerer_cfg())
        ans = answerer.answer(args.question, top_k=gc.top_k,
                              use_rerank=want_rerank)
    except LLMError as exc:
        # exc 的消息里不含请求头；这里也只打印消息本身
        print(f"[X] 调用模型失败：{exc}")
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        # embedder / reranker 是懒加载的，缺模型文件会在第一次检索时才炸
        print(f"[X] 检索失败：{exc}")
        if "模型目录不存在" in str(exc):
            print("    模型不在配置的路径下时，用 --model-dir 指向实际目录。")
        return 2
    finally:
        llm.close()

    if args.json:
        sys.stdout.write(
            json.dumps(ans.to_dict(), ensure_ascii=False, indent=2) + "\n"
        )
        return 0

    mode = "hybrid" + (" + rerank" if want_rerank else "")
    print(f"问题: {ans.question}")
    print(f"索引: {index_dir}")
    print(f"模型: {ans.model}   检索模式: {mode}   "
          f"上下文块数: {ans.context_chunks}/{len(ans.hits)}   "
          f"耗时: {ans.latency_ms} ms")
    if ans.usage:
        print(f"token: prompt={ans.usage.get('prompt_tokens', '-')} "
              f"completion={ans.usage.get('completion_tokens', '-')} "
              f"total={ans.usage.get('total_tokens', '-')}")
    print("=" * 68)
    print(ans.render())
    print("=" * 68)

    if args.show_hits:
        print(f"检索原始结果（{len(ans.hits)} 条）：")
        for hit in ans.hits:
            print(f"[{hit.rank:>2}] {hit.score:>8.4f}  {hit.source:<7} "
                  f"{hit.breadcrumb or '(无标题)'}")
            print(f"     {hit.location()}")
            print(f"     {_summarize(hit.text)}")
            print("-" * 68)
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    """启动本地 Web 界面（长驻服务）。

    索引和模型**只在这里加载一次**：`Retriever` 构造完就交给 `create_app`，
    之后每个 HTTP 请求复用同一个实例，单次检索只剩「编码 query + 矩阵乘 +
    BM25」。启动时先跑一次预热检索，把模型首帧和分词器初始化的开销提前
    付掉，第一个用户请求不用等 70 秒。

    没有 API key 也能启动：`/api/search` 正常，`/api/ask` 返回 503。
    """
    import threading
    import time
    import webbrowser

    try:
        import uvicorn
    except ImportError:
        print("[X] 没有 uvicorn，Web 界面起不来（本机 pip 装不了包，"
              "用已装好的那套环境跑）")
        return 2

    from .retrieve.embedder import HashingEmbedder
    from .retrieve.pipeline import Retriever
    from .web.app import create_app
    from .web.editor import detect_editor

    rc = _load_retrieval_config(args)
    web_cfg = _load_ingest_config(args).web

    index_dir = rc.index_dir
    if not index_dir.exists():
        print(f"[X] 索引目录不存在：{index_dir}")
        print("    先建索引：python ingest.py index --chunks <chunks.jsonl>")
        return 2
    try:
        index_info = json.loads(
            (index_dir / "index_meta.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        print(f"[X] {index_dir / 'index_meta.json'} 读不出来，索引可能不完整，请重建。")
        return 2

    host = (args.host or web_cfg.host or "127.0.0.1").strip()
    port = int(args.port if args.port is not None else web_cfg.port)
    raw_dir = _resolve_raw_dir(args)
    editor = (args.editor or web_cfg.editor or "").strip() or detect_editor()

    # 0.0.0.0 是监听地址，不是能点开的地址
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    url = f"http://{display_host}:{port}"

    embedder = None
    if "hashing" in str(index_info.get("embedder", "")).lower():
        embedder = HashingEmbedder(dim=int(index_info.get("dim", 512)))

    n_chunks = index_info.get("n_chunks")
    print("=" * 68)
    print("  Phase 5 · 本地 Web 界面")
    print("=" * 68)
    print(f"  地址     : {url}")
    print(f"  索引     : {index_dir}"
          + (f"  ({_fmt(int(n_chunks))} 块)" if n_chunks else ""))
    print(f"  模型     : {rc.model_dir}")
    print(f"  语料     : {raw_dir}"
          + ("" if raw_dir.exists() else "   [!] 目录不存在，/api/open 会打不开文件"))
    print(f"  编辑器   : {editor or '未找到（code / notepad 都不在 PATH 上）'}")
    print("-" * 68)
    print("  正在加载索引（一次性，之后每个请求复用）...")
    sys.stdout.flush()

    started = time.perf_counter()
    try:
        retriever = Retriever(index_dir, embedder=embedder,
                              cfg=rc.to_retriever_cfg())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[X] {exc}")
        return 2
    load_sec = time.perf_counter() - started
    print(f"  索引就绪 : {load_sec:.1f}s")

    warmed = time.perf_counter()
    try:
        retriever.search("预热", top_k=1)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[X] 模型预热失败：{exc}")
        if "模型目录不存在" in str(exc):
            print("    用 --model-dir 指向 bge-m3 目录（例如 G:\\kb\\models\\bge-m3）。")
        return 2
    warm_sec = time.perf_counter() - warmed
    print(f"  模型就绪 : {warm_sec:.1f}s（预热检索）")

    answerer = None
    llm = None
    answer_state = "未启用（没有可用的模型客户端）"
    try:
        from .generate.llm import DeepSeekClient, KeyNotFoundError, LLMError
        from .generate.pipeline import Answerer

        gc = _load_generate_config(args)
        llm = DeepSeekClient(
            model=gc.model,
            base_url=gc.base_url,
            api_key_env=gc.api_key_env,
            timeout=gc.timeout,
        )
        answerer = Answerer(retriever, llm, cfg=gc.to_answerer_cfg())
        answer_state = f"已启用（{gc.model}）"
    except KeyNotFoundError as exc:
        # 没有 key 只影响问答，检索照常
        answer_state = f"未启用（{exc}）"
    except LLMError as exc:
        answer_state = f"未启用（初始化失败：{exc}）"

    print(f"  问答     : {answer_state}")
    print("-" * 68)

    app = create_app(retriever, answerer, raw_dir=raw_dir, editor=editor)
    print(f"  打开 {url}    （Ctrl+C 停止服务）")
    print("=" * 68)
    # uvicorn 启动后 stdout 会被它接管（日志走自己的 handler），这里先冲干净：
    # 不加的话输出被管道/重定向时是块缓冲，启动信息会一直卡在缓冲区里。
    sys.stdout.flush()

    if not args.no_browser:
        # 等 uvicorn 把端口 bind 完再开浏览器，1 秒足够本地启动
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        uvicorn.run(app, host=host, port=port)
    except KeyboardInterrupt:
        print("\n[停止] 已退出")
    finally:
        if llm is not None:
            llm.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kb",
        description="本地知识库 —— Phase 1 语料入库 / Phase 2 检索",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="遍历语料目录，切块并输出 JSONL + 统计")
    p_ingest.add_argument("--config", help="config.toml 路径（默认项目根目录下）")
    p_ingest.add_argument("--collection", metavar="NAMES",
                          help="只入库指定集合，逗号分隔（如 tech,papers）；"
                               "不写则入所有集合")
    p_ingest.add_argument("--dry-run", action="store_true", dest="dry_run",
                          help="只预览不写文件：块数、平均字符数、前 2 块 + 质量警告")
    p_ingest.add_argument("--raw", help="原始语料目录（覆盖所选集合的 raw_dir）")
    p_ingest.add_argument("--out", help="输出目录")
    p_ingest.add_argument("--max-chars", type=int, dest="max_chars", help="单块字符上限")
    p_ingest.add_argument("--min-chars", type=int, dest="min_chars", help="合并阈值")
    p_ingest.add_argument("--overlap", type=int, help="二次切分的重叠字符数")
    p_ingest.set_defaults(func=cmd_ingest)

    p_index = sub.add_parser("index", help="建检索索引（稠密 + BM25）")
    p_index.add_argument("--config", help="config.toml 路径（默认项目根目录下）")
    p_index.add_argument("--collection", metavar="NAMES",
                         help="只建这些集合的索引，逗号分隔；"
                              "不写则用 chunks_all.jsonl 建单个索引（旧行为）")
    p_index.add_argument("--chunks", help="chunks_all.jsonl 路径")
    p_index.add_argument("--index-dir", dest="index_dir", help="索引输出目录")
    p_index.add_argument("--out", help="输出目录（chunks_<collection>.jsonl 所在目录）")
    p_index.add_argument("--model-dir", dest="model_dir", help="bge-m3 ONNX 目录")
    p_index.add_argument("--embedder", choices=["onnx", "hashing"], default=None,
                         help="onnx=bge-m3（需要模型）；hashing=零依赖自检用。"
                              "集合模式下留空则沿用该集合已有索引的 embedder")
    p_index.add_argument("--batch-size", type=int, dest="batch_size", help="嵌入批大小")
    p_index.add_argument("--max-length", type=int, dest="max_length",
                         help="嵌入截断长度（默认 512）")
    p_index.add_argument("--limit", type=int,
                         help="只处理前 N 条，用于快速自检")
    p_index.set_defaults(func=cmd_index)

    p_drop = sub.add_parser("drop", help="从一个集合里删文档并重建该集合的索引")
    p_drop.add_argument("--config", help="config.toml 路径（默认项目根目录下）")
    p_drop.add_argument("--collection", metavar="NAME", required=True,
                        help="要操作的集合（一次一个）")
    p_drop.add_argument("--path", help="文件路径：相对 raw_dir、<repo>/<rel_path> 或文件名")
    p_drop.add_argument("--doc-id", dest="doc_id", help="按 doc_id 删（从 manifest 里找）")
    p_drop.add_argument("--index-dir", dest="index_dir", help="（兼容占位，不生效）")
    p_drop.add_argument("--model-dir", dest="model_dir", help="bge-m3 ONNX 目录")
    p_drop.add_argument("--embedder", choices=["onnx", "hashing"], default=None,
                        help="重建索引用哪种 embedder；留空沿用该集合已有索引")
    p_drop.add_argument("--batch-size", type=int, dest="batch_size", help="嵌入批大小")
    p_drop.add_argument("--max-length", type=int, dest="max_length",
                        help="嵌入截断长度")
    p_drop.add_argument("--out", help="输出目录（chunks/manifest 所在目录）")
    p_drop.add_argument("--yes", action="store_true", help="跳过二次确认")
    p_drop.set_defaults(func=cmd_drop)

    p_search = sub.add_parser("search", help="查询索引")
    p_search.add_argument("query", help="查询词")
    p_search.add_argument("--config", help="config.toml 路径（默认项目根目录下）")
    p_search.add_argument("--collection", metavar="NAMES",
                          help="查哪些集合，逗号分隔（如 tech,profile）；"
                               "不写则用 [collections].default_search")
    p_search.add_argument("--include-low-quality", action="store_true",
                          dest="include_low_quality",
                          help="把标记为 low_quality 的块也纳入检索（默认排除）")
    p_search.add_argument("--index-dir", dest="index_dir",
                          help="单索引模式下的索引目录（与 --collection 互斥）")
    p_search.add_argument("--model-dir", dest="model_dir", help="bge-m3 ONNX 目录")
    p_search.add_argument("--reranker-dir", dest="reranker_dir", help="重排模型目录")
    p_search.add_argument("--top-k", type=int, default=10, dest="top_k",
                          help="返回条数（默认 10）")
    p_search.add_argument("--no-dense", action="store_true", help="关闭稠密检索")
    p_search.add_argument("--no-sparse", action="store_true", help="关闭 BM25 检索")
    p_search.add_argument("--rerank", action="store_true", dest="use_rerank",
                          help="开启交叉编码重排")
    p_search.add_argument("--json", action="store_true", help="输出 JSON")
    p_search.add_argument("--open", type=int, default=None, dest="open_index",
                          metavar="N",
                          help="打印结果后用编辑器打开第 N 条（从 1 开始）的源文件")
    p_search.add_argument("--raw-dir", dest="raw_dir",
                          help="语料根目录（默认取 [path].raw_dir），--open 用它拼路径")
    p_search.add_argument("--out", help="输出目录（chunks/manifest 所在目录）")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="检索 + 大模型生成带引用的答案（Phase 4）")
    p_ask.add_argument("question", help="问题")
    p_ask.add_argument("--config", help="config.toml 路径（默认项目根目录下）")
    p_ask.add_argument("--index-dir", dest="index_dir", help="索引目录")
    p_ask.add_argument("--model-dir", dest="model_dir", help="bge-m3 ONNX 目录")
    p_ask.add_argument("--reranker-dir", dest="reranker_dir", help="重排模型目录")
    p_ask.add_argument("--top-k", type=int, dest="top_k",
                       help="送进提示词的块数（默认取 [generate].top_k）")
    p_ask.add_argument("--model", help="模型名（默认取 [generate].model）")
    p_ask.add_argument("--temperature", type=float,
                       help="采样温度（默认取 [generate].temperature）")
    p_ask.add_argument("--max-context-chars", type=int, dest="max_context_chars",
                       help="上下文总字符上限（默认取 [generate].max_context_chars）")
    p_ask.add_argument("--rerank", action="store_true", dest="use_rerank",
                       help="强制开启重排")
    p_ask.add_argument("--no-rerank", action="store_true", dest="no_rerank",
                       help="关闭重排（优先于配置和 --rerank）")
    p_ask.add_argument("--json", action="store_true", help="输出 Answer.to_dict()")
    p_ask.add_argument("--show-hits", action="store_true", dest="show_hits",
                       help="额外打印检索原始结果")
    p_ask.set_defaults(func=cmd_ask)

    p_web = sub.add_parser("web", help="启动本地 Web 界面（Phase 5，长驻服务）")
    p_web.add_argument("--config", help="config.toml 路径（默认项目根目录下）")
    p_web.add_argument("--host", help="监听地址（默认取 [web].host，127.0.0.1）")
    p_web.add_argument("--port", type=int, help="端口（默认取 [web].port，8000）")
    p_web.add_argument("--index-dir", dest="index_dir", help="索引目录")
    p_web.add_argument("--model-dir", dest="model_dir", help="bge-m3 ONNX 目录")
    p_web.add_argument("--raw-dir", dest="raw_dir",
                       help="语料根目录（默认取 [path].raw_dir），/api/open 用它拼路径")
    p_web.add_argument("--editor",
                       help="编辑器命令名或可执行文件路径（默认取 [web].editor，留空自动探测）")
    p_web.add_argument("--no-browser", action="store_true", dest="no_browser",
                       help="启动后不自动打开浏览器")
    p_web.set_defaults(func=cmd_web)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 cp936，中文输出会炸，这里强制 UTF-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
