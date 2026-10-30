"""三个候选嵌入模型在真实语料上的基准：速度 + 语义区分度 + 全量耗时外推。

用 onnxruntime 直接推理，不走项目封装，以免封装里的差异影响可比性。
预处理与项目保持一致：tokenizer -> input_ids/attention_mask -> mean pooling -> L2 归一化。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WORK = Path(r"G:\DeepSeek_Harness\.kb-phase2\work")
sys.path.insert(0, str(WORK / ".pylibs"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

CORPUS = Path(r"G:\kb\data\processed\chunks_all.jsonl")
MODELS = Path(r"G:\DeepSeek_Harness\.kb-phase2\models")
SAMPLE_N = 120
MAX_LEN = 512

CANDIDATES = [
    ("bge-m3 (fp32, 568M)", Path(r"G:\kb\models\bge-m3") / "model.onnx", "model.onnx"),
    ("bge-base-zh int8 (102M)", MODELS / "bge-base-zh-int8", "model_int8.onnx"),
    ("bge-small-zh int8 (24M)", MODELS / "bge-small-zh-int8", "model_int8.onnx"),
    ("bge-small-zh fp32 (24M)", MODELS / "bge-small-zh-int8", "model.onnx"),
]

# ---- 取真实语料样本 ----
print("=== 抽样真实语料 ===")
sample: list[str] = []
with CORPUS.open(encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i % 137 == 0:  # 均匀抽样，覆盖各仓库
            c = json.loads(line)
            sample.append(c["embed_text"])
        if len(sample) >= SAMPLE_N:
            break
lens = [len(s) for s in sample]
print(f"  {len(sample)} 条，字符数 p50={int(np.percentile(lens, 50))} p90={int(np.percentile(lens, 90))} max={max(lens)}")
print()

# ---- 语义区分度用的固定题目 ----
DOMAINS = {
    "hashmap": ["JDK 8 之后 HashMap 采用数组 + 链表 + 红黑树，链表长度超过 8 且容量大于等于 64 时转为红黑树。",
                "HashMap 扩容时容量翻倍，元素的索引要么保持不变，要么移动到原索引加旧容量的位置。"],
    "tcp": ["TCP 三次握手用于确认通信双方的发送能力和接收能力，两次握手无法确认服务端的发送能力。",
            "TIME_WAIT 状态持续 2MSL，用于保证最后一个 ACK 能到达，并让旧连接的重复报文消失。"],
    "lora": ["LoRA 通过低秩分解冻结原权重，只训练旁路的两个小矩阵，显著降低微调显存。",
             "用 peft 配置 LoRA 时需要指定 target_modules、秩 r、缩放系数 alpha 和 dropout。"],
}
QUERIES = {"hashmap": "HashMap 什么时候把链表转成红黑树",
           "tcp": "为什么 TCP 握手需要三次而不是两次",
           "lora": "LoRA 微调要设置哪些超参数"}


def load(model_dir: Path, fname: str):
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_LEN)
    tok.enable_padding()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_dir / fname), so, providers=["CPUExecutionProvider"])
    return tok, sess


def encode(tok, sess, texts: list[str], batch: int) -> tuple[np.ndarray, float]:
    in_names = {i.name for i in sess.get_inputs()}
    out_name = sess.get_outputs()[0].name
    vecs: list[np.ndarray] = []
    t0 = time.perf_counter()
    for s in range(0, len(texts), batch):
        chunk = texts[s:s + batch]
        encs = tok.encode_batch(chunk)
        feed = {
            "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
        }
        if "token_type_ids" in in_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
        feed = {k: v for k, v in feed.items() if k in in_names}
        out = sess.run([out_name], feed)[0]
        mask = feed["attention_mask"][..., None].astype(np.float32)
        pooled = (out * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        vecs.append(pooled.astype(np.float32))
    return np.vstack(vecs), time.perf_counter() - t0


print(f"{'模型':28} {'dim':>5} {'加载':>6} {'速度':>10} {'全量18k':>10} {'区分度':>7} {'索引':>8}")
print("-" * 84)

results = {}
for label, mdir, fname in CANDIDATES:
    if not (mdir / fname).exists():
        print(f"{label:28} 文件不存在，跳过")
        continue
    t0 = time.perf_counter()
    try:
        tok, sess = load(mdir, fname)
    except Exception as e:  # noqa: BLE001
        print(f"{label:28} 加载失败: {type(e).__name__}: {e}")
        continue
    load_s = time.perf_counter() - t0

    try:
        vecs, dt = encode(tok, sess, sample, 32)
    except Exception as e:  # noqa: BLE001
        print(f"{label:28} 推理失败: {type(e).__name__}: {e}")
        continue

    # 区分度
    names = list(DOMAINS)
    passages = [p for n in names for p in DOMAINS[n]]
    pv, _ = encode(tok, sess, passages, 8)
    qv, _ = encode(tok, sess, [QUERIES[n] for n in names], 3)
    exp = [n for n in names for _ in DOMAINS[n]]
    hit = 0
    for qi, qn in enumerate(names):
        sims = pv @ qv[qi]
        top2 = set(np.argsort(-sims)[:2].tolist())
        want = {i for i, n in enumerate(exp) if n == qn}
        hit += top2 == want

    spd = len(sample) / dt
    eta = 18276 / spd / 60
    idx_mb = 18276 * vecs.shape[1] * 4 / 1024**2
    print(f"{label:28} {vecs.shape[1]:>5} {load_s:>5.1f}s {spd:>7.1f}/s {eta:>8.1f}分 {hit}/{len(names):>4} {idx_mb:>6.0f}MB")
    results[label] = dict(dim=int(vecs.shape[1]), spd=spd, eta=eta, hit=hit, idx_mb=idx_mb)

print()
print("=== 结论 ===")
ok = [k for k, v in results.items() if v["hit"] == 3]
print(f"  语义区分度满分(3/3)的模型: {ok if ok else '无'}")
fast = sorted(results.items(), key=lambda kv: -kv[1]["spd"])
if fast:
    print(f"  最快: {fast[0][0]}  {fast[0][1]['spd']:.1f} 条/秒  全量 {fast[0][1]['eta']:.1f} 分钟")
