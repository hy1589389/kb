"""测 DirectML 能否让 bge-m3 用上 4GB 显卡。

把 onnxruntime-directml 的目录放在 sys.path 最前面，遮蔽掉已装的 CPU 版 onnxruntime。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WORK = Path(r"G:\DeepSeek_Harness\.kb-phase2\work")
sys.path.insert(0, str(WORK / ".pylibs"))
sys.path.insert(0, r"G:\DeepSeek_Harness\.kb-phase2\.pylibs-dml")   # 必须最前

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

print("=== 当前 onnxruntime ===")
print(f"  版本: {ort.__version__}")
print(f"  路径: {Path(ort.__file__).parent}")
providers = ort.get_available_providers()
print(f"  可用后端: {providers}")

if "DmlExecutionProvider" not in providers:
    print()
    print("  DirectML 后端不可用，GPU 这条路走不通。")
    raise SystemExit(0)

MDIR = Path(r"G:\kb\models\bge-m3")
CORPUS = Path(r"G:\kb\data\processed\chunks_all.jsonl")

sample = []
with CORPUS.open(encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i % 137 == 0:
            sample.append(json.loads(line)["embed_text"])
        if len(sample) >= 40:
            break
lens = [len(s) for s in sample]
print()
print(f"=== 真实块样本 {len(sample)} 条，字符 p50={int(np.percentile(lens, 50))} max={max(lens)} ===")

tok = Tokenizer.from_file(str(MDIR / "tokenizer.json"))
tok.enable_truncation(max_length=512)
tok.enable_padding()

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

print()
print("=== 加载到 GPU ===")
t0 = time.perf_counter()
try:
    sess = ort.InferenceSession(
        str(MDIR / "model.onnx"), so,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
except Exception as e:  # noqa: BLE001
    print(f"  加载失败: {type(e).__name__}: {e}")
    raise SystemExit(1)
print(f"  加载 {time.perf_counter() - t0:.1f}s")
print(f"  实际使用: {sess.get_providers()}")

in_names = {i.name for i in sess.get_inputs()}
out_idx = [o.name for o in sess.get_outputs()].index("sentence_embedding")


def run(texts: list[str], batch: int) -> float:
    t0 = time.perf_counter()
    for s in range(0, len(texts), batch):
        encs = tok.encode_batch(texts[s:s + batch])
        feed = {
            "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
        }
        if "token_type_ids" in in_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
        sess.run(None, {k: v for k, v in feed.items() if k in in_names})
    return time.perf_counter() - t0


print()
print("=== 速度 ===")
for batch in (1, 4, 8, 16):
    try:
        dt = run(sample[:16], batch)
        spd = 16 / dt
        print(f"  batch={batch:3}  {spd:6.1f} 条/秒   全量 18,276 块 = {18276 / spd / 60:5.1f} 分钟")
    except Exception as e:  # noqa: BLE001
        print(f"  batch={batch:3}  失败: {type(e).__name__}: {str(e)[:100]}")

print()
print("=== 正确性：GPU 结果与 CPU 参考是否一致 ===")
encs = tok.encode_batch(sample[:8])
feed = {"input_ids": np.array([e.ids for e in encs], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64)}
gpu_vec = sess.run(None, feed)[out_idx]
print(f"  shape={gpu_vec.shape}  模长={np.linalg.norm(gpu_vec, axis=1)[:3]}")
