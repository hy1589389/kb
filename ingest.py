"""便捷入口：python ingest.py [ingest 子命令参数]

等价于  python -m kb.cli ingest ...，只是省掉了设置 PYTHONPATH 的麻烦。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kb.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
