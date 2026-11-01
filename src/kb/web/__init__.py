"""Phase 5：本地 Web 界面 + 编辑器回链。

两个模块，职责分开：

    editor  检索结果 -> 源文件绝对路径 -> 编辑器（code / notepad）
    app     FastAPI 应用工厂，路由只做参数校验和格式转换

`__init__` 故意不 re-export 任何东西：`editor` 只依赖标准库，Web 层和 CLI
都可能单独用它，不该为了导入一个函数把 fastapi 一起拉起来。
"""

from __future__ import annotations

__all__: list[str] = []
