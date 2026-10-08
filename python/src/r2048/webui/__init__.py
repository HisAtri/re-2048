"""WebUI：Gradio 交互界面（可选组件，见 ADR 0039）。

跑谱、训练、评测、推理都不依赖这里——所以 gradio 放在可选依赖组 `webui` 里，不进核心依赖。
装它：`uv sync --group webui`；跑它：`make webui`（或 `python -m r2048 webui`）。

三个标签页都只跟**会话进程**说话（`r2048.session_io`），界面本身不做棋规、不选招。
"""

from __future__ import annotations

__all__ = ["inventory"]
