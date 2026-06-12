"""Compatibility entrypoint for the NiceGUI Mythos Content Engine UI.

The old Gradio implementation has been retired from this launcher. Keep this
file as the stable `python src/ui.py` entrypoint, while the actual NiceGUI app
lives in `nicegui_ui.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

from nicegui_ui import APP_TITLE, ui


if __name__ in {"__main__", "__mp_main__"}:
    Path("outputs").mkdir(exist_ok=True)
    ui.run(
        host="127.0.0.1",
        port=int(os.getenv("NICEGUI_SERVER_PORT", "7860")),
        title=APP_TITLE,
    )
