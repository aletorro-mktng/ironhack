from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import error, request


LEGACY_OPENAPI_URL = os.getenv("LEGACY_GRADIO_OPENAPI_URL", "http://127.0.0.1:7860/openapi.json")
OUTPUT_DIR = Path("docs")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    try:
        with request.urlopen(LEGACY_OPENAPI_URL, timeout=5) as response:
            schema = json.loads(response.read())
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        (OUTPUT_DIR / "legacy_gradio_operations.md").write_text(
            "\n".join(
                [
                    "# Legacy Gradio Operations",
                    "",
                    f"Could not fetch `{LEGACY_OPENAPI_URL}`.",
                    "",
                    f"Reason: `{exc}`",
                    "",
                    "Start the legacy Gradio server, set `LEGACY_GRADIO_OPENAPI_URL` if it is not on port 7860, then rerun `npm run catalog:legacy`.",
                ]
            ),
            encoding="utf-8",
        )
        return

    (OUTPUT_DIR / "legacy_gradio_openapi.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    rows = ["# Legacy Gradio Operations", "", "| Method | Path | Operation |", "| --- | --- | --- |"]
    for path, methods in sorted(schema.get("paths", {}).items()):
        for method, operation in sorted(methods.items()):
            if method.startswith("x-"):
                continue
            rows.append(f"| {method.upper()} | `{path}` | {operation.get('operationId') or operation.get('summary') or ''} |")
    (OUTPUT_DIR / "legacy_gradio_operations.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
