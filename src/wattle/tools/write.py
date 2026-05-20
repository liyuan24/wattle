from pathlib import Path

from .base import Tool
from .edit import render_file_diff


class WriteTool(Tool):
    name = "write"
    description = (
        "Create or overwrite a text file with exact content. Use this tool for new files "
        "or full-file rewrites; do not use bash shell redirection to write files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "content": {
                "type": "string",
                "description": "Full file contents to write.",
            },
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str) -> str:
        p = Path(path)
        before = p.read_text() if p.exists() else ""
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        diff = render_file_diff(before, content, p)
        return f"Wrote {len(content)} bytes to {p}\n{diff}"
