from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, TypedDict

from .base import Tool


def render_file_diff(
    before: str,
    after: str,
    path: Path,
    *,
    max_lines: int = 120,
) -> str:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    diff = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
            lineterm="",
        )
    )
    if not diff:
        return "[no changes]"
    if len(diff) <= max_lines:
        return "\n".join(diff)
    shown = diff[:max_lines]
    return "\n".join([*shown, f"... +{len(diff) - max_lines} diff lines"])


class _EditReplacement(TypedDict):
    old_text: str
    new_text: str
    replace_all: bool


class EditTool(Tool):
    name = "edit"
    description = (
        "Edit an existing text file by replacing old_text with new_text. For multiple "
        "changes in one file, pass edits as an array of replacements so they are applied "
        "together. Use this tool for file edits; do not use bash, sed, perl, or python "
        "scripts to modify files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to replace. Must appear in the file.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of only the first.",
            },
            "edits": {
                "type": "array",
                "description": (
                    "Multiple replacements to apply sequentially to the same file in "
                    "one edit call. Each item accepts old_text, new_text, and optional "
                    "replace_all."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "old_text": {
                            "type": "string",
                            "description": "Exact text to replace. Must appear in the file.",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text.",
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": (
                                "Replace every occurrence for this edit instead of only "
                                "the first."
                            ),
                        },
                    },
                    "required": ["old_text", "new_text"],
                },
            },
        },
        "required": ["path"],
    }

    def run(
        self,
        path: str,
        old_text: str | None = None,
        new_text: str | None = None,
        replace_all: bool = False,
        edits: list[dict[str, Any]] | None = None,
    ) -> str:
        p = Path(path)
        before = p.read_text()
        replacements = _normalize_replacements(
            old_text=old_text,
            new_text=new_text,
            replace_all=replace_all,
            edits=edits,
        )
        after = before
        total_replacements = 0
        for index, replacement in enumerate(replacements, start=1):
            try:
                after, occurrences = _apply_replacement(after, replacement)
            except ValueError as exc:
                if len(replacements) == 1:
                    raise ValueError(f"old_text not found in {p}") from exc
                raise ValueError(f"old_text not found in {p} for edit {index}") from exc
            total_replacements += occurrences
        p.write_text(after)
        diff = render_file_diff(before, after, p)
        replacement_label = "replacement" if total_replacements == 1 else "replacements"
        if len(replacements) == 1:
            summary = f"Edited {p} ({total_replacements} {replacement_label})"
        else:
            edit_label = "edit" if len(replacements) == 1 else "edits"
            summary = (
                f"Edited {p} ({total_replacements} {replacement_label} "
                f"across {len(replacements)} {edit_label})"
            )
        return f"{summary}\n{diff}"


def _normalize_replacements(
    *,
    old_text: str | None,
    new_text: str | None,
    replace_all: bool,
    edits: list[dict[str, Any]] | None,
) -> list[_EditReplacement]:
    if edits is not None:
        if old_text is not None or new_text is not None:
            raise ValueError("Use either old_text/new_text or edits, not both")
        if not edits:
            raise ValueError("edits must contain at least one replacement")
        return [
            _replacement_from_mapping(edit, index=index)
            for index, edit in enumerate(edits, start=1)
        ]
    if old_text is None or new_text is None:
        raise ValueError("old_text and new_text are required when edits is not provided")
    return [_validate_replacement(old_text, new_text, replace_all=replace_all)]


def _replacement_from_mapping(mapping: dict[str, Any], *, index: int) -> _EditReplacement:
    old_text = mapping.get("old_text")
    new_text = mapping.get("new_text")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        raise ValueError(f"edit {index} requires string old_text and new_text")
    replace_all = mapping.get("replace_all", False)
    if not isinstance(replace_all, bool):
        raise ValueError(f"edit {index} replace_all must be a boolean")
    return _validate_replacement(old_text, new_text, replace_all=replace_all)


def _validate_replacement(
    old_text: str,
    new_text: str,
    *,
    replace_all: bool,
) -> _EditReplacement:
    if old_text == "":
        raise ValueError("old_text must not be empty")
    return {"old_text": old_text, "new_text": new_text, "replace_all": replace_all}


def _apply_replacement(
    text: str,
    replacement: _EditReplacement,
) -> tuple[str, int]:
    old_text = replacement["old_text"]
    new_text = replacement["new_text"]
    replace_all = replacement["replace_all"]
    if old_text in text:
        occurrences = text.count(old_text) if replace_all else 1
        count = -1 if replace_all else 1
        return text.replace(old_text, new_text, count), occurrences
    return _apply_whitespace_flexible_replacement(text, replacement)


def _apply_whitespace_flexible_replacement(
    text: str,
    replacement: _EditReplacement,
) -> tuple[str, int]:
    old_lines = replacement["old_text"].splitlines(keepends=True)
    if not old_lines or not any(_is_whitespace_only_line(line) for line in old_lines):
        raise ValueError("old_text not found")

    text_lines = text.splitlines(keepends=True)
    window = len(old_lines)
    match_starts: list[int] = []
    for start in range(0, len(text_lines) - window + 1):
        if _lines_match_with_flexible_blank_lines(
            text_lines[start : start + window],
            old_lines,
        ):
            match_starts.append(start)
            if not replacement["replace_all"]:
                break
    if not match_starts:
        raise ValueError("old_text not found")

    new_lines = replacement["new_text"].splitlines(keepends=True)
    for start in reversed(match_starts):
        text_lines[start : start + window] = new_lines
    return "".join(text_lines), len(match_starts)


def _lines_match_with_flexible_blank_lines(
    file_lines: list[str],
    old_lines: list[str],
) -> bool:
    return all(
        _is_whitespace_only_line(old_line) and _is_whitespace_only_line(file_line)
        or old_line == file_line
        for file_line, old_line in zip(file_lines, old_lines, strict=True)
    )


def _is_whitespace_only_line(line: str) -> bool:
    return line.strip("\r\n").strip(" \t") == ""
