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


class _EditMatch(TypedDict):
    replacement: _EditReplacement
    index: int
    start: int
    end: int


class EditTool(Tool):
    name = "edit"
    description = (
        "Edit an existing text file by replacing exact old_text with new_text. Each "
        "old_text must match a unique, non-overlapping region of the original file. "
        "For multiple disjoint changes in one file, pass edits as an array of "
        "replacements so they are applied together. Merge nearby or overlapping "
        "changes into one replacement. Use this tool for file edits; do not use bash, "
        "sed, perl, or python scripts to modify files."
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
            "edits": {
                "type": "array",
                "description": (
                    "Multiple replacements to apply together to the same file in one "
                    "edit call. Each item accepts old_text and new_text. Each "
                    "old_text is matched against the original file and must be unique."
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
        edits: list[dict[str, Any]] | None = None,
    ) -> str:
        p = Path(path)
        before_raw = _read_text_preserving_newlines(p)
        bom, before_without_bom = _strip_utf8_bom(before_raw)
        line_ending = _detect_line_ending(before_without_bom)
        before = _normalize_line_endings(before_without_bom)
        replacements = _normalize_replacements(
            old_text=old_text,
            new_text=new_text,
            edits=edits,
        )
        matches = _match_replacements(before, replacements, p)
        after = _apply_matches(before, matches)
        if after == before:
            raise ValueError(f"No changes made to {p}")
        _write_text_preserving_newlines(p, bom + _restore_line_endings(after, line_ending))
        diff = render_file_diff(before, after, p)
        total_replacements = len(matches)
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
    return [_validate_replacement(old_text, new_text)]


def _replacement_from_mapping(mapping: dict[str, Any], *, index: int) -> _EditReplacement:
    old_text = mapping.get("old_text")
    new_text = mapping.get("new_text")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        raise ValueError(f"edit {index} requires string old_text and new_text")
    if "replace_all" in mapping:
        raise ValueError(f"edit {index} replace_all is no longer supported")
    return _validate_replacement(old_text, new_text)


def _validate_replacement(old_text: str, new_text: str) -> _EditReplacement:
    if old_text == "":
        raise ValueError("old_text must not be empty")
    return {
        "old_text": _normalize_line_endings(old_text),
        "new_text": _normalize_line_endings(new_text),
    }


def _match_replacements(
    text: str,
    replacements: list[_EditReplacement],
    path: Path,
) -> list[_EditMatch]:
    matches: list[_EditMatch] = []
    for index, replacement in enumerate(replacements, start=1):
        found = _find_replacement_matches(text, replacement)
        if not found:
            if len(replacements) == 1:
                raise ValueError(f"old_text not found in {path}")
            raise ValueError(f"old_text not found in {path} for edit {index}")
        if len(found) > 1:
            if len(replacements) == 1:
                raise ValueError(
                    f"old_text matched {len(found)} occurrences in {path}; "
                    "include more context so it is unique"
                )
            raise ValueError(
                f"old_text matched {len(found)} occurrences in {path} for edit {index}; "
                "include more context so it is unique"
            )
        match = found[0]
        matches.append(
            {
                "replacement": replacement,
                "index": index,
                "start": match["start"],
                "end": match["end"],
            }
        )

    matches.sort(key=lambda match: match["start"])
    for previous, current in zip(matches, matches[1:], strict=False):
        if previous["end"] > current["start"]:
            raise ValueError(
                f"edits {previous['index']} and {current['index']} overlap in {path}; "
                "merge them into one replacement"
            )
    return matches


def _find_replacement_matches(
    text: str,
    replacement: _EditReplacement,
) -> list[dict[str, int]]:
    old_text = replacement["old_text"]
    exact_matches = _find_exact_matches(text, old_text)
    if exact_matches:
        return exact_matches
    return _find_whitespace_flexible_matches(text, replacement)


def _find_exact_matches(text: str, old_text: str) -> list[dict[str, int]]:
    matches: list[dict[str, int]] = []
    start = 0
    while True:
        index = text.find(old_text, start)
        if index == -1:
            return matches
        end = index + len(old_text)
        matches.append({"start": index, "end": end})
        start = end


def _find_whitespace_flexible_matches(
    text: str,
    replacement: _EditReplacement,
) -> list[dict[str, int]]:
    old_lines = replacement["old_text"].splitlines(keepends=True)
    if not old_lines or not any(_is_whitespace_only_line(line) for line in old_lines):
        return []

    text_lines = text.splitlines(keepends=True)
    offsets = _line_offsets(text_lines)
    window = len(old_lines)
    matches: list[dict[str, int]] = []
    for start in range(0, len(text_lines) - window + 1):
        if _lines_match_with_flexible_blank_lines(
            text_lines[start : start + window],
            old_lines,
        ):
            matches.append({"start": offsets[start], "end": offsets[start + window]})
    return matches


def _apply_matches(text: str, matches: list[_EditMatch]) -> str:
    after = text
    for match in sorted(matches, key=lambda item: item["start"], reverse=True):
        replacement = match["replacement"]
        after = after[: match["start"]] + replacement["new_text"] + after[match["end"] :]
    return after


def _line_offsets(lines: list[str]) -> list[int]:
    offsets = [0]
    total = 0
    for line in lines:
        total += len(line)
        offsets.append(total)
    return offsets


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


def _strip_utf8_bom(text: str) -> tuple[str, str]:
    if text.startswith("\ufeff"):
        return "\ufeff", text[1:]
    return "", text


def _detect_line_ending(text: str) -> str:
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    cr = text.count("\r") - crlf
    if crlf >= lf and crlf >= cr and crlf > 0:
        return "\r\n"
    if cr > lf and cr > 0:
        return "\r"
    return "\n"


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _restore_line_endings(text: str, line_ending: str) -> str:
    if line_ending == "\n":
        return text
    return text.replace("\n", line_ending)


def _read_text_preserving_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_text_preserving_newlines(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
