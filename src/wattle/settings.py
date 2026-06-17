"""Persistent user settings for Wattle."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from wattle.permissions import PermissionMode, parse_permission_mode

SETTINGS_PATH_ENV = "WATTLE_SETTINGS_PATH"
DEFAULT_TUI_STATUSLINE_FIELDS = ("model", "thinking", "cwd")

type Effort = Literal["low", "medium", "high", "xhigh", "max"]


def _default_statusline_fields() -> tuple[str, ...]:
    return DEFAULT_TUI_STATUSLINE_FIELDS


@dataclass(frozen=True, slots=True)
class TuiSettings:
    statusline: tuple[str, ...] = field(default_factory=_default_statusline_fields)
    show_thinking: bool = False

    @property
    def statusline_fields(self) -> tuple[str, ...]:
        return self.statusline


@dataclass(frozen=True, slots=True)
class WattleSettings:
    provider: str | None = None
    model: str | None = None
    max_tokens: int | None = None
    thinking: bool = False
    effort: Effort | None = None
    permission_mode: PermissionMode = PermissionMode.YOLO
    statusline: bool = True
    tui: TuiSettings = field(default_factory=TuiSettings)
    compaction_keep_recent_tokens: int = 20_000
    git_commit_attribution: bool = True


def default_settings_path() -> Path:
    configured = os.environ.get(SETTINGS_PATH_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".wattle" / "settings.json"


def load_settings(path: str | Path | None = None) -> WattleSettings:
    target = Path(path) if path is not None else default_settings_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return WattleSettings()
    if not isinstance(data, dict):
        return WattleSettings()
    return settings_from_dict(data)


def save_settings(settings: WattleSettings, path: str | Path | None = None) -> Path:
    target = Path(path) if path is not None else default_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(settings_to_dict(settings), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def update_settings(path: str | Path | None = None, **changes: Any) -> WattleSettings:
    old_settings = load_settings(path)
    settings = replace(old_settings, **changes)
    target = save_settings(settings, path)
    _audit_settings_update(target, old_settings, settings, changes)
    return settings


def _audit_settings_update(
    target: Path,
    old_settings: WattleSettings,
    new_settings: WattleSettings,
    changes: dict[str, Any],
) -> None:
    """Best-effort audit trail for settings writes."""
    try:
        resolved_target = target.expanduser().resolve(strict=False)
        record = {
            "event": "settings_update",
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "cwd": str(Path.cwd()),
            "settings_path": str(resolved_target),
            "settings_path_input": str(target),
            "env_settings_path": os.environ.get(SETTINGS_PATH_ENV),
            "changed_keys": sorted(changes),
            "provider": {"old": old_settings.provider, "new": new_settings.provider},
            "model": {"old": old_settings.model, "new": new_settings.model},
        }
        audit_path = resolved_target.with_name(f"{resolved_target.stem}.audit.jsonl")
        with audit_path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        return


def settings_to_dict(settings: WattleSettings) -> dict[str, Any]:
    return {
        "provider": settings.provider,
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "thinking": settings.thinking,
        "effort": settings.effort,
        "permission_mode": settings.permission_mode.value,
        "tui": {
            "statusline": list(settings.tui.statusline),
            "show_thinking": settings.tui.show_thinking,
        },
        "compaction_keep_recent_tokens": settings.compaction_keep_recent_tokens,
        "git_commit_attribution": settings.git_commit_attribution,
    }


def settings_from_dict(data: dict[str, Any]) -> WattleSettings:
    defaults = WattleSettings()
    tui = _tui_settings(data.get("tui"), data)
    return WattleSettings(
        provider=_str(data.get("provider"), defaults.provider),
        model=_str(data.get("model"), defaults.model),
        max_tokens=_optional_int(data.get("max_tokens")),
        thinking=_bool(data.get("thinking"), defaults.thinking),
        effort=_effort(data.get("effort")),
        permission_mode=_permission_mode(
            data.get("permission_mode"),
            defaults.permission_mode,
        ),
        statusline=bool(tui.statusline),
        tui=tui,
        compaction_keep_recent_tokens=max(
            1,
            _int(
                data.get("compaction_keep_recent_tokens"),
                defaults.compaction_keep_recent_tokens,
            ),
        ),
        git_commit_attribution=_bool(
            data.get("git_commit_attribution"),
            defaults.git_commit_attribution,
        ),
    )


def _str(value: object, default: str | None) -> str | None:
    return value if isinstance(value, str) and value else default


def _int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _bool(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _effort(value: object) -> Effort | None:
    if value in {"low", "medium", "high", "xhigh", "max"}:
        return cast(Effort, value)
    return None


def _permission_mode(value: object, default: PermissionMode) -> PermissionMode:
    if not isinstance(value, str):
        return default
    try:
        return parse_permission_mode(value)
    except ValueError:
        return PermissionMode.YOLO


def _tui_settings(value: object, root: dict[str, Any]) -> TuiSettings:
    defaults = TuiSettings()
    show_thinking = defaults.show_thinking
    if isinstance(value, dict):
        show_thinking = _bool(value.get("show_thinking"), defaults.show_thinking)
        if "statusline" in value:
            statusline = value.get("statusline")
            if isinstance(statusline, dict):
                fields = statusline.get("fields")
                if fields is None:
                    fields = statusline.get("statusline_fields")
                return TuiSettings(
                    statusline=_str_tuple(fields),
                    show_thinking=show_thinking,
                )
            return TuiSettings(
                statusline=_str_tuple(statusline),
                show_thinking=show_thinking,
            )
        if "statusline_fields" in value:
            return TuiSettings(
                statusline=_str_tuple(value.get("statusline_fields")),
                show_thinking=show_thinking,
            )
        return TuiSettings(show_thinking=show_thinking)
    if "statusline_fields" in root:
        return TuiSettings(statusline=_str_tuple(root.get("statusline_fields")))
    legacy_statusline = root.get("statusline")
    if legacy_statusline is False:
        return TuiSettings(statusline=())
    return TuiSettings()


def _str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)
