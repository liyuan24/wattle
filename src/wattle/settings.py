"""Persistent user settings for Wattle."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
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
    max_tokens: int = 4096
    thinking: bool = False
    effort: Effort | None = None
    permission_mode: PermissionMode = PermissionMode.YOLO
    statusline: bool = True
    tui: TuiSettings = field(default_factory=TuiSettings)
    enabled_models: tuple[str, ...] = field(default_factory=tuple)
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
    settings = replace(load_settings(path), **changes)
    save_settings(settings, path)
    return settings


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
        "enabled_models": list(settings.enabled_models),
        "compaction_keep_recent_tokens": settings.compaction_keep_recent_tokens,
        "git_commit_attribution": settings.git_commit_attribution,
    }


def settings_from_dict(data: dict[str, Any]) -> WattleSettings:
    defaults = WattleSettings()
    tui = _tui_settings(data.get("tui"), data)
    return WattleSettings(
        provider=_str(data.get("provider"), defaults.provider),
        model=_str(data.get("model"), defaults.model),
        max_tokens=_int(data.get("max_tokens"), defaults.max_tokens),
        thinking=_bool(data.get("thinking"), defaults.thinking),
        effort=_effort(data.get("effort")),
        permission_mode=_permission_mode(
            data.get("permission_mode"),
            defaults.permission_mode,
        ),
        statusline=bool(tui.statusline),
        tui=tui,
        enabled_models=_str_tuple(data.get("enabled_models")),
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
        return default


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
