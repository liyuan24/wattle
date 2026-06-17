from __future__ import annotations

import json
import sys
from pathlib import Path

from wattle import settings


def test_update_settings_writes_audit_record(tmp_path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    monkeypatch.setattr(sys, "argv", ["wattle", "--test-flag"])
    settings_path = Path("..") / "settings.json"

    settings.save_settings(
        settings.WattleSettings(provider="openai_codex", model="gpt-5.5"),
        settings_path,
    )

    updated = settings.update_settings(
        settings_path,
        provider="xiaomi-token-plan-sgp",
        model="mimo-v2.5-pro",
    )

    assert updated.provider == "xiaomi-token-plan-sgp"
    assert updated.model == "mimo-v2.5-pro"

    resolved_settings_path = settings_path.resolve(strict=False)
    audit_path = resolved_settings_path.with_name("settings.audit.jsonl")
    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 1
    record = records[0]
    assert record["event"] == "settings_update"
    assert record["argv"] == ["wattle", "--test-flag"]
    assert record["cwd"] == str(Path.cwd())
    assert record["settings_path"] == str(resolved_settings_path)
    assert record["settings_path_input"] == str(settings_path)
    assert record["changed_keys"] == ["model", "provider"]
    assert record["provider"] == {
        "old": "openai_codex",
        "new": "xiaomi-token-plan-sgp",
    }
    assert record["model"] == {"old": "gpt-5.5", "new": "mimo-v2.5-pro"}
    assert isinstance(record["pid"], int)
    assert record["created_at"].endswith("Z")
