"""Tests for wattle.auth.

These tests redirect ``wattle.auth.AUTH_PATH`` at a per-test ``tmp_path`` so
they never read or write the real ``~/.wattle/auth.json``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib import parse

import pytest

from wattle import auth


@pytest.fixture
def auth_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point wattle.auth.AUTH_PATH at tmp_path/auth.json and return the path."""
    path = tmp_path / "auth.json"
    monkeypatch.setattr(auth, "AUTH_PATH", path)
    return path


def _jwt_with_payload(payload: dict[str, object]) -> str:
    def encode(payload: dict[str, object]) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    return f"{encode({'alg': 'none'})}.{encode(payload)}.signature"


def _jwt_with_exp(exp: int) -> str:
    return _jwt_with_payload({"exp": exp})


# ---------------------------------------------------------------------------
# AUTH_PATH constant
# ---------------------------------------------------------------------------


def test_auth_path_default_is_home_dotwattle_auth_json() -> None:
    """The unpatched constants must resolve to documented auth paths."""
    # Re-import to be safe; the module attribute should equal the documented path.
    assert Path.home() / ".wattle" / "auth.json" == auth.AUTH_PATH


def test_auth_path_is_monkeypatchable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contract requires AUTH_PATH be overridable via monkeypatch.setattr."""
    new = tmp_path / "elsewhere.json"
    monkeypatch.setattr(auth, "AUTH_PATH", new)
    assert new == auth.AUTH_PATH


# ---------------------------------------------------------------------------
# load_auth + get_api_key happy path
# ---------------------------------------------------------------------------


def test_get_api_key_returns_anthropic_and_openai(auth_file: Path) -> None:
    auth_file.write_text(
        json.dumps(
            {
                "anthropic": {"api_key": {"api_key": "sk-ant-abc123"}},
                "openai": {"api_key": {"api_key": "sk-openai-xyz789"}},
            }
        )
    )
    assert auth.get_api_key("anthropic") == "sk-ant-abc123"
    assert auth.get_api_key("openai") == "sk-openai-xyz789"


def test_get_credential_returns_api_key_metadata(auth_file: Path) -> None:
    auth_file.write_text(
        json.dumps({"openai": {"api_key": {"api_key": "sk-openai-xyz789"}}})
    )

    credential = auth.get_credential("openai")

    assert credential.kind == "api_key"
    assert credential.bearer_token == "sk-openai-xyz789"
    assert credential.source == f"{auth_file} openai.api_key"
    assert credential.expires_at is None


def test_get_credential_returns_openai_oauth_from_wattle(auth_file: Path) -> None:
    token = _jwt_with_exp(2_000_000_000)
    auth_file.write_text(
        json.dumps(
            {
                "openai": {
                    "oauth": {
                        "access_token": token,
                    }
                }
            }
        )
    )

    credential = auth.get_credential("openai")

    assert credential.kind == "oauth"
    assert credential.bearer_token == token
    assert credential.source == f"{auth_file} openai.oauth"
    assert credential.expires_at == 2_000_000_000


def test_get_credential_uses_oauth_by_default_when_both_methods_exist(
    auth_file: Path,
) -> None:
    token = _jwt_with_exp(2_000_000_000)
    auth_file.write_text(
        json.dumps(
            {
                "example": {
                    "api_key": {"api_key": "sk-example-existing"},
                    "oauth": {"access_token": token},
                }
            }
        )
    )

    credential = auth.get_credential("example")

    assert credential.kind == "oauth"
    assert credential.bearer_token == token
    assert credential.source == f"{auth_file} example.oauth"


def test_get_api_key_credential_uses_api_key_when_oauth_also_exists(
    auth_file: Path,
) -> None:
    token = _jwt_with_exp(2_000_000_000)
    auth_file.write_text(
        json.dumps(
            {
                "openai": {
                    "api_key": {"api_key": "sk-openai-existing"},
                    "oauth": {"access_token": token},
                }
            }
        )
    )

    credential = auth.get_api_key_credential("openai")

    assert credential.kind == "api_key"
    assert credential.bearer_token == "sk-openai-existing"
    assert credential.source == f"{auth_file} openai.api_key"


def test_get_api_key_credential_rejects_oauth_only_vendor(auth_file: Path) -> None:
    token = _jwt_with_exp(2_000_000_000)
    auth_file.write_text(json.dumps({"openai": {"oauth": {"access_token": token}}}))

    with pytest.raises(KeyError, match="api_key.*required"):
        auth.get_api_key_credential("openai")


def test_get_credential_accepts_legacy_explicit_oauth_over_api_key(
    auth_file: Path,
) -> None:
    token = _jwt_with_exp(2_000_000_000)
    auth_file.write_text(
        json.dumps(
            {
                "openai": {
                    "auth_type": "oauth",
                    "api_key": "sk-openai-existing",
                    "access_token": token,
                }
            }
        )
    )

    credential = auth.get_credential("openai")

    assert credential.kind == "oauth"
    assert credential.bearer_token == token


def test_get_api_key_returns_oauth_access_token_for_compatibility(auth_file: Path) -> None:
    token = _jwt_with_exp(2_000_000_000)
    auth_file.write_text(
        json.dumps({"openai": {"oauth": {"access_token": token}}})
    )

    assert auth.get_api_key("openai") == token


def test_get_openai_codex_credential_requires_wattle_scoped_oauth(
    auth_file: Path,
) -> None:
    token = _jwt_with_payload(
        {
            "exp": 2_000_000_000,
            "scp": [
                "openid",
                "profile",
                "email",
                "offline_access",
                "api.connectors.read",
                "api.connectors.invoke",
            ],
        }
    )
    auth_file.write_text(json.dumps({"openai": {"oauth": {"access_token": token}}}))

    credential = auth.get_openai_codex_credential()

    assert credential.kind == "oauth"
    assert credential.bearer_token == token
    assert credential.source == f"{auth_file} openai.oauth"


def test_get_openai_codex_credential_rejects_under_scoped_wattle_oauth(
    auth_file: Path,
) -> None:
    token = _jwt_with_payload(
        {
            "exp": 2_000_000_000,
            "scp": ["openid", "profile", "email", "offline_access"],
        }
    )
    auth_file.write_text(json.dumps({"openai": {"oauth": {"access_token": token}}}))

    with pytest.raises(ValueError, match="missing required Codex scopes"):
        auth.get_openai_codex_credential()


def test_get_credential_rejects_expired_oauth_token(auth_file: Path) -> None:
    auth_file.write_text(
        json.dumps(
            {
                "openai": {
                    "oauth": {
                        "access_token": _jwt_with_exp(1),
                    }
                }
            }
        )
    )

    with pytest.raises(KeyError, match="refresh_token"):
        auth.get_credential("openai")


def test_get_credential_refreshes_expired_oauth_for_any_provider(
    auth_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_token = _jwt_with_exp(1)
    new_token = _jwt_with_exp(2_000_000_000)
    calls: list[dict[str, object]] = []

    def fake_refresh(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"access_token": new_token, "expires_in": 3600}

    monkeypatch.setattr(auth, "_request_oauth_refresh", fake_refresh)
    auth_file.write_text(
        json.dumps(
            {
                "example": {
                    "oauth": {
                        "access_token": old_token,
                        "refresh_token": "refresh-existing",
                        "token_url": "https://provider.example/oauth/token",
                        "client_id": "example-client",
                    }
                }
            }
        )
    )

    credential = auth.get_credential("example")
    saved = json.loads(auth_file.read_text(encoding="utf-8"))

    assert credential.kind == "oauth"
    assert credential.bearer_token == new_token
    assert saved["example"]["oauth"]["access_token"] == new_token
    assert saved["example"]["oauth"]["refresh_token"] == "refresh-existing"
    assert calls == [
        {
            "token_url": "https://provider.example/oauth/token",
            "refresh_token": "refresh-existing",
            "client_id": "example-client",
            "client_secret": None,
            "scope": None,
        }
    ]


def test_get_credential_passes_generic_oauth_refresh_scope(
    auth_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_token = _jwt_with_exp(1)
    new_token = _jwt_with_exp(2_000_000_000)
    seen: dict[str, object] = {}

    def fake_refresh(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"access_token": new_token}

    monkeypatch.setattr(auth, "_request_oauth_refresh", fake_refresh)
    auth_file.write_text(
        json.dumps(
            {
                "example": {
                    "oauth": {
                        "access_token": old_token,
                        "refresh_token": "refresh-existing",
                        "token_url": "https://provider.example/oauth/token",
                        "client_id": "example-client",
                        "scope": "example.read",
                    }
                }
            }
        )
    )

    credential = auth.get_credential("example")

    assert credential.bearer_token == new_token
    assert seen["scope"] == "example.read"


def test_get_credential_refreshes_near_expiry_oauth(
    auth_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_token = _jwt_with_exp(int(auth.time.time()) + 10)
    new_token = _jwt_with_exp(2_000_000_000)

    monkeypatch.setattr(
        auth,
        "_request_oauth_refresh",
        lambda **_kwargs: {
            "access_token": new_token,
            "refresh_token": "refresh-rotated",
        },
    )
    auth_file.write_text(
        json.dumps(
            {
                "example": {
                    "oauth": {
                        "access_token": old_token,
                        "refresh_token": "refresh-existing",
                        "token_url": "https://provider.example/oauth/token",
                        "client_id": "example-client",
                    }
                }
            }
        )
    )

    credential = auth.get_credential("example")
    saved = json.loads(auth_file.read_text(encoding="utf-8"))

    assert credential.bearer_token == new_token
    assert saved["example"]["oauth"]["refresh_token"] == "refresh-rotated"


def test_get_credential_does_not_refresh_unexpired_oauth(
    auth_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = _jwt_with_exp(2_000_000_000)

    def fail_refresh(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("refresh should not be called")

    monkeypatch.setattr(auth, "_request_oauth_refresh", fail_refresh)
    auth_file.write_text(json.dumps({"example": {"oauth": {"access_token": token}}}))

    assert auth.get_credential("example").bearer_token == token


def test_get_credential_refreshes_openai_with_provider_defaults(
    auth_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_token = _jwt_with_exp(1)
    new_token = _jwt_with_exp(2_000_000_000)
    seen: dict[str, object] = {}

    def fake_refresh(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"access_token": new_token}

    monkeypatch.setattr(auth, "_request_oauth_refresh", fake_refresh)
    codex_scope = "openid profile email offline_access api.connectors.read api.connectors.invoke"
    auth_file.write_text(
        json.dumps(
            {
                "openai": {
                    "oauth": {
                        "access_token": old_token,
                        "refresh_token": "refresh-existing",
                        "scope": codex_scope,
                    }
                }
            }
        )
    )

    credential = auth.get_credential("openai")

    assert credential.bearer_token == new_token
    assert seen["token_url"] == "https://auth.openai.com/oauth/token"
    assert seen["client_id"] == "app_EMoamEEZ73f0CkXaXp7hrann"
    assert seen["scope"] is None


def test_login_openai_codex_persists_openai_oauth(
    auth_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_urls: list[str] = []
    opened_urls: list[str] = []
    seen_exchange: dict[str, str] = {}

    monkeypatch.setattr(auth, "_generate_pkce", lambda: ("verifier", "challenge"))
    monkeypatch.setattr(auth.secrets, "token_hex", lambda _n: "state_123")
    monkeypatch.setattr(
        auth,
        "_start_openai_codex_callback_server",
        lambda *, state: None,
    )

    def fake_exchange(**kwargs: str) -> dict[str, object]:
        seen_exchange.update(kwargs)
        return {
            "access_token": _jwt_with_payload(
                {
                    "exp": 2_000_000_000,
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "acct_123",
                    },
                }
            ),
            "refresh_token": "refresh-token",
            "expires_in": 3600,
        }

    monkeypatch.setattr(auth, "_request_openai_codex_token", fake_exchange)

    credential = auth.login_openai_codex(
        on_auth=seen_urls.append,
        prompt=lambda _message: "code=auth-code&state=state_123",
        open_browser=lambda url: opened_urls.append(url) is None,
    )

    saved = json.loads(auth_file.read_text())
    oauth = saved["openai"]["oauth"]
    assert credential.kind == "oauth"
    assert credential.bearer_token == oauth["access_token"]
    assert oauth["refresh_token"] == "refresh-token"
    assert oauth["token_url"] == "https://auth.openai.com/oauth/token"
    assert oauth["client_id"] == "app_EMoamEEZ73f0CkXaXp7hrann"
    assert (
        oauth["scope"]
        == "openid profile email offline_access api.connectors.read api.connectors.invoke"
    )
    assert oauth["account_id"] == "acct_123"
    assert seen_exchange == {"code": "auth-code", "verifier": "verifier"}
    assert len(seen_urls) == 1
    assert opened_urls == seen_urls
    assert "code_challenge=challenge" in seen_urls[0]
    assert "state=state_123" in seen_urls[0]
    query = parse.parse_qs(parse.urlparse(seen_urls[0]).query)
    assert query["scope"] == [
        "openid profile email offline_access api.connectors.read api.connectors.invoke"
    ]


def test_login_openai_codex_accepts_callback_after_manual_prompt(
    auth_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeServer:
        def __init__(self) -> None:
            self.waits: list[float] = []
            self.closed = False

        def wait(self, timeout: float) -> str | None:
            self.waits.append(timeout)
            if len(self.waits) == 1:
                return None
            return "auth-code"

        def close(self) -> None:
            self.closed = True

    fake_server = FakeServer()
    seen_exchange: dict[str, str] = {}

    monkeypatch.setattr(auth, "_generate_pkce", lambda: ("verifier", "challenge"))
    monkeypatch.setattr(auth.secrets, "token_hex", lambda _n: "state_123")
    monkeypatch.setattr(
        auth,
        "_start_openai_codex_callback_server",
        lambda *, state: fake_server,
    )

    def fake_exchange(**kwargs: str) -> dict[str, object]:
        seen_exchange.update(kwargs)
        return {
            "access_token": _jwt_with_payload(
                {
                    "exp": 2_000_000_000,
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "acct_123",
                    },
                }
            ),
            "refresh_token": "refresh-token",
            "expires_in": 3600,
        }

    monkeypatch.setattr(auth, "_request_openai_codex_token", fake_exchange)

    credential = auth.login_openai_codex(
        prompt=lambda _message: "",
        open_browser=None,
        callback_timeout_seconds=0.01,
    )

    saved = json.loads(auth_file.read_text())
    assert credential.bearer_token == saved["openai"]["oauth"]["access_token"]
    assert seen_exchange == {"code": "auth-code", "verifier": "verifier"}
    assert fake_server.waits == [0.01, 0]
    assert fake_server.closed is True


def test_login_openai_codex_replaces_existing_openai_oauth(
    auth_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_file.write_text(
        json.dumps(
            {
                "openai": {
                    "api_key": {"value": "keep-api-key"},
                    "oauth": {
                        "access_token": "old-access-token",
                        "refresh_token": "old-refresh-token",
                        "scope": "openid profile email offline_access",
                        "stale_field": "remove-me",
                    },
                },
                "anthropic": {"api_key": {"value": "keep-anthropic-key"}},
            }
        )
    )
    monkeypatch.setattr(auth, "_generate_pkce", lambda: ("verifier", "challenge"))
    monkeypatch.setattr(auth.secrets, "token_hex", lambda _n: "state_123")
    monkeypatch.setattr(
        auth,
        "_start_openai_codex_callback_server",
        lambda *, state: None,
    )
    monkeypatch.setattr(
        auth,
        "_request_openai_codex_token",
        lambda **_kwargs: {
            "access_token": _jwt_with_payload(
                {
                    "exp": 2_000_000_000,
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "acct_123",
                    },
                }
            ),
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
        },
    )

    credential = auth.login_openai_codex(
        prompt=lambda _message: "code=auth-code&state=state_123",
        open_browser=None,
    )

    saved = json.loads(auth_file.read_text())
    oauth = saved["openai"]["oauth"]
    assert credential.bearer_token == oauth["access_token"]
    assert oauth["access_token"] != "old-access-token"
    assert oauth["refresh_token"] == "new-refresh-token"
    assert (
        oauth["scope"]
        == "openid profile email offline_access api.connectors.read api.connectors.invoke"
    )
    assert "stale_field" not in oauth
    assert saved["openai"]["api_key"] == {"value": "keep-api-key"}
    assert saved["anthropic"] == {"api_key": {"value": "keep-anthropic-key"}}


def test_auth_credential_repr_hides_bearer_token() -> None:
    credential = auth.AuthCredential(
        kind="oauth",
        bearer_token="secret-token",
        source="test",
    )

    assert "secret-token" not in repr(credential)


def test_load_auth_returns_full_dict(auth_file: Path) -> None:
    payload = {
        "anthropic": {"api_key": {"api_key": "sk-ant-abc"}},
        "openai": {"api_key": {"api_key": "sk-oai"}},
    }
    auth_file.write_text(json.dumps(payload))
    assert auth.load_auth() == payload


def test_get_api_key_tolerates_extra_fields_under_vendor(auth_file: Path) -> None:
    """Future fields like org_id/base_url must not break api_key retrieval."""
    auth_file.write_text(
        json.dumps(
            {
                "anthropic": {
                    "api_key": {
                        "api_key": "sk-ant-abc",
                        "base_url": "https://example.com",
                    }
                },
                "openai": {
                    "api_key": {
                        "api_key": "sk-oai",
                        "org_id": "org-123",
                    }
                },
            }
        )
    )
    assert auth.get_api_key("anthropic") == "sk-ant-abc"
    assert auth.get_api_key("openai") == "sk-oai"


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------


def test_load_auth_missing_file_raises_filenotfound_with_path(auth_file: Path) -> None:
    assert not auth_file.exists()
    with pytest.raises(FileNotFoundError) as excinfo:
        auth.load_auth()
    msg = str(excinfo.value)
    assert str(auth_file) in msg
    # Message should hint at the expected JSON shape (actionable).
    assert "anthropic" in msg
    assert "api_key" in msg


def test_get_api_key_missing_file_raises_filenotfound_with_path(auth_file: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        auth.get_api_key("anthropic")
    assert str(auth_file) in str(excinfo.value)


# ---------------------------------------------------------------------------
# Vendor not in file
# ---------------------------------------------------------------------------


def test_get_api_key_vendor_not_present_raises_keyerror(auth_file: Path) -> None:
    auth_file.write_text(json.dumps({"anthropic": {"api_key": "sk-ant-abc"}}))
    with pytest.raises(KeyError) as excinfo:
        auth.get_api_key("openai")
    # KeyError stringifies with quotes around the message; check the args[0].
    msg = excinfo.value.args[0]
    assert "openai" in msg
    assert str(auth_file) in msg


# ---------------------------------------------------------------------------
# Vendor present but no api_key
# ---------------------------------------------------------------------------


def test_get_api_key_missing_api_key_field_raises_keyerror(auth_file: Path) -> None:
    auth_file.write_text(json.dumps({"anthropic": {"org_id": "org-123"}}))
    with pytest.raises(KeyError) as excinfo:
        auth.get_api_key("anthropic")
    msg = excinfo.value.args[0]
    assert "anthropic" in msg
    assert "api_key" in msg
    assert str(auth_file) in msg


def test_get_api_key_empty_string_api_key_raises_keyerror(auth_file: Path) -> None:
    auth_file.write_text(json.dumps({"anthropic": {"api_key": ""}}))
    with pytest.raises(KeyError) as excinfo:
        auth.get_api_key("anthropic")
    msg = excinfo.value.args[0]
    assert "anthropic" in msg
    assert str(auth_file) in msg


def test_get_api_key_non_string_api_key_raises_keyerror(auth_file: Path) -> None:
    auth_file.write_text(json.dumps({"anthropic": {"api_key": 12345}}))
    with pytest.raises(KeyError) as excinfo:
        auth.get_api_key("anthropic")
    assert "anthropic" in excinfo.value.args[0]


def test_get_api_key_vendor_entry_not_object_raises_keyerror(auth_file: Path) -> None:
    auth_file.write_text(json.dumps({"anthropic": "sk-ant-abc"}))
    with pytest.raises(KeyError) as excinfo:
        auth.get_api_key("anthropic")
    assert "anthropic" in excinfo.value.args[0]


# ---------------------------------------------------------------------------
# Malformed JSON / wrong top-level shape
# ---------------------------------------------------------------------------


def test_load_auth_malformed_json_raises_valueerror_with_path(auth_file: Path) -> None:
    auth_file.write_text("{not valid json")
    with pytest.raises(ValueError) as excinfo:
        auth.load_auth()
    msg = str(excinfo.value)
    assert str(auth_file) in msg
    # Helpful: indicate it's a JSON problem.
    assert "JSON" in msg or "json" in msg


def test_load_auth_top_level_not_object_raises_valueerror(auth_file: Path) -> None:
    auth_file.write_text(json.dumps(["anthropic", "openai"]))
    with pytest.raises(ValueError) as excinfo:
        auth.load_auth()
    msg = str(excinfo.value)
    assert str(auth_file) in msg
    assert "object" in msg or "dict" in msg.lower()


def test_get_api_key_malformed_json_propagates_valueerror(auth_file: Path) -> None:
    auth_file.write_text("garbage")
    with pytest.raises(ValueError):
        auth.get_api_key("anthropic")


# ---------------------------------------------------------------------------
# wattle package does not eagerly import auth
# ---------------------------------------------------------------------------


def test_importing_wattle_does_not_auto_load_auth() -> None:
    """`import wattle` must not read the auth file (no side effects)."""
    import importlib
    import sys

    # Drop any cached references so we re-execute wattle/__init__.py.
    for name in list(sys.modules):
        if name == "wattle" or name.startswith("wattle."):
            del sys.modules[name]

    wattle = importlib.import_module("wattle")
    # The package should not have eagerly bound `auth` as an attribute.
    assert not hasattr(wattle, "auth")
