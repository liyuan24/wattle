"""Credential auth for Wattle.

Reads vendor credentials from Wattle's canonical JSON file at
``~/.wattle/auth.json``.

Canonical file format::

    {
      "anthropic": {
        "api_key": {"api_key": "sk-ant-..."}
      },
      "deepseek": {
        "api_key": {"api_key": "sk-..."}
      },
      "kimi": {
        "api_key": {"api_key": "sk-..."}
      },
      "minimax": {
        "api_key": {"api_key": "sk-..."}
      },
      "openai": {
        "oauth": {
          "access_token": "...",
          "refresh_token": "...",
          "token_url": "...",
          "client_id": "...",
          "expires_at": 1779133973
        }
      }
    }

The top-level keys are vendors (for example ``"anthropic"``, ``"deepseek"``,
``"kimi"``, ``"minimax"``, ``"openai"``). Each value is an object containing
auth method objects: ``"api_key"`` or ``"oauth"``. If a vendor has both, OAuth
is used by default for generic credential lookup. Callers that need a platform
API key should request an API-key credential explicitly. Legacy flat entries are
still accepted for compatibility.

Both OpenAI providers (Chat Completions and Responses) share one OpenAI key,
hence the vendor name ``"openai"`` rather than provider-specific names.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import html
import json
import queue
import secrets
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal
from urllib import error, parse, request

# Module-level constant. Importable AND monkeypatchable: tests and alternate
# entry points override this attribute to redirect reads.
AUTH_PATH: Path = Path.home() / ".wattle" / "auth.json"
OAUTH_REFRESH_SKEW_SECONDS = 300
OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
OPENAI_CODEX_SCOPE = "openid profile email offline_access api.connectors.read api.connectors.invoke"
OPENAI_CODEX_REQUIRED_SCOPES = frozenset({"api.connectors.read", "api.connectors.invoke"})
OPENAI_CODEX_AUTH_CLAIM_PATH = "https://api.openai.com/auth"

_PROVIDER_OAUTH_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        "token_url": OPENAI_CODEX_TOKEN_URL,
        "client_id": OPENAI_CODEX_CLIENT_ID,
    },
}


_EXPECTED_SHAPE = (
    '{\n'
    '  "anthropic": {"api_key": {"api_key": "sk-ant-..."}},\n'
    '  "deepseek": {"api_key": {"api_key": "sk-..."}},\n'
    '  "kimi": {"api_key": {"api_key": "sk-..."}},\n'
    '  "minimax": {"api_key": {"api_key": "sk-..."}},\n'
    '  "openai": {"oauth": {"access_token": "...", "refresh_token": "..."}}\n'
    '}'
)


@dataclass(frozen=True)
class AuthCredential:
    """A provider-ready bearer credential.

    ``bearer_token`` is intentionally hidden from ``repr`` so test failures and
    logs do not leak API keys or OAuth access tokens.
    """

    kind: Literal["api_key", "oauth"]
    bearer_token: str = field(repr=False)
    source: str
    expires_at: int | None = None


def load_auth() -> dict:
    """Read and parse :data:`AUTH_PATH`. Return the parsed JSON object.

    Raises:
        FileNotFoundError: If :data:`AUTH_PATH` does not exist. The message
            includes the expected path and the expected JSON shape.
        ValueError: If the file exists but is not valid JSON, or if the
            top-level value is not a JSON object. The message includes the
            path and the expected shape.
    """
    path = AUTH_PATH
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Wattle auth file not found at {path}. "
            f"Create it with the following shape:\n{_EXPECTED_SHAPE}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Wattle auth file at {path} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno}). "
            f"Expected shape:\n{_EXPECTED_SHAPE}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Wattle auth file at {path} must contain a JSON object at the "
            f"top level, got {type(data).__name__}. "
            f"Expected shape:\n{_EXPECTED_SHAPE}"
        )

    return data


def _load_json_object(path: Path, *, label: str) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{label} auth file at {path} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"{label} auth file at {path} must contain a JSON object at the "
            f"top level, got {type(data).__name__}."
        )

    return data


def _jwt_payload(token: str) -> dict[str, object] | None:
    """Decode a JWT payload without validating its signature.

    This is only used to read non-sensitive metadata such as ``exp`` so Wattle
    can reject obviously expired OAuth access tokens before SDK construction.
    """
    parts = token.split(".")
    if len(parts) < 2:
        return None

    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        parsed = json.loads(decoded)
    except (ValueError, UnicodeDecodeError):
        return None

    return parsed if isinstance(parsed, dict) else None


def _jwt_expires_at(token: str) -> int | None:
    payload = _jwt_payload(token)
    if payload is None:
        return None

    exp = payload.get("exp")
    if isinstance(exp, int):
        return exp
    if isinstance(exp, float):
        return int(exp)
    return None


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _generate_pkce() -> tuple[str, str]:
    verifier = _base64url(secrets.token_bytes(32))
    challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _build_openai_codex_authorization_url(
    *,
    challenge: str,
    state: str,
    originator: str = "wattle",
) -> str:
    params = {
        "response_type": "code",
        "client_id": OPENAI_CODEX_CLIENT_ID,
        "redirect_uri": OPENAI_CODEX_REDIRECT_URI,
        "scope": OPENAI_CODEX_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": originator,
    }
    return f"{OPENAI_CODEX_AUTHORIZE_URL}?{parse.urlencode(params)}"


def _parse_authorization_input(value: str) -> tuple[str | None, str | None]:
    raw = value.strip()
    if not raw:
        return None, None

    try:
        parsed_url = parse.urlparse(raw)
    except ValueError:
        parsed_url = parse.ParseResult("", "", "", "", "", "")
    if parsed_url.scheme and parsed_url.netloc:
        params = parse.parse_qs(parsed_url.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        return code, state

    if "#" in raw:
        code, state = raw.split("#", 1)
        return code or None, state or None

    if "code=" in raw:
        params = parse.parse_qs(raw)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        return code, state

    return raw, None


def _openai_codex_account_id(access_token: str) -> str | None:
    payload = _jwt_payload(access_token)
    auth_claim = (
        payload.get(OPENAI_CODEX_AUTH_CLAIM_PATH)
        if payload is not None
        else None
    )
    if not isinstance(auth_claim, dict):
        return None
    account_id = auth_claim.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) and account_id else None


def _coerce_expires_at(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _oauth_expires_at(token_map: dict) -> int | None:
    expires_at = _coerce_expires_at(token_map.get("expires_at"))
    if expires_at is not None:
        return expires_at

    access_token = token_map.get("access_token")
    if isinstance(access_token, str):
        return _jwt_expires_at(access_token)
    return None


def _oauth_needs_refresh(token_map: dict) -> bool:
    expires_at = _oauth_expires_at(token_map)
    if expires_at is None:
        return False
    return expires_at <= int(time.time()) + OAUTH_REFRESH_SKEW_SECONDS


def _oauth_string(
    token_map: dict,
    key: str,
    *,
    vendor: str,
    path: Path,
    default: str | None = None,
) -> str:
    value = token_map.get(key, default)
    if isinstance(value, str) and value:
        return value
    raise KeyError(
        f"OAuth credential for vendor {vendor!r} in {path} needs a non-empty "
        f"{key!r} field to refresh."
    )


def _request_oauth_refresh(
    *,
    token_url: str,
    refresh_token: str,
    client_id: str,
    client_secret: str | None = None,
    scope: str | None = None,
) -> dict:
    form: dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    if client_secret:
        form["client_secret"] = client_secret
    if scope:
        form["scope"] = scope

    req = request.Request(
        token_url,
        data=parse.urlencode(form).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"OAuth refresh endpoint returned HTTP {exc.code}: {detail}"
        ) from exc
    except error.URLError as exc:
        raise ValueError(f"OAuth refresh request failed: {exc.reason}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("OAuth refresh endpoint returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise ValueError("OAuth refresh endpoint returned a non-object JSON response.")
    return data


def _request_openai_codex_token(
    *,
    code: str,
    verifier: str,
) -> dict:
    form = {
        "grant_type": "authorization_code",
        "client_id": OPENAI_CODEX_CLIENT_ID,
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": OPENAI_CODEX_REDIRECT_URI,
    }
    req = request.Request(
        OPENAI_CODEX_TOKEN_URL,
        data=parse.urlencode(form).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"OpenAI Codex OAuth token endpoint returned HTTP {exc.code}: {detail}"
        ) from exc
    except error.URLError as exc:
        raise ValueError(f"OpenAI Codex OAuth token request failed: {exc.reason}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenAI Codex OAuth token endpoint returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise ValueError("OpenAI Codex OAuth token endpoint returned a non-object response.")
    return data


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.chmod(0o600)
    tmp_path.replace(path)


def _success_html(message: str) -> bytes:
    escaped = html.escape(message)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Authentication successful</title></head>"
        f"<body><h1>Authentication successful</h1><p>{escaped}</p></body></html>"
    ).encode()


def _error_html(message: str) -> bytes:
    escaped = html.escape(message)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Authentication failed</title></head>"
        f"<body><h1>Authentication failed</h1><p>{escaped}</p></body></html>"
    ).encode()


class _OpenAICodexCallbackServer:
    def __init__(self, *, state: str) -> None:
        self._codes: queue.Queue[str | None] = queue.Queue(maxsize=1)
        codes = self._codes

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                parsed = parse.urlparse(self.path)
                if parsed.path != "/auth/callback":
                    self._respond(404, _error_html("Callback route not found."))
                    return

                params = parse.parse_qs(parsed.query)
                if params.get("state", [None])[0] != state:
                    self._respond(400, _error_html("State mismatch."))
                    return

                code = params.get("code", [None])[0]
                if not code:
                    self._respond(400, _error_html("Missing authorization code."))
                    return

                self._respond(
                    200,
                    _success_html(
                        "OpenAI authentication completed. You can close this window."
                    ),
                )
                with contextlib.suppress(queue.Full):
                    codes.put_nowait(code)

            def _respond(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 1455), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def wait(self, timeout: float) -> str | None:
        try:
            return self._codes.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1)


def _start_openai_codex_callback_server(
    *,
    state: str,
) -> _OpenAICodexCallbackServer | None:
    try:
        server = _OpenAICodexCallbackServer(state=state)
    except OSError:
        return None
    server.start()
    return server


def _apply_oauth_refresh_response(
    *,
    vendor: str,
    path: Path,
    token_map: dict,
    response: dict,
) -> None:
    access_token = response.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise ValueError(
            f"OAuth refresh response for vendor {vendor!r} from {path} did not "
            f"include a non-empty access_token."
        )

    token_map["access_token"] = access_token

    refresh_token = response.get("refresh_token")
    if isinstance(refresh_token, str) and refresh_token:
        token_map["refresh_token"] = refresh_token

    expires_at = _jwt_expires_at(access_token)
    expires_in = response.get("expires_in")
    if expires_at is None and isinstance(expires_in, int | float):
        expires_at = int(time.time()) + int(expires_in)
    if expires_at is not None:
        token_map["expires_at"] = expires_at


def _openai_codex_token_map_from_response(response: dict) -> dict[str, object]:
    access_token = response.get("access_token")
    refresh_token = response.get("refresh_token")
    if not isinstance(access_token, str) or not access_token:
        raise ValueError("OpenAI Codex OAuth response did not include access_token.")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise ValueError("OpenAI Codex OAuth response did not include refresh_token.")
    account_id = _openai_codex_account_id(access_token)
    if account_id is None:
        raise ValueError("OpenAI Codex OAuth token is missing chatgpt_account_id.")

    expires_at = _jwt_expires_at(access_token)
    expires_in = response.get("expires_in")
    if expires_at is None and isinstance(expires_in, int | float):
        expires_at = int(time.time()) + int(expires_in)

    token_map: dict[str, object] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_url": OPENAI_CODEX_TOKEN_URL,
        "client_id": OPENAI_CODEX_CLIENT_ID,
        "scope": OPENAI_CODEX_SCOPE,
        "account_id": account_id,
    }
    if expires_at is not None:
        token_map["expires_at"] = expires_at
    return token_map


def save_openai_codex_oauth(token_map: dict[str, object]) -> AuthCredential:
    """Persist OpenAI Codex OAuth credentials into Wattle's auth file."""
    path = AUTH_PATH
    try:
        data = load_auth()
    except FileNotFoundError:
        data = {}

    entry = data.get("openai")
    if entry is None:
        entry = {}
        data["openai"] = entry
    if not isinstance(entry, dict):
        raise ValueError(
            f"Entry for vendor 'openai' in {path} must be an object, "
            f"got {type(entry).__name__}."
        )

    entry["oauth"] = dict(token_map)
    _write_json_atomic(path, data)
    return _oauth_credential_from_token_map(
        vendor="openai",
        path=path,
        token_map=entry["oauth"],
        source=f"{path} openai.oauth",
    )


def login_openai_codex(
    *,
    on_auth: Callable[[str], None] | None = None,
    prompt: Callable[[str], str] = input,
    open_browser: Callable[[str], bool] | None = webbrowser.open,
    callback_timeout_seconds: float = 300,
    originator: str = "wattle",
) -> AuthCredential:
    """Run the OpenAI Codex OAuth flow and persist the resulting credential.

    This is intentionally OpenAI Codex-only. Wattle stores the resulting token
    under ``openai.oauth`` because ``openai_codex`` is wired through the existing
    OpenAI vendor credential path.
    """
    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(16)
    auth_url = _build_openai_codex_authorization_url(
        challenge=challenge,
        state=state,
        originator=originator,
    )
    server = _start_openai_codex_callback_server(state=state)

    try:
        if on_auth is not None:
            on_auth(auth_url)
        if open_browser is not None:
            with contextlib.suppress(Exception):
                open_browser(auth_url)

        code = server.wait(callback_timeout_seconds) if server is not None else None
        if code is None:
            raw = prompt("Paste the authorization code or full redirect URL: ")
            code, pasted_state = _parse_authorization_input(raw)
            if code is None and server is not None:
                code = server.wait(0)
            if pasted_state is not None and pasted_state != state:
                raise ValueError("OpenAI Codex OAuth state mismatch.")

        if not code:
            raise ValueError("OpenAI Codex OAuth did not return an authorization code.")

        token_response = _request_openai_codex_token(code=code, verifier=verifier)
        token_map = _openai_codex_token_map_from_response(token_response)
        return save_openai_codex_oauth(token_map)
    finally:
        if server is not None:
            server.close()


def _refresh_oauth_token_map_if_needed(
    *,
    vendor: str,
    path: Path,
    token_map: dict,
    root_data: dict,
) -> None:
    if not _oauth_needs_refresh(token_map):
        return

    defaults = _PROVIDER_OAUTH_DEFAULTS.get(vendor, {})
    token_url = _oauth_string(
        token_map,
        "token_url",
        vendor=vendor,
        path=path,
        default=defaults.get("token_url"),
    )
    client_id = _oauth_string(
        token_map,
        "client_id",
        vendor=vendor,
        path=path,
        default=defaults.get("client_id"),
    )
    refresh_token = _oauth_string(token_map, "refresh_token", vendor=vendor, path=path)

    client_secret = token_map.get("client_secret")
    if not isinstance(client_secret, str):
        client_secret = None
    scope = token_map.get("scope")
    if vendor == "openai" or not isinstance(scope, str):
        scope = None

    response = _request_oauth_refresh(
        token_url=token_url,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        scope=scope,
    )
    _apply_oauth_refresh_response(
        vendor=vendor,
        path=path,
        token_map=token_map,
        response=response,
    )
    _write_json_atomic(path, root_data)


def _oauth_credential_from_token_map(
    *,
    vendor: str,
    path: Path,
    token_map: dict,
    source: str,
) -> AuthCredential:
    access_token = token_map.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise KeyError(
            f"OAuth credential for vendor {vendor!r} in {path} is missing a "
            f"non-empty 'access_token' field."
        )

    expires_at = _oauth_expires_at(token_map)

    return AuthCredential(
        kind="oauth",
        bearer_token=access_token,
        source=source,
        expires_at=expires_at,
    )


def _scope_set(value: object) -> set[str]:
    if isinstance(value, str):
        return {scope for scope in value.split() if scope}
    if isinstance(value, list):
        return {scope for scope in value if isinstance(scope, str) and scope}
    return set()


def _oauth_token_scopes(token_map: dict) -> set[str]:
    access_token = token_map.get("access_token")
    if isinstance(access_token, str):
        payload = _jwt_payload(access_token)
        if payload is not None and "scp" in payload:
            return _scope_set(payload.get("scp"))
    return _scope_set(token_map.get("scope"))


def _has_openai_codex_scopes(token_map: dict) -> bool:
    return _oauth_token_scopes(token_map) >= OPENAI_CODEX_REQUIRED_SCOPES


def _oauth_credential_from_vendor_entry(
    *,
    vendor: str,
    path: Path,
    entry: dict,
) -> AuthCredential | None:
    nested = entry.get("oauth")
    if nested is not None:
        if not isinstance(nested, dict):
            raise KeyError(
                f"'oauth' for vendor {vendor!r} in {path} must be an object, "
                f"got {type(nested).__name__}."
            )
        return _oauth_credential_from_token_map(
            vendor=vendor,
            path=path,
            token_map=nested,
            source=f"{path} {vendor}.oauth",
        )

    # Legacy flat OAuth shape from early Wattle builds.
    if entry.get("auth_type") == "oauth":
        return _oauth_credential_from_token_map(
            vendor=vendor,
            path=path,
            token_map=entry,
            source=str(path),
        )

    # Legacy Codex-shaped OpenAI entry.
    tokens = entry.get("tokens")
    if isinstance(tokens, dict):
        return _oauth_credential_from_token_map(
            vendor=vendor,
            path=path,
            token_map=tokens,
            source=f"{path} {vendor}.tokens",
        )

    return None


def _api_key_credential_from_vendor_entry(
    *,
    vendor: str,
    path: Path,
    entry: dict,
) -> AuthCredential | None:
    api_key_entry = entry.get("api_key")
    if api_key_entry is None:
        return None

    if isinstance(api_key_entry, dict):
        api_key = api_key_entry.get("api_key")
        source = f"{path} {vendor}.api_key"
    else:
        # Legacy flat API-key shape from early Wattle builds.
        api_key = api_key_entry
        source = str(path)

    if isinstance(api_key, str) and api_key:
        return AuthCredential(kind="api_key", bearer_token=api_key, source=source)

    raise KeyError(
        f"'api_key' for vendor {vendor!r} in {path} must contain a non-empty "
        f"string field named 'api_key', got {type(api_key).__name__!r}."
    )


def get_openai_codex_credential() -> AuthCredential:
    """Return an OAuth credential suitable for ChatGPT's Codex endpoint."""
    path = AUTH_PATH
    data = load_auth()

    entry = data.get("openai")
    if isinstance(entry, dict):
        oauth_entry = entry.get("oauth")
        if isinstance(oauth_entry, dict):
            _refresh_oauth_token_map_if_needed(
                vendor="openai",
                path=path,
                token_map=oauth_entry,
                root_data=data,
            )
            if _has_openai_codex_scopes(oauth_entry):
                return _oauth_credential_from_token_map(
                    vendor="openai",
                    path=path,
                    token_map=oauth_entry,
                    source=f"{path} openai.oauth",
                )

    raise ValueError(
        "OpenAI Codex OAuth token is missing required Codex scopes. "
        "Run /login again with this Wattle build."
    )


def get_api_key_credential(vendor: str) -> AuthCredential:
    """Return an API-key credential for ``vendor``.

    This is intentionally stricter than :func:`get_credential`: OAuth access
    tokens, including the ChatGPT Codex OAuth token stored under
    ``openai.oauth``, are not valid API keys for OpenAI-compatible platform
    endpoints.
    """
    path = AUTH_PATH
    data = load_auth()

    if vendor not in data:
        raise _missing_vendor_error(vendor, path)

    entry = data[vendor]
    if not isinstance(entry, dict):
        raise KeyError(
            f"Entry for vendor {vendor!r} in {path} must be an object with "
            f"an 'api_key' method object, got {type(entry).__name__}."
        )

    credential = _api_key_credential_from_vendor_entry(
        vendor=vendor,
        path=path,
        entry=entry,
    )
    if credential is not None:
        return credential

    raise KeyError(
        f"Vendor {vendor!r} in Wattle auth file at {path} is missing the "
        f"'api_key' method object required by this provider. Add it: "
        f"{{{vendor!r}: {{'api_key': {{'api_key': '...'}}}}}}."
    )


def _missing_vendor_error(vendor: str, path: Path) -> KeyError:
    return KeyError(
        f"Vendor {vendor!r} not found in Wattle auth file at {path}. "
        f"Add an entry like {{{vendor!r}: {{'api_key': {{'api_key': '...'}}}}}} "
        f"to that file."
    )


def get_credential(vendor: str) -> AuthCredential:
    """Return a provider-ready credential for ``vendor``.

    Each vendor entry should contain method objects: ``api_key`` or ``oauth``.
    If both are present, OAuth is used by default.

    Raises:
        KeyError: If ``vendor`` is not configured, or its entry is malformed.
        FileNotFoundError: If the Wattle auth file is missing.
        ValueError: If a credential file is malformed JSON.
    """
    path = AUTH_PATH

    try:
        data = load_auth()
    except FileNotFoundError:
        raise

    if vendor not in data:
        raise _missing_vendor_error(vendor, path)

    entry = data[vendor]
    if not isinstance(entry, dict):
        raise KeyError(
            f"Entry for vendor {vendor!r} in {path} must be an object with "
            f"an 'api_key' or 'oauth' method object, got {type(entry).__name__}."
        )

    has_oauth = "oauth" in entry

    credential = _oauth_credential_from_vendor_entry(
        vendor=vendor,
        path=path,
        entry=entry,
    )
    if credential is not None and has_oauth:
        oauth_entry = entry["oauth"]
        if not isinstance(oauth_entry, dict):
            raise KeyError(
                f"'oauth' for vendor {vendor!r} in {path} must be an object, "
                f"got {type(oauth_entry).__name__}."
            )
        _refresh_oauth_token_map_if_needed(
            vendor=vendor,
            path=path,
            token_map=oauth_entry,
            root_data=data,
        )
        return _oauth_credential_from_token_map(
            vendor=vendor,
            path=path,
            token_map=oauth_entry,
            source=f"{path} {vendor}.oauth",
        )

    if credential is not None and entry.get("auth_type") == "oauth":
        _refresh_oauth_token_map_if_needed(
            vendor=vendor,
            path=path,
            token_map=entry,
            root_data=data,
        )
        return _oauth_credential_from_token_map(
            vendor=vendor,
            path=path,
            token_map=entry,
            source=str(path),
        )

    api_key_credential = _api_key_credential_from_vendor_entry(
        vendor=vendor,
        path=path,
        entry=entry,
    )
    if api_key_credential is not None:
        return api_key_credential

    if credential is not None:
        return credential

    if "api_key" not in entry:
        raise KeyError(
            f"Vendor {vendor!r} in Wattle auth file at {path} is missing the "
            f"'api_key' or 'oauth' method object. Add it: "
            f"{{{vendor!r}: {{'api_key': {{'api_key': '...'}}}}}}."
        )
    raise KeyError(
        f"Vendor {vendor!r} in Wattle auth file at {path} does not contain a "
        f"usable 'api_key' or 'oauth' credential."
    )


def get_api_key(vendor: str) -> str:
    """Return the provider bearer token for ``vendor``.

    This compatibility wrapper returns API keys for API-key credentials and
    OAuth access tokens for OAuth credentials. New code should use
    :func:`get_credential` when it needs to distinguish credential kinds.

    Raises:
        KeyError: If ``vendor`` is not configured, or its entry is malformed.
        FileNotFoundError: If the Wattle auth file is missing.
        ValueError: If a credential file is malformed JSON.
    """
    return get_credential(vendor).bearer_token
