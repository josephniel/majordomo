"""Google OAuth 2.0 helpers shared by the Gmail and Calendar connectors.

Credential file format matches what the gongrzhe Gmail MCP writes:

    {
        "access_token": "ya29...",
        "refresh_token": "1//0g...",
        "scope": "https://www.googleapis.com/auth/gmail.modify ...",
        "token_type": "Bearer",
        "expiry_date": 1711234567890   // ms since epoch
    }

So existing Gmail credentials produced by the old npx auth flow keep working
unchanged with the new in-process implementation.
"""
from __future__ import annotations

import http.server
import json
import logging
import socketserver
import time
import webbrowser
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

log = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"

DEFAULT_REDIRECT_PORT = 8765
REFRESH_LEEWAY_S = 60  # refresh if access token expires in <60s


class GoogleOAuthError(Exception):
    pass


class GoogleOAuthClient:
    """Wraps a Google OAuth client (gcp-oauth.keys.json) — handles auth flow + refresh.

    Client identity comes from the OAuth client JSON; per-user tokens live in a
    separate credentials file managed by this class.
    """

    def __init__(self, oauth_keys_path: Path) -> None:
        self.oauth_keys_path = oauth_keys_path
        keys = json.loads(oauth_keys_path.read_text(encoding="utf-8"))
        config = keys.get("installed") or keys.get("web") or keys
        if "client_id" not in config or "client_secret" not in config:
            raise GoogleOAuthError(
                f"{oauth_keys_path} doesn't look like a Google OAuth client JSON"
            )
        self.client_id: str = config["client_id"]
        self.client_secret: str = config["client_secret"]

    # ---- token refresh ----

    async def refresh(self, refresh_token: str) -> dict:
        """Exchange refresh_token for a new access_token. Returns gongrzhe-format dict."""
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )
            if r.status_code != 200:
                raise GoogleOAuthError(
                    f"token refresh failed: {r.status_code} {r.text[:300]}"
                )
            data = r.json()
        expires_in = int(data.get("expires_in", 3600))
        return {
            "access_token": data["access_token"],
            # Google often omits refresh_token on refresh — caller preserves the old one
            "refresh_token": data.get("refresh_token") or refresh_token,
            "scope": data.get("scope", ""),
            "token_type": data.get("token_type", "Bearer"),
            "expiry_date": int(time.time() * 1000) + expires_in * 1000,
        }

    # ---- browser auth flow (synchronous; runs interactively from CLI) ----

    def browser_auth_flow(
        self,
        scopes: list[str],
        port: int = DEFAULT_REDIRECT_PORT,
    ) -> dict:
        """Spawn a local callback server, open a browser to Google's auth URL,
        capture the returned code, exchange for tokens. Returns gongrzhe-format
        credentials dict ready to save.
        """
        redirect_uri = f"http://localhost:{port}/oauth2callback"
        auth_url = AUTH_URL + "?" + urlencode({
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",
            "prompt": "consent",
        })

        captured: dict[str, str] = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                qs = parse_qs(urlparse(self.path).query)
                if "code" in qs:
                    captured["code"] = qs["code"][0]
                if "error" in qs:
                    captured["error"] = qs["error"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authentication complete.</h2>"
                    b"<p>You can close this tab and return to your terminal.</p>"
                    b"</body></html>"
                )

            def log_message(self, *args, **kwargs):  # silence default access log
                pass

        # SO_REUSEADDR so consecutive auth flows don't get blocked by the
        # OS holding the previous socket in TIME_WAIT for ~60s.
        class _ReuseAddrTCPServer(socketserver.TCPServer):
            allow_reuse_address = True

        try:
            server = _ReuseAddrTCPServer(("", port), Handler)
        except OSError as e:
            raise GoogleOAuthError(
                f"could not bind to localhost:{port} for OAuth callback: {e}. "
                "If a different process is using that port, kill it: "
                f"`lsof -i :{port} -P -n` then `kill <pid>`."
            )

        try:
            print(f"\nOpen this URL in your browser to authenticate:\n  {auth_url}\n")
            try:
                webbrowser.open(auth_url)
            except Exception:
                pass
            while not captured:
                server.handle_request()
        finally:
            server.server_close()

        if "error" in captured:
            raise GoogleOAuthError(f"OAuth error: {captured['error']}")
        if "code" not in captured:
            raise GoogleOAuthError("no authorization code received")

        # Exchange code for tokens (sync httpx is fine inside this sync flow).
        # NOTE: must NOT name this `http` — would shadow the `http.server`
        # module reference used earlier in this function.
        with httpx.Client(timeout=30) as client:
            r = client.post(
                TOKEN_URL,
                data={
                    "code": captured["code"],
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
        if r.status_code != 200:
            raise GoogleOAuthError(
                f"token exchange failed: {r.status_code} {r.text[:300]}"
            )
        data = r.json()
        expires_in = int(data.get("expires_in", 3600))
        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "scope": data.get("scope", " ".join(scopes)),
            "token_type": data.get("token_type", "Bearer"),
            "expiry_date": int(time.time() * 1000) + expires_in * 1000,
        }


class CredentialStore:
    """Reads/writes a gongrzhe-format credentials.json with in-memory caching
    and on-the-fly token refresh.
    """

    def __init__(self, oauth: GoogleOAuthClient, credentials_path: Path) -> None:
        self._oauth = oauth
        self._path = credentials_path
        self._cache: Optional[dict] = None

    def _load(self) -> dict:
        if self._cache is None:
            if not self._path.exists():
                raise GoogleOAuthError(f"credentials file not found: {self._path}")
            self._cache = json.loads(self._path.read_text(encoding="utf-8"))
        return self._cache

    def _save(self, creds: dict) -> None:
        self._cache = creds
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(creds, indent=2), encoding="utf-8")

    async def access_token(self) -> str:
        """Return a non-expired access token, refreshing transparently if needed."""
        creds = self._load()
        expiry_ms = int(creds.get("expiry_date", 0))
        now_ms = int(time.time() * 1000)
        if expiry_ms > now_ms + REFRESH_LEEWAY_S * 1000:
            return creds["access_token"]

        refresh_token = creds.get("refresh_token")
        if not refresh_token:
            raise GoogleOAuthError(
                f"no refresh_token in {self._path}; run the auth flow again"
            )
        new_creds = await self._oauth.refresh(refresh_token)
        if not new_creds.get("refresh_token"):
            new_creds["refresh_token"] = refresh_token
        self._save(new_creds)
        return new_creds["access_token"]
