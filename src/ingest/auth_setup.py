"""One-time interactive WHOOP OAuth 2.0 bootstrap.

Run **manually, once**::

    python -m src.ingest.auth_setup

It walks the authorization-code flow end to end:

1. Reads ``WHOOP_CLIENT_ID`` / ``WHOOP_CLIENT_SECRET`` (from the
   environment / ``.env``, or prompts for them).
2. Prints the WHOOP consent URL (with the scopes the agent needs,
   including ``offline`` so a refresh token is issued).
3. Spins up a throwaway ``localhost`` server to catch the redirect and
   capture the ``code``.
4. Exchanges the code for a refresh token.
5. Writes ``WHOOP_CLIENT_ID`` / ``WHOOP_CLIENT_SECRET`` /
   ``WHOOP_REFRESH_TOKEN`` into ``.env`` (created if absent).

Nothing here runs on import — only :func:`main` (under ``__main__``)
performs network or filesystem side effects, so the service never
triggers the live OAuth flow. The recurring token *refresh* is handled
automatically by :class:`~src.ingest.whoop_client.WhoopClient`; this
script is only the initial grant.
"""

from __future__ import annotations

import http.server
import os
import secrets
import socketserver
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv, set_key

from src.ingest.whoop_client import (
    WHOOP_AUTH_URL,
    WHOOP_SCOPES,
    WHOOP_TOKEN_URL,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

DEFAULT_REDIRECT_URI = "http://localhost:8080/callback"


# --------------------------------------------------------------------------
# Redirect catcher
# --------------------------------------------------------------------------

class _CallbackResult:
    """Mutable slot the request handler drops the auth code/state into."""

    code: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None


def _make_handler(result: _CallbackResult, expected_state: str):
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # silence stdout noise
            pass

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            result.error = (params.get("error") or [None])[0]
            result.state = (params.get("state") or [None])[0]
            result.code = (params.get("code") or [None])[0]

            ok = (
                result.error is None
                and result.code is not None
                and result.state == expected_state
            )
            body = (
                b"<h2>WHOOP authorization complete.</h2>"
                b"<p>You can close this tab and return to the terminal.</p>"
                if ok
                else b"<h2>WHOOP authorization failed.</h2>"
                b"<p>Check the terminal for details.</p>"
            )
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

    return _Handler


def _await_redirect(redirect_uri: str, expected_state: str) -> _CallbackResult:
    """Serve exactly one request on the redirect host/port and return it."""
    parsed = urllib.parse.urlparse(redirect_uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 80

    result = _CallbackResult()
    handler = _make_handler(result, expected_state)
    with socketserver.TCPServer((host, port), handler) as httpd:
        print(f"Waiting for the WHOOP redirect on {redirect_uri} ...")
        thread = threading.Thread(target=httpd.handle_request)
        thread.start()
        thread.join()
    return result


# --------------------------------------------------------------------------
# Flow steps
# --------------------------------------------------------------------------

def _prompt(name: str, current: Optional[str]) -> str:
    if current:
        return current
    value = input(f"Enter {name}: ").strip()
    if not value:
        sys.exit(f"{name} is required.")
    return value


def _authorize_url(client_id: str, redirect_uri: str, state: str) -> str:
    query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": WHOOP_SCOPES,
            "state": state,
        }
    )
    return f"{WHOOP_AUTH_URL}?{query}"


def _exchange_code(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> str:
    """Trade the authorization code for a refresh token."""
    resp = httpx.post(
        WHOOP_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        },
        timeout=30.0,
    )
    if resp.status_code != 200:
        sys.exit(
            f"Token exchange failed (HTTP {resp.status_code}): "
            f"{resp.text[:300]}"
        )
    payload = resp.json()
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        sys.exit(
            "WHOOP did not return a refresh_token. Make sure the 'offline' "
            "scope is enabled for your app in the WHOOP developer portal."
        )
    return refresh_token


def main() -> None:
    load_dotenv(ENV_PATH)

    client_id = _prompt(
        "WHOOP_CLIENT_ID", os.environ.get("WHOOP_CLIENT_ID")
    )
    client_secret = _prompt(
        "WHOOP_CLIENT_SECRET", os.environ.get("WHOOP_CLIENT_SECRET")
    )
    redirect_uri = os.environ.get(
        "WHOOP_REDIRECT_URI", DEFAULT_REDIRECT_URI
    )

    state = secrets.token_urlsafe(24)
    url = _authorize_url(client_id, redirect_uri, state)

    print("\n1. Open this URL in your browser and approve access:\n")
    print(f"   {url}\n")
    print(
        "   (Make sure this exact redirect URI is registered for your app "
        f"in the WHOOP developer portal: {redirect_uri})\n"
    )
    try:
        webbrowser.open(url)
    except Exception:  # pragma: no cover - headless / no browser
        pass

    result = _await_redirect(redirect_uri, state)
    if result.error:
        sys.exit(f"WHOOP returned an error: {result.error}")
    if not result.code:
        sys.exit("No authorization code received.")
    if result.state != state:
        sys.exit("State mismatch — possible CSRF; aborting.")

    refresh_token = _exchange_code(
        result.code, client_id, client_secret, redirect_uri
    )

    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), "WHOOP_CLIENT_ID", client_id)
    set_key(str(ENV_PATH), "WHOOP_CLIENT_SECRET", client_secret)
    set_key(str(ENV_PATH), "WHOOP_REFRESH_TOKEN", refresh_token)

    print(
        f"\nDone. Credentials written to {ENV_PATH}.\n"
        "The refresh token is long-lived; WhoopClient refreshes the "
        "access token automatically from here on."
    )


if __name__ == "__main__":
    main()
