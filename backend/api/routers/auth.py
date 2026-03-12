import secrets
import time
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import RedirectResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from src.core.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])

STATE_TTL_SECONDS = 600
_oauth_state_store: dict[str, dict] = {}


def _append_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query[key] = value
    return urlunparse(parsed._replace(query=urlencode(query)))


@lru_cache(maxsize=1)
def _token_serializer() -> URLSafeTimedSerializer:
    settings = get_settings()
    return URLSafeTimedSerializer(settings.auth_secret_key, salt="legal-assistant-auth")


def _cleanup_oauth_states() -> None:
    now = time.time()
    stale = [k for k, v in _oauth_state_store.items() if v.get("exp", 0) < now]
    for key in stale:
        _oauth_state_store.pop(key, None)


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


@router.get("/google/login")
def google_login(next_url: str | None = Query(default=None, alias="next")) -> RedirectResponse:
    settings = get_settings()
    if not settings.google_client_id or not settings.google_client_secret or not settings.google_redirect_uri:
        raise HTTPException(status_code=500, detail="Google OAuth is not configured.")

    _cleanup_oauth_states()
    state = secrets.token_urlsafe(24)
    _oauth_state_store[state] = {
        "exp": time.time() + STATE_TTL_SECONDS,
        "next": next_url or settings.frontend_base_url,
    }

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "include_granted_scopes": "true",
        "state": state,
        "prompt": "select_account",
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return RedirectResponse(auth_url)


@router.get("/google/callback")
def google_callback(code: str | None = None, state: str | None = None, error: str | None = None) -> RedirectResponse:
    settings = get_settings()
    redirect_base = settings.frontend_base_url

    if error:
        return RedirectResponse(_append_query_param(redirect_base, "auth_error", error))
    if not code or not state:
        return RedirectResponse(_append_query_param(redirect_base, "auth_error", "missing_code_or_state"))

    state_data = _oauth_state_store.pop(state, None)
    if not state_data or state_data.get("exp", 0) < time.time():
        return RedirectResponse(_append_query_param(redirect_base, "auth_error", "invalid_state"))

    next_url = state_data.get("next") or redirect_base

    try:
        token_resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=30,
        )
        token_resp.raise_for_status()
        token_data = token_resp.json()
        raw_id_token = token_data.get("id_token")
        if not raw_id_token:
            return RedirectResponse(_append_query_param(next_url, "auth_error", "missing_id_token"))

        id_info = google_id_token.verify_oauth2_token(
            raw_id_token,
            google_requests.Request(),
            settings.google_client_id,
        )
        user_payload = {
            "id": id_info.get("sub", ""),
            "email": id_info.get("email", ""),
            "name": id_info.get("name", "User"),
            "picture": id_info.get("picture", ""),
        }
        token = _token_serializer().dumps(user_payload)
        redirect_url = _append_query_param(next_url, "auth_token", token)
        return RedirectResponse(redirect_url)
    except Exception:
        return RedirectResponse(_append_query_param(next_url, "auth_error", "google_exchange_failed"))


@router.get("/me")
def auth_me(authorization: str | None = Header(default=None)) -> dict:
    settings = get_settings()
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token.")

    try:
        user = _token_serializer().loads(token, max_age=settings.auth_token_ttl_seconds)
        if not isinstance(user, dict) or not user.get("id"):
            raise HTTPException(status_code=401, detail="Invalid auth token.")
        return user
    except SignatureExpired as exc:
        raise HTTPException(status_code=401, detail="Auth token expired.") from exc
    except (BadSignature, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid auth token.") from exc
