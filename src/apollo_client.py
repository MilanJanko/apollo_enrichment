from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


class ApolloHTTPError(RuntimeError):
    def __init__(self, status_code: int, message: str, payload: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


@dataclass
class ApolloClient:
    api_key: str
    base_url: str = "https://api.apollo.io/api/v1"
    timeout_s: int = 40
    max_retries: int = 2  # for network/5xx only

    def _post(
        self,
        path: str,
        payload: Dict[str, Any],
        *,
        retry_on_auth_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        Apollo auth can vary depending on plan/account:
        - Common: X-Api-Key header
        - Sometimes: api_key in JSON payload (legacy)
        We try header first, then fallback to payload api_key if 401/403.
        """
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Api-Key": self.api_key,  # attempt #1
        }

        def do_request(p: Dict[str, Any]) -> Tuple[int, Dict[str, Any] | str]:
            resp = requests.post(url, json=p, headers=headers, timeout=self.timeout_s)
            if resp.headers.get("content-type", "").lower().startswith("application/json"):
                try:
                    return resp.status_code, resp.json()
                except Exception:
                    return resp.status_code, resp.text
            return resp.status_code, resp.text

        # Retry loop for transient errors
        last_err: Optional[ApolloHTTPError] = None
        for attempt in range(self.max_retries + 1):
            status, body = do_request(payload)

            # Auth fallback: retry once with api_key inside payload
            if retry_on_auth_fallback and status in (401, 403):
                payload2 = dict(payload)
                payload2["api_key"] = self.api_key
                status2, body2 = do_request(payload2)
                if 200 <= status2 < 300 and isinstance(body2, dict):
                    return body2
                # if still failing, raise with second response
                msg = f"Apollo request failed (auth). HTTP {status2}. Body: {body2}"
                raise ApolloHTTPError(status2, msg, payload2)

            # Success
            if 200 <= status < 300 and isinstance(body, dict):
                return body

            # Non-retryable client errors (422 etc.)
            if 400 <= status < 500:
                msg = f"Apollo request failed. HTTP {status}. Body: {body}"
                raise ApolloHTTPError(status, msg, payload)

            # Retryable server/network errors
            last_err = ApolloHTTPError(status, f"Apollo server error HTTP {status}. Body: {body}", payload)

            if attempt < self.max_retries:
                time.sleep(0.8 * (attempt + 1))

        # Exhausted retries
        raise last_err or ApolloHTTPError(500, "Apollo request failed after retries.", payload)


def get_client_from_env() -> ApolloClient:
    api_key = os.getenv("APOLLO_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing APOLLO_API_KEY in environment (.env).")
    return ApolloClient(api_key=api_key)
