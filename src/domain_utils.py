from __future__ import annotations

from urllib.parse import urlparse


def normalize_domain(raw: str) -> str | None:
    """
    Convert website/url/domain into a clean bare domain for Apollo:
    - Removes scheme, www., path, query, fragments
    - Returns hostname only (example: homall.com)
    """
    if not raw:
        return None

    s = raw.strip()
    if not s:
        return None

    # If user passes a bare domain, urlparse treats it as path. Add scheme.
    if "://" not in s:
        s = "https://" + s

    try:
        parsed = urlparse(s)
        host = parsed.hostname
        if not host:
            return None
        host = host.lower().strip()
        if host.startswith("www."):
            host = host[4:]
        # Apollo docs: do not include @, www, etc. We'll keep only hostname.
        if "." not in host:
            return None
        return host
    except Exception:
        return None
