from __future__ import annotations

from typing import Any, Dict, Optional

from .apollo_client import ApolloClient, ApolloHTTPError


def organization_enrich_by_domain(
    client: ApolloClient,
    domain: str,
) -> Optional[Dict[str, Any]]:
    """
    POST /api/v1/organizations/enrich
    - returns organization dict if found
    - returns None if Apollo doesn't know the domain (often HTTP 422)
    - raises on other HTTP errors
    """
    try:
        resp = client._post("/organizations/enrich", {"domain": domain})
    except ApolloHTTPError as e:
        # Apollo returns 422 when domain isn't in DB.
        if getattr(e, "status_code", None) == 422:
            return None
        raise

    org = resp.get("organization") or resp.get("data", {}).get("organization")
    return org or None