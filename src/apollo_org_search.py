from __future__ import annotations
from typing import Dict, Any, Optional

from .apollo_client import ApolloClient
from .org_cache import OrgSearchCache


def organization_search_by_domain(
    client: ApolloClient,
    domain: str,
    cache: OrgSearchCache,
) -> Optional[Dict[str, Any]]:
    """
    Correct Apollo endpoint:
    POST /api/v1/mixed_companies/search

    Mandatory cache:
    - returns cached org if present
    - negative caches not found domains
    """

    cached = cache.get(domain)
    if cached is not None:
        # cached={} means previously "not found"
        return cached or None

    payload = {
        "q_organization_domains_list": [domain],
        "per_page": 5,
        "page": 1,
    }

    resp = client._post("/mixed_companies/search", payload)

    companies = (
        resp.get("companies")
        or resp.get("organizations")
        or resp.get("data", {}).get("companies")
        or resp.get("data", {}).get("organizations")
        or []
    )

    if not companies:
        cache.set_not_found(domain)
        cache.flush()
        return None

    org = companies[0]
    cache.set_found(domain, org)
    cache.flush()
    return org
