from __future__ import annotations

from typing import Any, Dict, List, Optional

from .apollo_client import ApolloClient


def people_api_search(
    client: ApolloClient,
    domain: str,
    *,
    person_titles: list[str] = None,
    per_page: int = 100,
    page: int = 1,
) -> Dict[str, Any]:
    """
    POST /v1/mixed_people/search
    Body:
      - q_organization_domains: domain
      - person_titles: list[str]
      - per_page, page

    Note: This is intentionally NOT using organization_id or person_seniorities.
    """
    payload: Dict[str, Any] = {
        "q_organization_domains_list": [domain],
        "per_page": per_page,
        "page": page,
        "include_similar_titles": True,
    }
    if person_titles:
        payload["person_titles"] = person_titles

    return client._post("/mixed_people/api_search", payload)
