from __future__ import annotations

import os
from typing import Any, Dict, List

from .apollo_client import ApolloClient


def people_api_search(
    client: ApolloClient,
    domains: list[str],
    *,
    organization_id: str | None = None,
    per_page: int | None = None,
    page: int = 1,
    person_seniorities: list[str] | None = None,
    person_titles: list[str] | None = None,
    include_similar_titles: bool = True,
) -> dict:
    payload = {
        "q_organization_domains_list": domains,
        "per_page": per_page or 10,
        "page": page,
        "include_similar_titles": include_similar_titles,
    }

    if organization_id:
        payload["organization_ids"] = [organization_id]

    if person_seniorities:
        payload["person_seniorities"] = person_seniorities

    if person_titles:
        payload["person_titles"] = person_titles

    return client._post("/mixed_people/api_search", payload)

