from __future__ import annotations

from typing import Any, Dict, Optional

from .apollo_client import ApolloClient


def people_enrichment_match(
    client: ApolloClient,
    *,
    person_id: Optional[str] = None,
    domain: Optional[str] = None,
    full_name: Optional[str] = None,
    linkedin_url: Optional[str] = None,
    reveal_personal_emails: bool = False,
) -> Dict[str, Any]:
    """
    Calls:
    POST /api/v1/people/match

    Best practice for reliability:
    - Use person_id + domain when available
    """
    payload: Dict[str, Any] = {
        "reveal_personal_emails": bool(reveal_personal_emails),
    }

    if person_id:
        payload["id"] = person_id
    if domain:
        payload["domain"] = domain
    if full_name and not (payload.get("id") or payload.get("linkedin_url")):
        payload["name"] = full_name
    if linkedin_url:
        payload["linkedin_url"] = linkedin_url

    return client._post("/people/match", payload)
