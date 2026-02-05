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
    reveal_phone_number: bool = False,
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls:
    POST /api/v1/people/match

    IMPORTANT:
    - Phone reveal is async
    - webhook_url is REQUIRED when reveal_phone_number=True
    """

    if reveal_phone_number and not webhook_url:
        raise ValueError(
            "webhook_url must be provided when reveal_phone_number=True"
        )

    payload: Dict[str, Any] = {
        "reveal_personal_emails": bool(reveal_personal_emails),
        "reveal_phone_number": bool(reveal_phone_number),
    }

    if reveal_phone_number:
        payload["webhook_url"] = webhook_url

    if person_id:
        payload["id"] = person_id
    if domain:
        payload["domain"] = domain
    if linkedin_url:
        payload["linkedin_url"] = linkedin_url
    if full_name and not (person_id or linkedin_url):
        payload["name"] = full_name

    return client._post("/people/match", payload)
