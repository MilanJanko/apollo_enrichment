from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from .apollo_client import ApolloHTTPError, get_client_from_env
from .apollo_search import people_api_search
from .apollo_enrich import people_enrichment_match
from .decision_maker import pick_best_contact
from .domain_utils import normalize_domain
from .apollo_org_search import organization_search_by_domain
from .org_cache import OrgSearchCache


DEFAULT_SENIORITIES = ["owner", "founder", "c_suite", "vp", "director"]
RETRY_STATUSES = {"error", "no_enrich", "no_match"}


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def extract_people_from_search(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(resp.get("people"), list):
        return resp["people"]
    data = resp.get("data")
    if isinstance(data, dict) and isinstance(data.get("people"), list):
        return data["people"]
    for k in ("contacts", "results"):
        v = resp.get(k)
        if isinstance(v, list):
            return v
        if isinstance(data, dict) and isinstance(data.get(k), list):
            return data[k]
    return []


def extract_enriched_person(resp: Dict[str, Any]) -> Dict[str, Any] | None:
    if isinstance(resp.get("person"), dict):
        return resp["person"]
    data = resp.get("data")
    if isinstance(data, dict) and isinstance(data.get("person"), dict):
        return data["person"]
    if isinstance(data, dict) and any(
        k in data for k in ("email", "first_name", "last_name")
    ):
        return data
    return None


# -------------------------------------------------
# Output builder
# -------------------------------------------------
def build_output_row(
    *,
    source_row: int,
    source_website: str,
    company_domain: str,
    candidate: Dict[str, Any],
    enriched: Dict[str, Any] | None,
    status: str,
    note: str,
    org: Dict[str, Any] | None = None,
) -> Dict[str, Any]:

    first = candidate.get("first_name") or (enriched or {}).get("first_name")
    last = candidate.get("last_name") or (enriched or {}).get("last_name")
    title = candidate.get("title") or (enriched or {}).get("title")
    linkedin = candidate.get("linkedin_url") or (enriched or {}).get("linkedin_url")
    person_id = (
        candidate.get("id")
        or candidate.get("person_id")
        or (enriched or {}).get("id")
    )

    email = (enriched or {}).get("email") or (enriched or {}).get("email_address")
    full_name = " ".join(p for p in [first, last] if p) or None

    company_name = None
    employee_count = None
    organization_id = None

    if org:
        company_name = org.get("name")
        employee_count = org.get("estimated_num_employees") or org.get("num_employees")
        organization_id = org.get("id")

    return {
        "status": status,
        "note": note,

        "source_row": source_row,
        "source_website": source_website,

        "company_domain": company_domain,
        "organization_id": organization_id,
        "company_name": company_name,
        "employee_count": employee_count,

        "apollo_person_id": person_id,
        "first_name": first,
        "last_name": last,
        "full_name": full_name,
        "title": title,
        "email": email,
        "linkedin_url": linkedin,
    }


# -------------------------------------------------
# Core enrichment
# -------------------------------------------------
def enrich_one_company(
    client,
    cache: OrgSearchCache,
    *,
    source_row: int,
    website: str,
    reveal_phone_number: bool = False,
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:

    domain = normalize_domain(website)
    if not domain:
        return build_output_row(
            source_row=source_row,
            source_website=website,
            company_domain=str(website),
            candidate={},
            enriched=None,
            status="skipped",
            note="invalid_domain",
            org=None,
        )

    # STEP 0 — Org search
    try:
        org = organization_search_by_domain(client, domain, cache)
    except ApolloHTTPError as e:
        return build_output_row(
            source_row=source_row,
            source_website=website,
            company_domain=domain,
            candidate={},
            enriched=None,
            status="error",
            note=f"org_search_http_{e.status_code}",
            org=None,
        )

    if not org:
        return build_output_row(
            source_row=source_row,
            source_website=website,
            company_domain=domain,
            candidate={},
            enriched=None,
            status="skipped",
            note="organization_not_found",
            org=None,
        )

    organization_id = org.get("id")

    # STEP 1 — People search (free)
    try:
        search_resp = people_api_search(
            client,
            domains=[domain],
            organization_id=organization_id,
            person_seniorities=DEFAULT_SENIORITIES,
            include_similar_titles=True,
            per_page=10,
        )
        people = extract_people_from_search(search_resp)
    except ApolloHTTPError as e:
        return build_output_row(
            source_row=source_row,
            source_website=website,
            company_domain=domain,
            candidate={},
            enriched=None,
            status="error",
            note=f"people_search_http_{e.status_code}",
            org=org,
        )

    best = pick_best_contact(people)
    if not best:
        return build_output_row(
            source_row=source_row,
            source_website=website,
            company_domain=domain,
            candidate={},
            enriched=None,
            status="no_match",
            note="no_people_found",
            org=org,
        )

    candidate = best.raw
    person_id = candidate.get("id") or candidate.get("person_id")
    full_name = " ".join(
        filter(None, [candidate.get("first_name"), candidate.get("last_name")])
    ) or None
    linkedin = candidate.get("linkedin_url")

    # STEP 2 — Enrichment (paid)
    try:
        enrich_resp = people_enrichment_match(
            client,
            person_id=person_id,
            domain=domain,
            full_name=full_name,
            linkedin_url=linkedin,
            reveal_personal_emails=False,
            reveal_phone_number=reveal_phone_number,
            webhook_url=webhook_url,
        )
        enriched_person = extract_enriched_person(enrich_resp)
    except ApolloHTTPError as e:
        return build_output_row(
            source_row=source_row,
            source_website=website,
            company_domain=domain,
            candidate=candidate,
            enriched=None,
            status="error",
            note=f"enrich_http_{e.status_code}",
            org=org,
        )

    if not enriched_person:
        return build_output_row(
            source_row=source_row,
            source_website=website,
            company_domain=domain,
            candidate=candidate,
            enriched=None,
            status="no_enrich",
            note="enrichment_returned_no_person",
            org=org,
        )

    return build_output_row(
        source_row=source_row,
        source_website=website,
        company_domain=domain,
        candidate=candidate,
        enriched=enriched_person,
        status="ok",
        note=f"picked_score={best.score:.2f} ({best.reason})",
        org=org,
    )


# -------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------
def _save_checkpoint(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# -------------------------------------------------
# Batch enrichment
# -------------------------------------------------
def enrich_dataframe(
    df: pd.DataFrame,
    website_column: str,
    *,
    start_row: int = 1,
    end_row: int | None = None,
    progress_cb=None,
    checkpoint_path: str | None = None,
    reveal_phone_number: bool = False,
    webhook_url: Optional[str] = None,
) -> pd.DataFrame:

    client = get_client_from_env()
    cache = OrgSearchCache("data/cache/org_cache.json")

    total_rows = len(df)
    start_idx = max(start_row - 1, 0)
    end_idx = min(end_row if end_row is not None else total_rows, total_rows)

    subset = df.iloc[start_idx:end_idx].copy()
    websites = subset[website_column].astype(str).tolist()

    results: List[Dict[str, Any]] = []

    for i, website in enumerate(websites, start=1):
        source_row = start_idx + i

        row_result = enrich_one_company(
            client,
            cache,
            source_row=source_row,
            website=website,
            reveal_phone_number=reveal_phone_number,
            webhook_url=webhook_url,
        )
        results.append(row_result)

        if progress_cb:
            progress_cb(i / len(websites), f"Enriched row {source_row}")

        if checkpoint_path:
            _save_checkpoint(
                checkpoint_path,
                {
                    "last_completed_row": source_row,
                    "website_column": website_column,
                    "start_row": start_row,
                    "end_row": end_idx,
                },
            )

    return pd.DataFrame(results)


# -------------------------------------------------
# Retry failed only (UNCHANGED)
# -------------------------------------------------
def retry_failed_only(
    df_original: pd.DataFrame,
    website_column: str,
    previous_output_df: pd.DataFrame,
    *,
    progress_cb=None,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:

    client = get_client_from_env()
    cache = OrgSearchCache("data/cache/org_cache.json")

    failed = previous_output_df[
        previous_output_df["status"].isin(RETRY_STATUSES)
    ].copy()

    if failed.empty:
        return pd.DataFrame([])

    failed.sort_values("source_row", inplace=True)

    results = []

    for i, r in enumerate(failed.itertuples(index=False), start=1):
        source_row = int(getattr(r, "source_row"))
        website = str(getattr(r, "source_website"))

        row_result = enrich_one_company(
            client,
            cache,
            source_row=source_row,
            website=website,
        )
        results.append(row_result)

        if progress_cb:
            progress_cb(i / len(failed), f"Retry row {source_row}")

        if checkpoint_path:
            _save_checkpoint(
                checkpoint_path,
                {
                    "mode": "retry_failed_only",
                    "last_completed_source_row": source_row,
                },
            )

    return pd.DataFrame(results)


# -------------------------------------------------
# Merge back into original
# -------------------------------------------------
def append_enrichment_to_original_df(
    original_df: pd.DataFrame,
    enrichment_df: pd.DataFrame,
) -> pd.DataFrame:

    df = original_df.copy()

    column_map = {
        "full_name": "apollo_contact_name",
        "title": "apollo_contact_title",
        "email": "apollo_contact_email",
        "linkedin_url": "apollo_contact_linkedin",
        "status": "apollo_status",
        "apollo_person_id": "apollo_person_id",
        "company_name": "apollo_company_name",
        "employee_count": "apollo_employee_count",
    }

    phone_cols = {
        "apollo_phone_number": "apollo_phone_number",
        "apollo_phone_type": "apollo_phone_type",
        "apollo_phone_status": "apollo_phone_status",
    }

    for col in list(column_map.values()) + list(phone_cols.values()):
        if col not in df.columns:
            df[col] = None

    for r in enrichment_df.itertuples(index=False):
        source_row = int(getattr(r, "source_row", 0))
        if source_row <= 0 or source_row > len(df):
            continue

        idx = source_row - 1

        for src, tgt in column_map.items():
            if hasattr(r, src):
                v = getattr(r, src)
                if v is not None:
                    df.at[idx, tgt] = v

    return df


# -------------------------------------------------
# Phase 2 — merge phone events (async)
# -------------------------------------------------
def merge_phone_events_into_original_df(
    df: pd.DataFrame,
    phone_events_path: str = "data/phone_events.jsonl",
) -> pd.DataFrame:

    if not os.path.exists(phone_events_path):
        return df

    with open(phone_events_path, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    phone_by_person: Dict[str, Dict[str, Any]] = {}

    for e in events:
        payload = e.get("payload", {})
        people = payload.get("people", [])

        if not isinstance(people, list):
            continue

        for p in people:
            pid = p.get("id")
            phones = p.get("phone_numbers") or []

            if pid and phones:
                phone_by_person[pid] = phones[0]  # best / first phone

    if not phone_by_person:
        return df

    # Ensure columns exist
    for col in [
        "apollo_phone_number",
        "apollo_phone_type",
        "apollo_phone_status",
    ]:
        if col not in df.columns:
            df[col] = None

    for idx, row in df.iterrows():
        pid = row.get("apollo_person_id")
        phone = phone_by_person.get(pid)

        if phone:
            df.at[idx, "apollo_phone_number"] = (
                phone.get("sanitized_number") or phone.get("raw_number")
            )
            df.at[idx, "apollo_phone_type"] = phone.get("type_cd")
            df.at[idx, "apollo_phone_status"] = phone.get("status_cd")

    return df
