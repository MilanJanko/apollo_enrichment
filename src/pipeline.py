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

RETRY_STATUSES = {"error", "no_enrich", "no_match"}  # retry only these


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
    if isinstance(data, dict) and any(k in data for k in ("email", "first_name", "last_name")):
        return data
    return None


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
    person_id = candidate.get("id") or candidate.get("person_id") or (enriched or {}).get("id")

    email = (enriched or {}).get("email") or (enriched or {}).get("email_address")

    full_name = " ".join([p for p in [first, last] if p]) or None

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

        # source mapping
        "source_row": source_row,               # 1-based original row index
        "source_website": source_website,

        # org info
        "company_domain": company_domain,
        "organization_id": organization_id,
        "company_name": company_name,
        "employee_count": employee_count,

        # person info
        "apollo_person_id": person_id,
        "first_name": first,
        "last_name": last,
        "full_name": full_name,
        "title": title,
        "email": email,
        "linkedin_url": linkedin,
    }


def enrich_one_company(
    client,
    cache: OrgSearchCache,
    *,
    source_row: int,
    website: str,
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

    # STEP 0 — Org search (paid) but cached
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
    full_name = " ".join(filter(None, [candidate.get("first_name"), candidate.get("last_name")])) or None
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


def _save_checkpoint(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def enrich_dataframe(
    df: pd.DataFrame,
    website_column: str,
    *,
    start_row: int = 1,
    end_row: int | None = None,
    progress_cb=None,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """
    start_row/end_row are 1-based inclusive.
    Saves checkpoint after each row if checkpoint_path is provided.
    """
    client = get_client_from_env()
    cache = OrgSearchCache("data/cache/org_cache.json")

    total_rows = len(df)
    start_idx = max(start_row - 1, 0)
    end_idx = min(end_row if end_row is not None else total_rows, total_rows)

    if start_idx >= end_idx:
        raise ValueError("Invalid row range selected.")

    subset = df.iloc[start_idx:end_idx].copy()
    websites = subset[website_column].astype(str).tolist()

    total = len(websites)
    results: List[Dict[str, Any]] = []

    for i, website in enumerate(websites, start=1):
        source_row = start_idx + i  # 1-based original row number
        row_result = enrich_one_company(
            client,
            cache,
            source_row=source_row,
            website=website,
        )
        results.append(row_result)

        if progress_cb:
            progress_cb(i / total, f"Enriched row {source_row} of {total_rows}")

        if checkpoint_path:
            state = {
                "total_rows": total_rows,
                "website_column": website_column,
                "start_row": start_row,
                "end_row": end_idx,
                "last_completed_row": source_row,
                "results_so_far": len(results),
            }
            _save_checkpoint(checkpoint_path, state)

    return pd.DataFrame(results)


def retry_failed_only(
    df_original: pd.DataFrame,
    website_column: str,
    previous_output_df: pd.DataFrame,
    *,
    progress_cb=None,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """
    Retries only failed rows from a previous output.
    Uses source_row + source_website for mapping.
    """
    client = get_client_from_env()
    cache = OrgSearchCache("data/cache/org_cache.json")

    failed = previous_output_df[previous_output_df["status"].isin(RETRY_STATUSES)].copy()
    if failed.empty:
        return pd.DataFrame([])

    # Use original row order
    failed.sort_values("source_row", inplace=True)

    total = len(failed)
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
            progress_cb(i / total, f"Retry {i}/{total} (row {source_row})")

        if checkpoint_path:
            state = {
                "mode": "retry_failed_only",
                "website_column": website_column,
                "last_completed_source_row": source_row,
                "results_so_far": len(results),
                "total_to_retry": total,
            }
            _save_checkpoint(checkpoint_path, state)

    return pd.DataFrame(results)

def append_enrichment_to_original_df(
    original_df: pd.DataFrame,
    enrichment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Appends Apollo enrichment columns back into the original dataframe
    using source_row (1-based).

    Does NOT drop or reorder original columns.
    """

    df = original_df.copy()

    # Define output columns (safe, prefixed)
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

    # Ensure columns exist
    for col in column_map.values():
        if col not in df.columns:
            df[col] = None

    # Write back row by row
    for r in enrichment_df.itertuples(index=False):
        source_row = int(getattr(r, "source_row", 0))
        if source_row <= 0 or source_row > len(df):
            continue

        idx = source_row - 1  # convert to 0-based

        for src_col, target_col in column_map.items():
            if hasattr(r, src_col):
                value = getattr(r, src_col)
                if value is not None:
                    df.at[idx, target_col] = value

    return df


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--website_column", default="company_website")
    parser.add_argument("--start_row", type=int, default=1)
    parser.add_argument("--end_row", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    end_row = None if args.end_row == 0 else args.end_row

    out = enrich_dataframe(
        df=df,
        website_column=args.website_column,
        start_row=args.start_row,
        end_row=end_row,
        progress_cb=None,
    )

    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
