from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


TITLE_PRIORITY = [
    "ceo",
    "founder",
    "owner",
    "managing director",
    "executive director",
    "president",
    "coo",
    "chief operating officer",
    "vp operations",
    "vp of operations",
    "director of operations",
    "vp sales",
    "vp of sales",
    "director of sales",
    "vp marketing",
    "vp of marketing",
    "director of marketing",
    "vp business development",
    "vp of business development",
    "director of business development",
]

SENIORITY_PRIORITY = [
    "owner",
    "founder",
    "c_suite",
    "partner",
    "vp",
    "head",
    "director",
    "manager",
    "senior",
    "entry",
    "intern",
]


@dataclass
class Candidate:
    raw: Dict[str, Any]
    score: float
    reason: str


def _safe_str(x: Any) -> str:
    return (x or "").strip().lower()


def _contains_any(title: str, needles: List[str]) -> Optional[int]:
    for i, n in enumerate(needles):
        if n in title:
            return i
    return None


def score_person(person: Dict[str, Any]) -> Candidate:
    """
    Deterministic scoring:
    - Title priority (lower index = better)
    - Seniority priority (lower index = better)
    - Has linkedin_url bonus
    """
    title = _safe_str(person.get("title"))
    seniority = _safe_str(person.get("seniority"))

    title_rank = _contains_any(title, TITLE_PRIORITY)
    seniority_rank = _contains_any(seniority, SENIORITY_PRIORITY)

    # Base score: higher is better
    score = 0.0
    reasons = []

    if title_rank is not None:
        score += 100.0 - title_rank
        reasons.append(f"title_rank={title_rank}")
    else:
        score += 10.0
        reasons.append("title_rank=None")

    if seniority_rank is not None:
        score += 50.0 - (seniority_rank * 0.5)
        reasons.append(f"seniority_rank={seniority_rank}")
    else:
        reasons.append("seniority_rank=None")

    linkedin = person.get("linkedin_url") or person.get("linkedin_url_normalized")
    if linkedin:
        score += 5.0
        reasons.append("linkedin=Y")
    else:
        reasons.append("linkedin=N")

    return Candidate(raw=person, score=score, reason="; ".join(reasons))


def pick_best_contact(people: List[Dict[str, Any]]) -> Optional[Candidate]:
    if not people:
        return None

    scored = [score_person(p) for p in people]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[0]
