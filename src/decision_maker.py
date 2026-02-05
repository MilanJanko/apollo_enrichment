from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# --------------------------------------------------
# Data structure
# --------------------------------------------------

@dataclass
class Candidate:
    raw: Dict[str, Any]
    score: float
    reason: str


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _safe_str(x: Any) -> str:
    return (x or "").strip()


def _has_linkedin(p: Dict[str, Any]) -> bool:
    return bool(p.get("linkedin_url") or p.get("linkedin_url_normalized"))


def _has_email(p: Dict[str, Any]) -> bool:
    return bool(p.get("email") or p.get("email_address"))


# --------------------------------------------------
# Picker (NO REJECTION, TS-LIKE BEHAVIOR)
# --------------------------------------------------

def pick_best_contact(
    people: List[Dict[str, Any]],
    *,
    debug: bool = False,
) -> Optional[Candidate]:
    """
    TS-style behavior:
    - NEVER reject contacts
    - Rank by completeness
    - Prefer LinkedIn
    - Prefer titles
    - Return exactly ONE best contact
    """

    if not people:
        if debug:
            print("[DECISION] No people received")
        return None

    scored: List[Candidate] = []

    for p in people:
        first = _safe_str(p.get("first_name"))
        last = _safe_str(p.get("last_name"))
        title = _safe_str(p.get("title"))
        seniority = _safe_str(p.get("seniority"))

        score = 0.0
        reasons: list[str] = []

        # Name signal
        if first or last:
            score += 10
            reasons.append("name=Y")
        else:
            reasons.append("name=N")

        # Title signal (important)
        if title:
            score += 20
            reasons.append("title=Y")
        else:
            reasons.append("title=N")

        # Seniority (small bonus only)
        if seniority:
            score += 2
            reasons.append("seniority=Y")
        else:
            reasons.append("seniority=N")

        # LinkedIn (strong preference, TS does this implicitly)
        if _has_linkedin(p):
            score += 15
            reasons.append("linkedin=Y")
        else:
            reasons.append("linkedin=N")

        # Email (nice-to-have, not required)
        if _has_email(p):
            score += 5
            reasons.append("email=Y")
        else:
            reasons.append("email=N")

        scored.append(
            Candidate(
                raw=p,
                score=score,
                reason="; ".join(reasons),
            )
        )

    # Highest score first
    scored.sort(key=lambda c: c.score, reverse=True)

    if debug:
        print(f"[DECISION] Scored {len(scored)} candidates (top 10):")
        for c in scored[:10]:
            p = c.raw
            print(
                " -",
                f"{(p.get('first_name') or '')} {(p.get('last_name') or '')}".strip(),
                "|",
                p.get("title"),
                "|",
                p.get("seniority"),
                "| linkedin:",
                _has_linkedin(p),
                "| email:",
                _has_email(p),
                "| score:",
                c.score,
                "|",
                c.reason,
            )

    return scored[0]
