"""Brand fit on top of a brand-independent SafetyReport. Pure function: same report + tolerance -> same answer.

tolerance = {dimension: max acceptable severity ("none" | "low" | "medium" | "high")}.
Dimensions a brand doesn't mention fall back to the report's own verdict.
"""

from typing import Literal

from safety.models import SafetyReport, Severity
from safety.taxonomy import load_taxonomy

Fit = Literal["ok", "review", "reject"]


def suitability(report: SafetyReport, tolerance: dict[str, str] | None = None) -> dict:
    tolerance = tolerance or {}
    taxonomy = load_taxonomy()
    reject, review = [], []
    for dim, score in report.dimensions.items():
        if score.verdict == "safe":
            continue
        line = f"{dim} ({score.verdict}, {score.severity.name.lower()}): {score.why[:200]}"
        if taxonomy[dim].floor and score.verdict == "unsafe":
            reject.append("FLOOR " + line)  # no brand can tolerate a floor violation
        elif not score.confirmed:
            review.append(line)  # unconfirmed evidence: a person (or the safety-reviewer agent) decides
        elif dim in tolerance:
            if score.severity > Severity[tolerance[dim].upper()]:
                reject.append(line)  # within the brand's tolerance -> fine for this brand
        else:
            (reject if score.verdict == "unsafe" else review).append(line)
    fit: Fit = "reject" if reject else "review" if review else "ok"
    return {"handle": report.handle, "fit": fit, "reject_reasons": reject, "review_reasons": review}
