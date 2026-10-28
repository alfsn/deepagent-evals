"""Creator-level rules: things no single post shows (audience age, partnership history, disclosure habits)."""

from safety.detectors.disclosure import is_disclosed
from safety.models import Creator, DimensionScore, Post, Severity, Signal
from safety.taxonomy import load_config


class CreatorRulesDetector:
    name = "creator_rules"
    version = "cr-1"

    def _signal(self, dimension: str, severity: Severity, evidence: list[str], kind: str = "decision") -> Signal:
        return Signal(dimension=dimension, source="creator_rules", detector_version=self.version,
                      kind=kind, severity=severity, evidence=evidence)

    async def detect(self, creator: Creator, posts: list[Post], post_scores: dict[str, DimensionScore]) -> list[Signal]:
        cfg = load_config("risk_lists.yaml")
        out = []

        under_18 = creator.audience.get("under_18_pct", 0)
        if under_18 >= cfg["minor_audience_threshold_pct"]:
            for dim in cfg["age_gated_dimensions"]:
                score = post_scores.get(dim)
                if score and score.verdict != "safe":
                    # Only a decision if the age-gated content itself was confirmed; otherwise it's a candidate.
                    out.append(self._signal("minor_audience_exposure", Severity.MEDIUM,
                                            [f"{under_18}% of audience under 18 + {dim} in posts {score.post_ids}"],
                                            kind="decision" if score.confirmed else "candidate"))

        risky = [
            f"{brand} ({category})"
            for brand in creator.past_brands
            for category, needles in cfg["risky_brand_keywords"].items()
            if any(n in brand.casefold() for n in needles)
        ]
        if risky:
            out.append(self._signal("risky_partnerships", Severity.MEDIUM, risky))

        brand_posts = [p for p in posts if p.brand_mentions]
        if len(brand_posts) >= 2:
            disclosed = sum(is_disclosed(p.text) for p in brand_posts)
            rate, floor = disclosed / len(brand_posts), cfg["min_disclosure_rate"]
            if rate < floor:
                sev = Severity.MEDIUM if rate < floor / 2 else Severity.LOW
                out.append(self._signal("disclosure_rate", sev, [f"{disclosed}/{len(brand_posts)} brand posts disclosed"]))
        return out
