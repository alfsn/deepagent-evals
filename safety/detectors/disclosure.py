"""Deterministic disclosure check: a brand is mentioned but there's no #ad / #sponsored marker.

A candidate, not a decision: an organic "I love this brand" isn't sponsorship, so the LLM decides.
"""

from safety.models import Post, Severity, Signal
from safety.taxonomy import load_config


def is_disclosed(text: str) -> bool:
    low = text.casefold()
    return any(m.casefold() in low for m in load_config("risk_lists.yaml")["disclosure_markers"])


class DisclosureDetector:
    name = "disclosure"
    version = "disc-1"

    async def detect(self, post: Post, prior: list[Signal]) -> list[Signal]:
        if not post.brand_mentions or is_disclosed(post.text):
            return []
        return [Signal(dimension="undisclosed_sponsorship", source="disclosure", detector_version=self.version,
                       kind="candidate", severity=Severity.MEDIUM, post_id=post.id,
                       evidence=[f"mentions {', '.join(post.brand_mentions)} without a disclosure marker"])]
