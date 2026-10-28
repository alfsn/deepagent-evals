"""Which posts are worth an LLM call. Config, not code: thresholds live here, the pipeline just asks."""

import hashlib
from dataclasses import dataclass

from safety.models import Post, Signal


@dataclass(frozen=True)
class EscalationPolicy:
    # Clean posts are sampled too, so we see what the keyword layer misses (and can measure its recall).
    sample_rate: float = 0.3
    seed: str = "safety-v1"
    all_posts: bool = False  # evals: run every layer on every post to measure each one independently

    def _sampled(self, post: Post) -> bool:
        # Hash-based, so the same post always gets the same decision (reproducible runs, stable evals).
        h = int(hashlib.sha256(f"{self.seed}:{post.id}".encode()).hexdigest()[:8], 16)
        return h / 0xFFFFFFFF < self.sample_rate

    def select(self, posts: list[Post], screened: list[Signal]) -> list[Post]:
        flagged = {s.post_id for s in screened}
        return [p for p in posts if self.all_posts or p.id in flagged or self._sampled(p)]
