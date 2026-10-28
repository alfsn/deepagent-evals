"""Data contracts shared by every detector, the aggregator and the pipeline."""

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field


class Severity(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


Verdict = Literal["safe", "review", "unsafe"]
# "embedding" is reserved for the future EmbeddingDetector (see docs/safety-embeddings.md).
Source = Literal["keyword", "disclosure", "llm", "creator_rules", "embedding"]


class Post(BaseModel):
    id: str
    handle: str
    lang: str = "en"
    text: str
    brand_mentions: list[str] = []
    days_ago: int = 0


class Creator(BaseModel):
    handle: str
    name: str
    niche: str
    audience: dict
    past_brands: list[str] = []


class Signal(BaseModel):
    """One piece of evidence from one detector about one dimension."""

    dimension: str
    source: Source
    detector_version: str
    # candidate = "look at this" (keywords); decision = authoritative for its dimension (LLM, rules).
    kind: Literal["candidate", "decision"]
    severity: Severity = Severity.NONE
    evidence: list[str] = []
    reasoning: str | None = None
    post_id: str | None = None
    error: str | None = None


class DimensionScore(BaseModel):
    dimension: str
    verdict: Verdict
    severity: Severity
    confirmed: bool  # backed by a decision (LLM / rules), not just keyword candidates
    post_ids: list[str] = []
    why: str


class SafetyReport(BaseModel):
    """Brand-independent, so it can be cached. Brand fit is applied afterwards by suitability.py."""

    handle: str
    verdict: Verdict
    dimensions: dict[str, DimensionScore]
    signals: list[Signal] = Field(default_factory=list)
    llm_calls: int = 0

    def compact(self) -> dict:
        """What the agent sees: verdicts and short evidence, never full post text."""
        return {
            "handle": self.handle,
            "verdict": self.verdict,
            "dimensions": {
                d: {"verdict": s.verdict, "severity": s.severity.name.lower(), "posts": s.post_ids, "why": s.why[:300]}
                for d, s in self.dimensions.items()
                if s.verdict != "safe"
            },
        }
