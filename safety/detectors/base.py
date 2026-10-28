"""Detector contracts. Every detector returns list[Signal] and never raises on content:
failures become Signal(error=...), which the aggregator treats as "review", never "safe"."""

from typing import Protocol, runtime_checkable

from safety.models import Creator, DimensionScore, Post, Signal


@runtime_checkable
class PostDetector(Protocol):
    name: str
    version: str

    async def detect(self, post: Post, prior: list[Signal]) -> list[Signal]:
        """prior = signals earlier detectors produced for this post (e.g. keyword hints for the LLM)."""
        ...


@runtime_checkable
class CreatorDetector(Protocol):
    name: str
    version: str

    async def detect(self, creator: Creator, posts: list[Post], post_scores: dict[str, DimensionScore]) -> list[Signal]:
        ...
