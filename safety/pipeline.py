"""Cascade: cheap screeners on every post -> expensive detectors on escalated posts -> creator rules -> report.

Everything is injected, so new detectors (embeddings, Llama Guard) slot in without changing this file.
"""

from safety.aggregate import Aggregator, PrecedenceAggregator
from safety.detectors.base import CreatorDetector, PostDetector
from safety.detectors.creator_rules import CreatorRulesDetector
from safety.detectors.disclosure import DisclosureDetector
from safety.detectors.keywords import KeywordDetector
from safety.detectors.llm_classifier import LLMClassifier
from safety.escalation import EscalationPolicy
from safety.models import Creator, Post, SafetyReport
from safety.taxonomy import preload


class SafetyPipeline:
    def __init__(self, screeners: list[PostDetector], escalated: list[PostDetector],
                 creator_detectors: list[CreatorDetector], aggregator: Aggregator, escalation: EscalationPolicy):
        self.screeners = screeners
        self.escalated = escalated
        self.creator_detectors = creator_detectors
        self.aggregator = aggregator
        self.escalation = escalation

    async def run(self, creator: Creator, posts: list[Post]) -> SafetyReport:
        screened = [s for p in posts for d in self.screeners for s in await d.detect(p, [])]
        to_judge = self.escalation.select(posts, screened)

        judged, llm_calls = [], 0
        for d in self.escalated:
            before = getattr(d, "calls", 0)
            if hasattr(d, "detect_many"):
                judged += await d.detect_many(to_judge, screened)
            else:
                for p in to_judge:
                    judged += await d.detect(p, [s for s in screened if s.post_id == p.id])
            llm_calls += getattr(d, "calls", 0) - before

        signals = screened + judged
        post_scores = self.aggregator.score_posts(posts, signals)
        creator_signals = [s for d in self.creator_detectors for s in await d.detect(creator, posts, post_scores)]
        return self.aggregator.report(creator.handle, post_scores, creator_signals, signals, llm_calls)


def default_pipeline(escalation: EscalationPolicy | None = None, llm=None) -> SafetyPipeline:
    """Build once, outside the event loop (e.g. at import): construction reads the YAML config."""
    preload()
    return SafetyPipeline(
        screeners=[KeywordDetector(), DisclosureDetector()],
        escalated=[LLMClassifier(llm=llm)],
        creator_detectors=[CreatorRulesDetector()],
        aggregator=PrecedenceAggregator(),
        escalation=escalation or EscalationPolicy(),
    )
