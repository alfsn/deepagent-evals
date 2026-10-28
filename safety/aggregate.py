"""Evidence -> verdicts. The core rules, in order:

1. Per (post, dimension), decisions (LLM, rules) are authoritative; keyword candidates never override them.
2. Candidates with no decision -> "review" (unconfirmed, needs a look).
3. An error -> at least "review", never "safe".
4. Decision severity >= the dimension's unsafe_at -> "unsafe"; >= review_at -> "review"; lower -> cleared.
5. Unsafe evidence older than stale_after_days only -> downgraded to "review" (people change).
"""

from collections import defaultdict
from typing import Protocol

from safety.models import DimensionScore, Post, SafetyReport, Severity, Signal, Verdict
from safety.taxonomy import DimensionSpec, load_taxonomy

RANK = {"safe": 0, "review": 1, "unsafe": 2}


class Aggregator(Protocol):
    def score_posts(self, posts: list[Post], signals: list[Signal]) -> dict[str, DimensionScore]: ...
    def report(self, handle: str, post_scores: dict[str, DimensionScore], creator_signals: list[Signal],
               signals: list[Signal], llm_calls: int) -> SafetyReport: ...


def _judge(spec: DimensionSpec, sigs: list[Signal]) -> tuple[Verdict, Severity, bool, str]:
    """Verdict for one (post, dimension) or one creator dimension, from that unit's signals."""
    decisions = [s for s in sigs if s.kind == "decision" and not s.error]
    errors = [s for s in sigs if s.error]
    candidates = [s for s in sigs if s.kind == "candidate"]
    if decisions:
        sev = max(s.severity for s in decisions)
        unsafe = spec.unsafe_at is not None and sev >= spec.unsafe_at
        verdict: Verdict = "unsafe" if unsafe else "review" if sev >= max(spec.review_at, Severity.LOW) else "safe"
        why = "; ".join(filter(None, (s.reasoning or ", ".join(s.evidence) for s in decisions if s.severity)))
        if verdict == "safe" and errors:
            verdict, why = "review", errors[0].error
        return verdict, sev, True, why or "cleared"
    if errors:
        return "review", Severity.NONE, False, errors[0].error
    if candidates:
        return "review", max(s.severity for s in candidates), False, "unconfirmed: " + ", ".join(
            e for s in candidates for e in s.evidence)
    return "safe", Severity.NONE, False, ""


class PrecedenceAggregator:
    def __init__(self, stale_after_days: int = 365):
        self.stale_after_days = stale_after_days
        self.taxonomy = load_taxonomy()

    def score_posts(self, posts: list[Post], signals: list[Signal]) -> dict[str, DimensionScore]:
        age = {p.id: p.days_ago for p in posts}
        by_unit: dict[tuple[str, str], list[Signal]] = defaultdict(list)
        for s in signals:
            by_unit[(s.post_id, s.dimension)].append(s)

        per_dim: dict[str, list[tuple[str, Verdict, Severity, bool, str]]] = defaultdict(list)
        for (post_id, dim), sigs in by_unit.items():
            verdict, sev, confirmed, why = _judge(self.taxonomy[dim], sigs)
            if verdict != "safe":
                per_dim[dim].append((post_id, verdict, sev, confirmed, why))

        scores = {}
        for dim, hits in per_dim.items():
            worst = max(RANK[v] for _, v, *_ in hits)
            verdict: Verdict = ["safe", "review", "unsafe"][worst]
            unsafe_ids = [pid for pid, v, *_ in hits if v == "unsafe"]
            if unsafe_ids and all(age.get(pid, 0) > self.stale_after_days for pid in unsafe_ids):
                verdict = "review"
            scores[dim] = DimensionScore(
                dimension=dim, verdict=verdict,
                severity=max(sev for *_, sev, _, _ in hits),
                confirmed=any(c for *_, c, _ in hits),
                post_ids=[pid for pid, *_ in hits],
                why=f"{len(hits)}/{len(posts)} posts. " + " | ".join(f"{pid}: {why}" for pid, *_, why in hits),
            )
        return scores

    def report(self, handle: str, post_scores: dict[str, DimensionScore], creator_signals: list[Signal],
               signals: list[Signal], llm_calls: int) -> SafetyReport:
        dims = dict(post_scores)
        by_dim: dict[str, list[Signal]] = defaultdict(list)
        for s in creator_signals:
            by_dim[s.dimension].append(s)
        for dim, sigs in by_dim.items():
            verdict, sev, confirmed, why = _judge(self.taxonomy[dim], sigs)
            if verdict != "safe":
                dims[dim] = DimensionScore(dimension=dim, verdict=verdict, severity=sev, confirmed=confirmed, why=why)
        overall: Verdict = ["safe", "review", "unsafe"][max((RANK[d.verdict] for d in dims.values()), default=0)]
        return SafetyReport(handle=handle, verdict=overall, dimensions=dims,
                            signals=signals + creator_signals, llm_calls=llm_calls)
