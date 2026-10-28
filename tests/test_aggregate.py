import asyncio

from safety.aggregate import PrecedenceAggregator
from safety.detectors.llm_classifier import LLMClassifier
from safety.escalation import EscalationPolicy
from safety.models import Creator, Post, Severity, Signal
from safety.pipeline import default_pipeline
from safety.suitability import suitability

agg = PrecedenceAggregator()
POSTS = [Post(id="p1", handle="@x", text="t"), Post(id="old", handle="@x", text="t", days_ago=900)]


def sig(dim, kind, sev=Severity.NONE, source="keyword", post_id="p1", error=None):
    return Signal(dimension=dim, source=source, detector_version="t", kind=kind, severity=sev, post_id=post_id, error=error)


def test_llm_decision_overrides_keyword_candidate():
    scores = agg.score_posts(POSTS, [sig("violence_weapons", "candidate", Severity.LOW),
                                     sig("violence_weapons", "decision", Severity.NONE, "llm")])
    assert "violence_weapons" not in scores  # cleared -> safe


def test_candidate_without_decision_is_review():
    s = agg.score_posts(POSTS, [sig("gambling", "candidate", Severity.LOW)])["gambling"]
    assert s.verdict == "review" and not s.confirmed


def test_error_is_never_safe():
    s = agg.score_posts(POSTS, [sig("misinformation", "decision", source="llm", error="timeout")])["misinformation"]
    assert s.verdict == "review"


def test_floor_dimension_low_is_unsafe():
    s = agg.score_posts(POSTS, [sig("hate_harassment", "decision", Severity.LOW, "llm")])["hate_harassment"]
    assert s.verdict == "unsafe" and s.confirmed


def test_below_threshold_is_review():
    s = agg.score_posts(POSTS, [sig("drugs_alcohol_tobacco", "decision", Severity.MEDIUM, "llm")])["drugs_alcohol_tobacco"]
    assert s.verdict == "review"  # unsafe_at: high


def test_stale_unsafe_downgraded():
    s = agg.score_posts(POSTS, [sig("misinformation", "decision", Severity.HIGH, "llm", post_id="old")])["misinformation"]
    assert s.verdict == "review"


def test_suitability_floor_cannot_be_tolerated():
    report = agg.report("@x", agg.score_posts(POSTS, [sig("hate_harassment", "decision", Severity.LOW, "llm")]), [], [], 0)
    assert suitability(report, {"hate_harassment": "high"})["fit"] == "reject"


def test_suitability_brand_tolerance():
    report = agg.report("@x", agg.score_posts(POSTS, [sig("drugs_alcohol_tobacco", "decision", Severity.HIGH, "llm")]), [], [], 0)
    assert suitability(report)["fit"] == "reject"
    assert suitability(report, {"drugs_alcohol_tobacco": "high"})["fit"] == "ok"  # e.g. a beer brand


class FakeLLM:
    """Stands in for the model: 'scam' -> misinformation, 'sportsbook' -> gambling (both high), else clean."""

    def __init__(self, schema):
        self.schema = schema

    async def ainvoke(self, prompt):
        post = prompt.split("<<<POST", 1)[1]
        flagged = {d for word, d in (("scam", "misinformation"), ("sportsbook", "gambling")) if word in post}
        dims = [d for d in self.schema.model_fields if d not in ("analysis", "findings")]
        return self.schema(analysis="a", **{d: "high" if d in flagged else "none" for d in dims})


def run_pipeline(posts, creator=None, policy=None):
    pipe = default_pipeline(policy or EscalationPolicy(sample_rate=0))
    llm = pipe.escalated[0]
    assert isinstance(llm, LLMClassifier)
    llm._llm = FakeLLM(llm.schema)
    creator = creator or Creator(handle="@x", name="X", niche="beauty", audience={})
    return asyncio.run(pipe.run(creator, posts))


def test_pipeline_trap_cleared_and_only_flagged_posts_escalated():
    posts = [Post(id="a", handle="@x", text="killer bomb of a serum"), Post(id="b", handle="@x", text="morning routine")]
    report = run_pipeline(posts)
    assert report.verdict == "safe" and report.llm_calls == 1


def test_pipeline_keyword_free_caught_only_when_sampled():
    posts = [Post(id="a", handle="@x", text="total scam product, buy now")]
    assert run_pipeline(posts).verdict == "safe"  # not screened, not sampled -> missed (the recall gap)
    assert run_pipeline(posts, policy=EscalationPolicy(all_posts=True)).verdict == "unsafe"


def test_pipeline_creator_rules():
    creator = Creator(handle="@x", name="X", niche="fitness", audience={"under_18_pct": 40}, past_brands=["Unknown Supplements Co"])
    posts = [Post(id="a", handle="@x", text="sportsbook promo, bet now")]
    report = run_pipeline(posts, creator)
    assert report.dimensions["risky_partnerships"].verdict == "unsafe"
    assert report.dimensions["gambling"].verdict == "unsafe"
    assert report.dimensions["minor_audience_exposure"].verdict == "unsafe"  # confirmed gambling + 40% under 18


def test_pipeline_llm_failure_is_review_not_safe():
    pipe = default_pipeline(EscalationPolicy(all_posts=True))

    class Broken:
        async def ainvoke(self, prompt):
            raise TimeoutError("model down")

    pipe.escalated[0]._llm = Broken()
    report = asyncio.run(pipe.run(Creator(handle="@x", name="X", niche="food", audience={}),
                                  [Post(id="a", handle="@x", text="dal recipe")]))
    assert report.verdict == "review"


def test_unsafe_never_caps_at_review():
    s = agg.score_posts(POSTS, [sig("undisclosed_sponsorship", "decision", Severity.HIGH, "llm")])["undisclosed_sponsorship"]
    assert s.verdict == "review"


def test_below_review_at_is_cleared():
    assert "profanity" not in agg.score_posts(POSTS, [sig("profanity", "decision", Severity.LOW, "llm")])
