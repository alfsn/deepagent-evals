import asyncio

import pytest

from safety.detectors.disclosure import DisclosureDetector
from safety.detectors.keywords import KeywordDetector
from safety.models import Post

kw = KeywordDetector()


@pytest.mark.parametrize("text", ["Killer leg day today", "grabbed a shot of espresso", "Drugstore haul!", "Scunthorpe United"])
def test_allowlist_and_word_boundaries_avoid_false_positives(text):
    assert kw.scan(text) == {}


@pytest.mark.parametrize("text,dim", [
    ("this tea cures acne", "misinformation"),
    ("get your w33d here", "drugs_alcohol_tobacco"),        # leetspeak
    ("ＦＲＥＥ ＳＰＩＮＳ tonight", "gambling"),                # full-width unicode -> NFKC
    ("No   diet   needed!!", "misinformation"),           # multi-word phrase, odd spacing
    ("fuuuuck", "profanity"),                               # regex pattern
])
def test_hits(text, dim):
    assert dim in kw.scan(text)


def test_language_lexicon_plus_english():
    hits = kw.scan("chá detox, bora apostar? free spins!", "pt")
    assert {"misinformation", "gambling"} <= set(hits)


def test_unknown_language_falls_back_to_english():
    assert "gambling" in kw.scan("casino night", "de")


def test_keyword_signals_are_candidates():
    sigs = asyncio.run(kw.detect(Post(id="p", handle="@x", text="huge jackpot"), []))
    assert sigs and all(s.kind == "candidate" and s.post_id == "p" for s in sigs)


@pytest.mark.parametrize("text,flagged", [
    ("Love this balm, code MAYA20", True),
    ("Love this balm #ad", False),
    ("Gifted by them, honest review", False),
])
def test_disclosure(text, flagged):
    post = Post(id="p", handle="@x", text=text, brand_mentions=["Brand"])
    assert bool(asyncio.run(DisclosureDetector().detect(post, []))) is flagged


def test_no_brand_mention_no_disclosure_signal():
    assert asyncio.run(DisclosureDetector().detect(Post(id="p", handle="@x", text="code X"), [])) == []
