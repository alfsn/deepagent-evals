"""Deterministic keyword detector: normalize -> blank out allowlisted phrases -> word-boundary regex.

Emits candidates only. Keywords can't read context ("killer leg day"), so the LLM confirms or dismisses them.
Uses stdlib `re`; swap in pyahocorasick inside this class if lexicons grow to thousands of terms.
"""

import re
import unicodedata
from functools import cached_property
from pathlib import Path

import yaml

from safety.models import Post, Severity, Signal

LEXICONS = Path(__file__).parent.parent / "config" / "lexicons"
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s"})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text)


def _phrase_regex(phrase: str) -> str:
    return r"\s+".join(re.escape(w) for w in normalize(phrase).split())


class KeywordDetector:
    name = "keyword"
    version = "kw-1"

    def __init__(self, lexicon_dir: Path = LEXICONS, base_lang: str = "en"):
        self.lexicon_dir = lexicon_dir
        self.base_lang = base_lang  # always applied: influencer posts mix English into other languages

    @cached_property
    def _compiled(self) -> dict[str, tuple[re.Pattern | None, dict[str, re.Pattern]]]:
        out = {}
        for f in sorted(self.lexicon_dir.glob("*.yaml")):
            lex = yaml.safe_load(f.read_text())
            allow = [_phrase_regex(p) for p in lex.get("allowlist") or []]
            allow_re = re.compile(r"(?<!\w)(?:" + "|".join(allow) + r")(?!\w)") if allow else None
            dims = {}
            for dim, spec in (lex.get("dimensions") or {}).items():
                alts = [_phrase_regex(t) for t in spec.get("terms", [])] + spec.get("patterns", [])
                dims[dim] = re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)")
            out[f.stem] = (allow_re, dims)
        return out

    def scan(self, text: str, lang: str = "en") -> dict[str, list[str]]:
        """dimension -> matched terms. Pure and synchronous, so it's easy to unit test."""
        hits: dict[str, set[str]] = {}
        for code in dict.fromkeys([self.base_lang, lang]):
            if code not in self._compiled:
                continue
            allow_re, dims = self._compiled[code]
            plain = normalize(text)
            for variant in dict.fromkeys([plain, plain.translate(LEET)]):
                if allow_re:
                    variant = allow_re.sub(" ", variant)
                for dim, rx in dims.items():
                    hits.setdefault(dim, set()).update(m.group(0) for m in rx.finditer(variant))
        return {d: sorted(h) for d, h in hits.items() if h}

    async def detect(self, post: Post, prior: list[Signal]) -> list[Signal]:
        return [
            Signal(dimension=dim, source="keyword", detector_version=self.version, kind="candidate",
                   severity=Severity.LOW, evidence=terms, post_id=post.id)
            for dim, terms in self.scan(post.text, post.lang).items()
        ]
