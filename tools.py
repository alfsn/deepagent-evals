"""Local LangChain tools over the mock influencer dataset."""

import json
from pathlib import Path

from langchain_core.tools import tool

DATA = json.loads((Path(__file__).parent / "data" / "influencers.json").read_text())
BY_HANDLE = {i["handle"].lower(): i for i in DATA}
KNOWN_HANDLES = set(BY_HANDLE)


@tool
def search_influencers(niche: str, min_followers: int = 0, platform: str | None = None) -> list[dict]:
    """Search influencers by niche (beauty, fitness, food), minimum follower count and optional platform.

    Returns a short summary for each match. Use get_influencer_metrics for details.
    """
    hits = [
        i for i in DATA
        if i["niche"] == niche.lower()
        and i["followers"] >= min_followers
        and (platform is None or i["platform"] == platform.lower())
    ]
    return [{k: i[k] for k in ("handle", "name", "platform", "followers")} for i in hits]


@tool
def get_influencer_metrics(handle: str) -> dict:
    """Get engagement rate, audience demographics, 30-day follower growth and past brand deals for one handle."""
    inf = BY_HANDLE.get(handle.lower() if handle.startswith("@") else f"@{handle.lower()}")
    if inf is None:
        return {"error": f"unknown handle {handle}"}
    return inf


LOCAL_TOOLS = [search_influencers, get_influencer_metrics]
