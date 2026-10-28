"""Loads the dimension taxonomy (what we measure) from YAML."""

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from safety.models import Severity

CONFIG = Path(__file__).parent / "config"


class DimensionSpec(BaseModel):
    id: str
    level: str  # "post" | "creator"
    garm: str | None = None
    floor: bool = False
    unsafe_at: Severity | None = Severity.MEDIUM  # None ("never"): this dimension alone never makes a post unsafe
    review_at: Severity = Severity.LOW  # confirmed severity below this is cleared (not worth a reviewer's time)
    description: str


@cache
def load_taxonomy() -> dict[str, DimensionSpec]:
    raw = yaml.safe_load((CONFIG / "taxonomy.yaml").read_text())
    def sev(name: str) -> Severity | None:
        return None if name == "never" else Severity[name.upper()]

    return {
        dim_id: DimensionSpec(id=dim_id, level=level, **{
            **spec, "unsafe_at": sev(spec["unsafe_at"]), "review_at": sev(spec.get("review_at", "low"))})
        for level, dims in raw.items()
        for dim_id, spec in dims.items()
    }


def post_dimensions() -> dict[str, DimensionSpec]:
    return {k: v for k, v in load_taxonomy().items() if v.level == "post"}


@cache
def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIG / name).read_text())
