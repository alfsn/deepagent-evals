"""Append-only JSONL journal of every run, so runs and evals persist across sessions."""

import json
from datetime import datetime, timezone
from pathlib import Path

JOURNAL = Path(__file__).parent / "journal" / "runs.jsonl"


def log_run(record: dict) -> None:
    JOURNAL.parent.mkdir(exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with JOURNAL.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def recent_runs(n: int = 20) -> list[dict]:
    if not JOURNAL.exists():
        return []
    return [json.loads(line) for line in JOURNAL.read_text().splitlines()[-n:]]
