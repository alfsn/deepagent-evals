"""End-of-run reflection: turn eval failures into lessons the agent loads (via memory) on its next run."""

from pathlib import Path

from pydantic import BaseModel, Field

from llm import AGENT_MODEL, get_llm

MEMORY = Path(__file__).parent / "workspace" / "memory" / "AGENTS.md"
MARKER = "<!-- reflect.py appends lessons below this line -->"
MAX_LESSONS = 10


class Lessons(BaseModel):
    lessons: list[str] = Field(max_length=3, description="Short, general, imperative rules (not case-specific trivia)")


REFLECT_PROMPT = """You are reviewing an AI agent's evaluation results. Write at most 3 lessons that would
have prevented the failures. Each lesson is one imperative sentence the agent can follow next time.
If nothing failed, return an empty list.

EXISTING LESSONS (don't repeat these):
{existing}

FAILURES:
{failures}"""


def current_lessons() -> list[str]:
    text = MEMORY.read_text()
    return [line[2:] for line in text.split(MARKER, 1)[1].splitlines() if line.startswith("- ")]


def reflect(results: list[dict]) -> list[str]:
    failures = []
    for r in results:
        bad = [f"{g['criterion']} ({g['score']}): {g['reasoning'][:400]}" for g in r["judge"] if not g["passed"]]
        bad += [f"check failed: {name}" for name, ok in r["checks"].items() if not ok]
        if bad:
            failures.append(f"Case {r['id']} - {r['input']}\n  " + "\n  ".join(bad))
    if not failures:
        return []

    existing = current_lessons()
    out = get_llm(AGENT_MODEL).with_structured_output(Lessons, method="function_calling").invoke(
        REFLECT_PROMPT.format(existing="\n".join(existing) or "(none)", failures="\n\n".join(failures))
    )
    new = [lesson for lesson in (out.lessons if out else []) if lesson not in existing]
    kept = (existing + new)[-MAX_LESSONS:]
    head = MEMORY.read_text().split(MARKER, 1)[0]
    MEMORY.write_text(head + MARKER + "\n" + "".join(f"- {lesson}\n" for lesson in kept))
    return new
