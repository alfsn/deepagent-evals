"""LLM-as-judge: one criterion per call, rubric-based, reasoning before score, structured output."""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from llm import JUDGE_MODEL, get_llm


class Verdict(BaseModel):
    # Field order matters: the model writes its reasoning before committing to a score.
    reasoning: str = Field(description="Step-by-step justification citing the evidence")
    score: int = Field(ge=1, le=5, description="1 = fails the criterion, 5 = fully meets it")


RUBRICS = {
    "groundedness": """Is every influencer, number and claim in the ANSWER supported by the TOOL OUTPUTS?
5: everything is traceable to the tool outputs. 3: minor unsupported details. 1: invented influencers or numbers.""",
    "correctness": """Does the ANSWER satisfy the EXPECTATIONS (the reference written by a human expert)?
5: meets every expectation. 3: meets most, with one notable miss. 1: misses the main point or recommends something it should reject.""",
    "helpfulness": """Would a brand manager be able to act on this ANSWER directly?
5: clear ranked recommendation with reasons and key numbers. 3: usable but vague or poorly organised. 1: unusable.""",
}

JUDGE_PROMPT = """You are a strict evaluator of an influencer-marketing AI assistant.
Grade ONLY this criterion:

{rubric}

USER REQUEST:
{input}

EXPECTATIONS:
{expectations}

TOOL OUTPUTS (the agent's evidence):
{evidence}

ANSWER:
{answer}

Don't reward length. Quote the evidence in your reasoning."""

PASS_THRESHOLD = 4

_judge = None


def judge_llm():
    global _judge
    if _judge is None:
        _judge = get_llm(JUDGE_MODEL).with_structured_output(Verdict, method="function_calling")
    return _judge


async def grade(criterion: str, case: dict, run: dict, retries: int = 2) -> dict:
    prompt = JUDGE_PROMPT.format(
        rubric=RUBRICS[criterion],
        input=case["input"],
        expectations=case["expectations"],
        evidence="\n".join(run["tool_outputs"])[:12000] or "(no tool calls)",
        answer=run["answer"],
    )
    for attempt in range(retries + 1):
        try:
            v = await judge_llm().ainvoke(prompt)
            if v is not None:
                return {"criterion": criterion, "score": v.score, "passed": v.score >= PASS_THRESHOLD, "reasoning": v.reasoning}
        except Exception as e:
            err = e
    return {"criterion": criterion, "score": None, "passed": False, "reasoning": f"judge error: {err!r}"}


async def grade_all(case: dict, run: dict) -> list[dict]:
    return await asyncio.gather(*(grade(c, case, run) for c in RUBRICS))


# --- Pairwise comparison (A/B two prompts or models), with a position swap to cancel position bias ---

class Preference(BaseModel):
    reasoning: str
    winner: Literal["A", "B", "tie"]


PAIRWISE_PROMPT = """Which answer better serves this request? Judge accuracy and actionability, not length.

REQUEST: {input}

ANSWER A:
{a}

ANSWER B:
{b}"""


async def pairwise(input: str, answer_1: str, answer_2: str) -> str:
    llm = get_llm(JUDGE_MODEL).with_structured_output(Preference, method="function_calling")
    first = await llm.ainvoke(PAIRWISE_PROMPT.format(input=input, a=answer_1, b=answer_2))
    swapped = await llm.ainvoke(PAIRWISE_PROMPT.format(input=input, a=answer_2, b=answer_1))
    unswap = {"A": "B", "B": "A", "tie": "tie"}[swapped.winner]
    if first.winner == unswap:
        return {"A": "answer_1", "B": "answer_2", "tie": "tie"}[first.winner]
    return "tie (inconsistent across orderings)"
