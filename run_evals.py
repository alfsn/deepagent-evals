"""Eval harness: run the agent on every case, apply deterministic checks + LLM judges, journal, reflect.

    uv run python run_evals.py                 # all cases, then reflect
    uv run python run_evals.py --limit 2 --no-reflect
"""

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from agent import build_agent, run_agent
from journal import JOURNAL, log_run
from judge import grade_all
from llm import AGENT_MODEL, JUDGE_MODEL
from reflect import reflect
from tools import KNOWN_HANDLES

DATASET = json.loads((Path(__file__).parent / "evals" / "dataset.json").read_text())


def deterministic_checks(case: dict, run: dict) -> dict:
    """Cheap, reliable checks first. Use the LLM judge only for what code can't verify."""
    mentioned = {h.lower().rstrip(".") for h in re.findall(r"@[\w.]+", run["answer"])}
    return {
        "used_expected_tools": set(case["expected_tools"]) <= set(run["tool_calls"]),
        "mentions_required": all(h.lower() in mentioned for h in case["must_mention"]),
        "no_hallucinated_handles": mentioned <= KNOWN_HANDLES,
    }


async def evaluate(limit: int | None, do_reflect: bool):
    agent = await build_agent()
    results = []
    for case in DATASET[:limit]:
        run = await run_agent(agent, case["input"])
        checks = deterministic_checks(case, run)
        judge = await grade_all(case, run)
        passed = all(checks.values()) and all(g["passed"] for g in judge)
        results.append({"id": case["id"], "input": case["input"], "passed": passed, "checks": checks, "judge": judge, **run})
        scores = " ".join(f"{g['criterion'][:5]}={g['score']}" for g in judge)
        fails = [k for k, ok in checks.items() if not ok]
        print(f"{'PASS' if passed else 'FAIL'}  {case['id']:<20} {scores}  {run['latency_s']:>5}s  {'checks failed: ' + ', '.join(fails) if fails else ''}")

    n_pass = sum(r["passed"] for r in results)
    by_crit = {c: [g["score"] for r in results for g in r["judge"] if g["criterion"] == c and g["score"]] for c in ("groundedness", "correctness", "helpfulness")}
    print(f"\npass rate {n_pass}/{len(results)}  " + "  ".join(f"{c}={sum(s) / len(s):.2f}" for c, s in by_crit.items() if s))

    out = JOURNAL.parent / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log_run({"kind": "eval", "agent_model": AGENT_MODEL, "judge_model": JUDGE_MODEL, "pass_rate": f"{n_pass}/{len(results)}", "details": str(out)})
    print(f"details -> {out}")

    if do_reflect:
        lessons = reflect(results)
        print("\nnew lessons (loaded into the agent's memory next run):" if lessons else "\nno new lessons")
        for lesson in lessons:
            print(" -", lesson)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--no-reflect", action="store_true")
    a = p.parse_args()
    asyncio.run(evaluate(a.limit, not a.no_reflect))
