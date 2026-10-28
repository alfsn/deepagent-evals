"""Safety eval: measure each layer (keyword, LLM, production cascade) against hand labels.

    uv run python run_safety_evals.py
    JUDGE_MODEL=deepseek-v4.1-flash uv run python run_safety_evals.py

The LLM runs once on EVERY post; the cascade is then simulated by dropping LLM signals for posts the
production EscalationPolicy wouldn't have sent. Same LLM outputs for every layer = a fair comparison.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from journal import JOURNAL, log_run
from llm import JUDGE_MODEL
from safety.escalation import EscalationPolicy
from safety.models import Severity
from safety.pipeline import default_pipeline
from safety.taxonomy import post_dimensions
from tools import BY_HANDLE, safety_inputs

LABELS = json.loads((Path(__file__).parent / "evals" / "safety_dataset.json").read_text())


def prf(pred: set, gold: set) -> tuple[float, float, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(gold) if gold else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def cohen_kappa(pred: set, gold: set, universe: set) -> float:
    n = len(universe)
    agree = sum((u in pred) == (u in gold) for u in universe) / n
    p_yes = (len(pred) / n) * (len(gold) / n)
    p_no = (1 - len(pred) / n) * (1 - len(gold) / n)
    return (agree - p_yes - p_no) / (1 - p_yes - p_no) if p_yes + p_no < 1 else 1.0


async def evaluate():
    production = EscalationPolicy()
    pipe = default_pipeline(EscalationPolicy(all_posts=True))
    llm = pipe.escalated[0]
    dims = list(post_dimensions())
    gold = {(pid, d) for pid, ds in LABELS["posts"].items() for d in ds}
    universe = {(pid, d) for pid in LABELS["posts"] for d in dims}
    layers = {"keyword": set(), "llm": set(), "cascade": set()}
    errors, cascade_calls, creators = 0, 0, {}

    for handle in BY_HANDLE:
        creator, posts = safety_inputs(handle)
        screened = [s for p in posts for d in pipe.screeners for s in await d.detect(p, [])]
        judged = await llm.detect_many(posts, screened)
        errors += sum(1 for s in judged if s.error)

        layers["keyword"] |= {(s.post_id, s.dimension) for s in screened}
        layers["llm"] |= {(s.post_id, s.dimension) for s in judged if s.error or s.severity > Severity.NONE}

        selected = {p.id for p in production.select(posts, screened)}
        cascade_calls += len(selected)
        signals = screened + [s for s in judged if s.post_id in selected]
        scores = pipe.aggregator.score_posts(posts, signals)
        layers["cascade"] |= {(pid, d) for d, sc in scores.items() for pid in sc.post_ids}
        creator_signals = [s for d in pipe.creator_detectors for s in await d.detect(creator, posts, scores)]
        report = pipe.aggregator.report(creator.handle, scores, creator_signals, signals, len(selected))
        creators[creator.handle] = report

    n_posts = len(LABELS["posts"])
    print(f"\n{'layer':<9} {'prec':>5} {'recall':>6} {'f1':>5} {'kappa':>6}  llm calls")
    summary = {}
    for name, pred in layers.items():
        p, r, f = prf(pred, gold)
        k = cohen_kappa(pred, gold, universe)
        calls = {"keyword": 0, "llm": n_posts, "cascade": cascade_calls}[name]
        summary[name] = {"precision": round(p, 2), "recall": round(r, 2), "f1": round(f, 2), "kappa": round(k, 2), "llm_calls": calls}
        print(f"{name:<9} {p:>5.2f} {r:>6.2f} {f:>5.2f} {k:>6.2f}  {calls}/{n_posts}")
    if errors:
        print(f"(llm errors: {errors} signals, counted as flagged)")

    print(f"\n{'dimension (recall)':<26} " + " ".join(f"{n:>8}" for n in layers))
    for d in dims:
        g = {u for u in gold if u[1] == d}
        if g:
            print(f"{d:<26} " + " ".join(f"{len(layers[n] & g)}/{len(g):<6}" for n in layers))

    print("\ntraps (should NOT be flagged):")
    for pid in LABELS["traps"]["ids"]:
        print(f"  {pid:<9} " + "  ".join(f"{n}={'FLAG' if any(u[0] == pid for u in layers[n]) else 'ok'}" for n in layers))
    print("keyword-free unsafe posts (should be flagged):")
    for pid in LABELS["keyword_free"]["ids"]:
        print(f"  {pid:<9} " + "  ".join(f"{n}={'caught' if any(u[0] == pid for u in layers[n]) else 'MISSED'}" for n in layers))

    print("\ncreator verdicts (cascade):")
    creator_pass = 0
    for handle, report in creators.items():
        ok = report.verdict in LABELS["creators"][handle]
        creator_pass += ok
        flagged = ", ".join(f"{d}={s.verdict}" for d, s in report.dimensions.items())
        print(f"  {'PASS' if ok else 'FAIL'} {handle:<18} {report.verdict:<7} expected {'/'.join(LABELS['creators'][handle]):<14} {flagged}")
    print(f"creator verdicts {creator_pass}/{len(creators)}")

    out = JOURNAL.parent / f"safety_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "layers": summary,
        "false_positives": {n: sorted(map(list, pred - gold)) for n, pred in layers.items()},
        "false_negatives": {n: sorted(map(list, gold - pred)) for n, pred in layers.items()},
        "creators": {h: r.model_dump(mode="json") for h, r in creators.items()},
    }, indent=2))
    log_run({"kind": "safety_eval", "judge_model": JUDGE_MODEL, "creator_pass": f"{creator_pass}/{len(creators)}",
             "layers": summary, "details": str(out)})
    print(f"details -> {out}")


if __name__ == "__main__":
    asyncio.run(evaluate())
