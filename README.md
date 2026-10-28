# deepagent-evals

End-to-end reference architecture for building, evaluating, and self-improving a **Deep Agent** (`deepagents` + LangGraph + MCP + skills + subagents) with a **deterministic + LLM-as-a-judge evaluation suite** and a **reflection/memory loop**.

The example domain is **influencer vetting for brand campaigns**: searching creator catalogs, checking brand exclusion rules via an MCP server, delegating fake-follower investigations to a subagent, grading responses against human expectations, and distilling eval failures into persistent agent memory.

## Architecture

| File | Role |
|---|---|
| [`llm.py`](llm.py) | Ollama Cloud model factory (`base_url=https://ollama.com` + Bearer auth) |
| [`tools.py`](tools.py) + [`data/`](data/) | Local `@tool` functions over a mock creator dataset |
| [`mcp_server.py`](mcp_server.py) | FastMCP `stdio` server exposing brand guidelines, loaded via `MultiServerMCPClient` |
| [`agent.py`](agent.py) | `create_deep_agent` combining local tools, MCP tools, skills, memory, and a `fraud-checker` subagent |
| [`workspace/skills/`](workspace/skills/) | Progressive-disclosure skills (`SKILL.md` descriptions in prompt; full body read on demand) |
| [`evals/dataset.json`](evals/dataset.json) | Evaluation cases covering multi-step vetting, exclusions, empty results, and prompt injection |
| [`judge.py`](judge.py) | Single-criterion 1–5 rubric judges (`reasoning` before `score`) + position-swapped pairwise A/B judge |
| [`run_evals.py`](run_evals.py) | Evaluation harness: deterministic checks → parallel rubric judges → JSONL journal → reflection |
| [`journal.py`](journal.py) | Append-only JSONL log of single runs and evaluation suites |
| [`reflect.py`](reflect.py) | Extracts general lessons from failed eval cases and appends them to [`workspace/memory/AGENTS.md`](workspace/memory/AGENTS.md) |
| [`safety/`](safety/) | Content-safety pipeline: keyword + disclosure screening → LLM classifier on escalated posts → creator rules → brand-independent `SafetyReport`; `suitability()` applies per-brand tolerance |
| [`run_safety_evals.py`](run_safety_evals.py) + [`evals/safety_dataset.json`](evals/safety_dataset.json) | Per-layer precision / recall / kappa against hand labels, plus creator verdicts |
| [`docs/safety-embeddings.md`](docs/safety-embeddings.md) | Design for the (not yet built) embedding layer |

## Quickstart

```bash
cp .env.example .env   # set OLLAMA_API_KEY
uv sync
```

```bash
# Verify API key, candidate models, tool calling, and structured output
uv run python smoke_test.py

# Run the agent on a single query
uv run python agent.py "Shortlist fitness influencers for Gymshark"

# Run the full eval suite (agent -> deterministic checks -> LLM judges -> journal -> reflection)
uv run python run_evals.py
uv run python run_evals.py --limit 2 --no-reflect

# Content-safety eval (keyword vs LLM vs cascade) and deterministic unit tests
uv run python run_safety_evals.py
uv run pytest tests -q

# Swap agent and judge models via environment variables
AGENT_MODEL=kimi-k3 JUDGE_MODEL=gemma4:31b uv run python run_evals.py

# Inspect the MCP server or step through the agent graph visually
uv run mcp dev mcp_server.py
uv run langgraph dev
```

## Content Safety

Three layers, each a pluggable detector emitting the same `Signal` type ([`safety/models.py`](safety/models.py)):

1. **Keywords** ([`safety/detectors/keywords.py`](safety/detectors/keywords.py)): NFKC + casefold + leetspeak normalization, word-boundary regex, per-language YAML lexicons with an allowlist for known traps. Its signals are *candidates* only.
2. **LLM judgement**: [`llm_classifier.py`](safety/detectors/llm_classifier.py) makes one structured call per escalated post (keyword hit or a seeded 30% sample), and its output is authoritative over keywords. The `safety-reviewer` subagent ([`agent.py`](agent.py)) then investigates `review` items with `get_safety_report`, `get_post_excerpt` and `judge_post`.
3. **Embeddings**: designed only; see [`docs/safety-embeddings.md`](docs/safety-embeddings.md).

Dimensions live in [`safety/config/taxonomy.yaml`](safety/config/taxonomy.yaml): GARM-mapped post dimensions, `undisclosed_sponsorship`, and creator-level `minor_audience_exposure`, `risky_partnerships` and `disclosure_rate`. Aggregation rules ([`safety/aggregate.py`](safety/aggregate.py)):
- LLM decisions override keyword candidates.
- An unconfirmed candidate becomes `review`.
- **Errors are never `safe`.**
- A confirmed floor dimension makes the creator `unsafe` regardless of brand.

| Layer (gemma4:31b, 41 posts) | Precision | Recall | Kappa | LLM calls |
|---|---|---|---|---|
| keyword | 0.82 | 0.74 | 0.77 | 0 |
| LLM on every post | 0.86 | 0.95 | 0.89 | 41 |
| cascade | 0.94 | 0.84 | 0.88 | 25 |

Keywords trip on traps ("serum is the bomb") and miss euphemisms ("a little green 🌿"). The cascade's recall gap (keyword-free posts that aren't sampled) is what the embedding layer is designed to close. *Caveat: one person wrote both the posts and the labels, and the thresholds were tuned on this same set.*

## Evaluation Design

1. **Deterministic checks first** ([`run_evals.py`](run_evals.py)): Fast, zero-cost assertions verify that expected tools were called, required handles were mentioned, and **no hallucinated handles** (outside `KNOWN_HANDLES`) appear in the final response.
2. **Rubric LLM judges** ([`judge.py`](judge.py)): Grades `groundedness`, `correctness`, and `helpfulness` in separate calls with explicit 1–5 anchors, `temperature=0`, and structured output (`method="function_calling"`) where `reasoning` precedes `score`.
3. **Position-swapped pairwise comparison** ([`judge.py`](judge.py)): Runs A/B comparisons in both `(A, B)` and `(B, A)` orders, declaring a tie if the winner flips due to position bias.
4. **Cross-family judging**: Supports setting `AGENT_MODEL` and `JUDGE_MODEL` independently to avoid self-preference bias:

| Setup | Pass Rate | Groundedness | Correctness | Latency / Case |
|---|---|---|---|---|
| `gemma4:31b` agent / `gemma4:31b` judge | 6/6 | 5.00 | 4.33 | ~1–7s |
| `kimi-k3` agent / `deepseek-v4.1-flash` judge | 2/6 | 3.50 | 4.83 | ~7–37s |

*Note: `kimi-k3` scores higher on correctness but occasionally states inferences as facts; a cross-family judge catches ungrounded claims that a same-family judge lets through.*

## Reflection & Safety

- **Closed-loop reflection** ([`reflect.py`](reflect.py)): Failed checks and judge critiques are distilled into at most 3 concise imperative rules and appended to [`workspace/memory/AGENTS.md`](workspace/memory/AGENTS.md) (capped at 10 lessons to prevent prompt bloat), which `create_deep_agent` loads on subsequent runs.
- **Sandboxed filesystem**: The agent's `FilesystemBackend` is rooted at `workspace/` with `virtual_mode=True`, preventing access to `.env` or source code outside the workspace.
