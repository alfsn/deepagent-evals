# Safety layer 3: embeddings (designed, not built)

Status: **not implemented.** This documents how embeddings plug into `safety/` without changing the pipeline, aggregator or agent.

## Why
The safety eval (`run_safety_evals.py`) shows the gap embeddings fill:

| Layer | Strength | Gap |
|---|---|---|
| Keywords | free, auditable, multilingual via lexicons | misses euphemisms and coded language (lena-4 "a little green 🌿", jules-3 "farine ❄️") |
| LLM classifier | understands context, clears traps ("killer leg day") | ~1 call per post; the cascade only sends keyword hits plus a 30% sample, so keyword-free posts are often never judged |
| **Embeddings** | **cheap like keywords, semantic like the LLM** | needs labeled exemplars; less explainable |

Keyword-free misses at `sample_rate=0.3` in the current eval: lena-4, jules-3, jules-6. An embedding screener would route these to the LLM without sending it every post.

## Post embeddings → `EmbeddingDetector(PostDetector)`
- **Model**: LangChain `Embeddings` interface (same idea as `llm.get_llm`: one factory, swappable). Start with `OllamaEmbeddings(model="embeddinggemma")` (multilingual, matches the existing Ollama setup). Later, **SigLIP/CLIP** for images and video thumbnails, since most influencer risk is visual.
- **Exemplar kNN**: per dimension, keep a set of labeled unsafe exemplars (seeded from LLM-confirmed + human-verified posts). Score = max cosine similarity to that dimension's exemplars. Above threshold → `Signal(source="embedding", kind="candidate")`.
- **Linear probes**: once there are a few hundred labels per dimension, train `sklearn.linear_model.LogisticRegression` on the embeddings (one per dimension). Calibrated probabilities → candidate severity.
- **Where it runs**: in `SafetyPipeline.screeners`, next to `KeywordDetector`. `EscalationPolicy.select` already escalates any post with a screener signal, so **no pipeline change is needed**. Embedding candidates are never authoritative: the precedence rules in `aggregate.py` already make the LLM decide.

```python
pipeline = SafetyPipeline(
    screeners=[KeywordDetector(), DisclosureDetector(), EmbeddingDetector(store, embedder, thresholds)],
    escalated=[LLMClassifier()], ...)
```

## Creator embeddings → `EmbeddingCreatorDetector(CreatorDetector)`
- **Vector** = recency-weighted mean of the creator's post embeddings (+ bio embedding). Half-life of about 90 days, so old content fades, mirroring `stale_after_days` in the aggregator.
- **Uses**:
  - *Similarity to known-unsafe creators*: kNN over creator vectors with labels → `creator_risk_similarity` dimension (add it to `taxonomy.yaml` under `creator:`).
  - *Drift*: cosine distance between this month's vector and the trailing 6-month vector, projected onto the "unsafe centroid" direction. Alert on movement toward it.
  - *Cold start / discovery*: screen creators before ingesting all their posts; find safe look-alikes of a brand's best partners (a positive use of the same index).

## Storage → `EmbeddingStore` protocol
```python
class EmbeddingStore(Protocol):
    def upsert(self, kind: Literal["post", "creator"], id: str, vector: list[float], metadata: dict) -> None: ...
    def query(self, kind: str, vector: list[float], k: int, filter: dict | None = None) -> list[tuple[str, float, dict]]: ...
```
- Local/dev: **LanceDB** (embedded, file-based, no server). The LangGraph store in `.langgraph_api/` also has a vector index for agent memory, but keep safety vectors separate.
- Production: **pgvector** (if Postgres is already the system of record) or **Qdrant**.
- Store `model` + `version` in metadata. Vectors from different models aren't comparable, so re-embed when the model changes.

## The label loop (and its risk)
LLM-confirmed signals (`source="llm"`, `kind="decision"`) are free training labels for the exemplars and probes, so the expensive judge gradually trains a cheap detector. **The risk:** the judge's biases get baked into the probe, which then decides what the judge sees. Safeguards:
- train only on LLM labels a human has spot-checked, or with high LLM–human agreement per dimension (kappa from `run_safety_evals.py`),
- keep `evals/safety_dataset.json` (human labels) **held out**, never in the exemplar set,
- keep the random LLM sample in `EscalationPolicy` so recall stays measurable even after embeddings take over screening.

## Evaluation
Add an `embedding` row to `run_safety_evals.py` (precision/recall/kappa per dimension) and an `embedding+llm` cascade row. Success criterion: cascade recall closes most of the gap to LLM-on-every-post at a similar number of LLM calls.

## Implementation checklist (when we build it)
1. `safety/embeddings.py`: embedder factory + `EmbeddingStore` + LanceDB implementation.
2. `safety/detectors/embedding.py`: `EmbeddingDetector`, `EmbeddingCreatorDetector`.
3. `safety/config/embeddings.yaml`: model, per-dimension thresholds, half-life.
4. Seed script: embed `data/posts.json`, load the exemplars.
5. Add both detectors to `default_pipeline()`; add the eval rows.
