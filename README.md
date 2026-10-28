# Adaptive AI Model Router

## Start the API

From this directory, activate the Python environment with the project
dependencies and run:

```powershell
uvicorn api.app:app --reload
```

The API uses LiteLLM to call the configured models. The default local Ollama
models are Qwen 2.5 0.5B and Llama 3.2 1B; Ollama and those models must be
available for the playground. API documentation is available at
`http://127.0.0.1:8000/docs`.

To select alternate LiteLLM providers, set `ROUTER_SMALL_MODEL` or
`ROUTER_LARGE_MODEL` to a LiteLLM model identifier. Set matching
`ROUTER_SMALL_API_KEY` / `ROUTER_LARGE_API_KEY` and optional
`ROUTER_SMALL_API_BASE` / `ROUTER_LARGE_API_BASE` environment variables.
Keep provider credentials out of source files. The local Ollama URL defaults
to `http://localhost:11434` and can be changed with `OLLAMA_BASE_URL`.

## Start the dashboard

In a second PowerShell window, from this directory, run:

```powershell
streamlit run dashboard.py
```

The dashboard reads benchmark metrics and feedback summaries from the API. Its
**Try it** tab submits questions to the API and allows a user to rate the answer.
Feedback is tied to the request ID returned with the answer. For low-confidence
ML routing, ratings can influence model choice after each configured model has
at least five ratings and their average ratings differ by at least 0.5 points.
Until that threshold is reached, uncertain queries continue to use the
large-model fallback. Only ML-router ratings are used for this preference;
queries and comments are not used as model-training data.

The answer API keeps a bounded, five-minute in-memory cache keyed by the
normalized exact question and selected router. Cache hits still receive a new
request ID so each answer request can be rated separately. Cache entries and
hit/miss counters are cleared when the API process restarts. The dashboard
shows the current process hit rate at `/api/cache/stats`.

Answer responses include input/output tokens and a USD cost estimate when
LiteLLM has pricing data for the configured provider and model. Local Ollama
does not have a provider API charge in this metric; `cost_usd` is `null`.

The benchmark records LiteLLM per-answer cost estimates when model pricing is
available and shows the percentage of calls with estimates. It only calculates
estimated API-cost savings when all queries in the paired comparison have
cost data. These are estimates, not invoice-verified charges; confirm them
against the provider's billing dashboard. For local Ollama runs, use the
separately labeled model-size cost proxy; it is not a cash-savings or
electricity measurement.

If the API runs at a different address, set `RAG_ROUTER_API_URL` before starting
Streamlit or change the URL in the dashboard sidebar.

Benchmark quality scores are heuristic unless manually reviewed. Virtual cost
units are a model-size proxy, not measured provider charges or energy use.

## Reproducible local technical-document benchmark

The focused evaluation uses the RAG overview and Transformer paper already
included in the indexed documents. Its retrieval is restricted to the sources
named in the question set so unrelated health and business files do not affect
the comparison. It also tracks whether the expected source appears in the
top-three retrieved chunks. Two unsupported questions check whether the system
abstains. This is a small initial test set, not proof of broad generalization.

Run the benchmark using local Ollama models only:

```powershell
python -m evaluation.benchmark_updated --input evaluation\technical_eval_questions.csv --output evaluation\technical_benchmark_results.csv
```

Export a review sheet for the latest run before assigning any manual scores.
Use a new review filename if one already exists; the exporter will not
overwrite reviewer work:

```powershell
python evaluation\manual_review.py --input evaluation\technical_benchmark_results.csv --review evaluation\technical_manual_reviews.csv
```

Review each answer in `technical_manual_reviews.csv` and fill `manual_correct`.
Use `1` only when the answer directly addresses the question and its material
claims are supported by the expected source; otherwise use `0`. For an
out-of-scope question, use `1` only when the answer correctly abstains without
inventing an answer. Add short notes for incorrect or questionable responses.
Then apply the review, calculate preliminary heuristic scores, and print the
report:

```powershell
python evaluation\score_results_updated.py --input evaluation\technical_benchmark_results.csv --output evaluation\technical_benchmark_scored.csv
python evaluation\manual_review.py --input evaluation\technical_benchmark_scored.csv --review evaluation\technical_manual_reviews.csv --output evaluation\technical_benchmark_reviewed.csv --apply
python evaluation\router_metrics.py evaluation\technical_benchmark_reviewed.csv
```

The paired human-reviewed accuracy metric is withheld until both baseline and
routed answers for all factual questions in that comparison are reviewed. The
automated score remains a preliminary keyword check; use human-reviewed
results when presenting answer quality.

To display the focused results in the dashboard, set the scored file before
starting the API:

```powershell
$env:RAG_ROUTER_SCORED_RESULTS = "evaluation\technical_benchmark_reviewed.csv"
uvicorn api.app:app --reload
```

Then start Streamlit in another terminal. Without this environment variable,
the API continues using its default benchmark file.

Repeat runs append to the specified results file. Router execution order
rotates by question to reduce consistent first-run bias. For a clean repeat,
use a new output filename and compare multiple runs rather than relying on one
latency sample.
