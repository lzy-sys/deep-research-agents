# Benchmarks

Results are reproducible artifacts containing Git SHA, dataset hash, model metadata, latency, and per-question details.

## Retrieval

Command:

```bash
uv run python evals/compare_retrieval.py
```

| Mode | hit@5 | MRR | p50 | p95 |
|---|---:|---:|---:|---:|
| Single-vector | 87.5% (21/24) | 0.719 | 0.138s | 0.157s |
| Agentic RAG | 95.8% (23/24) | 0.741 | 0.133s | 41.017s |

Agentic RAG only escalates low-confidence queries to model grading and query rewriting. Normal queries stay on the low-latency vector path; the tail latency appears only when extra retrieval rounds are needed.

Raw result: `evals/results/retrieval_comparison_latest.json`.

## Single Agent vs Multi Agent

Command:

```bash
uv run python evals/compare_agents.py
```

The comparison fixes the retrieval tool to a single vector search so the measured difference comes from agent orchestration rather than CRAG. Current sample size is 3 representative questions.

| Metric | Single Agent | Multi Agent |
|---|---:|---:|
| Deterministic quality proxy | 0.800 | 1.000 |
| Median latency | 11.147s | 79.503s |
| Total tokens | 6,059 | 68,755 |

The LLM judge is optional (`--judge`) and was unstable across repeated runs, so it is stored only as diagnostic data and is not used for the headline result.

Raw result: `evals/results/agent_comparison_latest.json`.
