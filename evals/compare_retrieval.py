"""对比单次向量检索与代码控制 Agentic RAG 的召回、延迟和改查次数。

用法: uv run python evals/compare_retrieval.py [--limit 10]
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_retrieval import _git_metadata, _percentile, is_hit

from deep_research import configuration as cfg
from deep_research.tools.retrieval import _agentic_retrieve_result, get_vectorstore

K = 5
DATASET_PATH = Path(__file__).parent / "qa_dataset.jsonl"
RESULT_PATH = Path(__file__).parent / "results" / "retrieval_comparison_latest.json"


def _rank(sources: list[str], expected: list[str]) -> int | None:
    return next((i + 1 for i, source in enumerate(sources) if is_hit(source, expected)), None)


def _metrics(rows: list[dict], mode: str) -> dict:
    ranks = [row[mode]["rank"] for row in rows]
    hits = sum(rank is not None for rank in ranks)
    latencies = [row[mode]["seconds"] for row in rows]
    return {
        "hit_at_5": hits / len(rows),
        "mrr": sum(1 / rank for rank in ranks if rank) / len(rows),
        "latency_p50_seconds": _percentile(latencies, 0.50),
        "latency_p95_seconds": _percentile(latencies, 0.95),
        "fallback_count": sum(not row[mode].get("relevant", True) for row in rows),
        "average_attempts": sum(row[mode].get("attempts", 1) for row in rows) / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，默认全部")
    args = parser.parse_args()

    dataset_text = DATASET_PATH.read_text(encoding="utf-8")
    dataset = [json.loads(line) for line in dataset_text.splitlines() if line.strip()]
    if args.limit:
        dataset = dataset[: args.limit]

    vs = get_vectorstore()
    rows = []
    for item in dataset:
        started = time.perf_counter()
        docs = vs.similarity_search(item["question"], k=K)
        baseline_seconds = time.perf_counter() - started
        baseline_sources = [doc.metadata.get("source", "?") for doc in docs]

        started = time.perf_counter()
        agentic = _agentic_retrieve_result(item["question"])
        agentic_seconds = time.perf_counter() - started

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected": item["must_include"],
                "baseline": {
                    "sources": baseline_sources,
                    "rank": _rank(baseline_sources, item["must_include"]),
                    "seconds": baseline_seconds,
                },
                "agentic": {
                    "sources": agentic.sources,
                    "rank": _rank(agentic.sources, item["must_include"]),
                    "seconds": agentic_seconds,
                    "attempts": len(agentic.queries),
                    "queries": agentic.queries,
                    "relevant": agentic.relevant,
                },
            }
        )
        print(
            f"{item['id']}: baseline={rows[-1]['baseline']['rank'] or '-'} "
            f"agentic={rows[-1]['agentic']['rank'] or '-'} attempts={rows[-1]['agentic']['attempts']}",
            flush=True,
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_metadata(),
        "dataset": {
            "path": str(DATASET_PATH.relative_to(cfg.ROOT)),
            "sha256": hashlib.sha256(dataset_text.encode("utf-8")).hexdigest(),
            "samples": len(rows),
        },
        "index": {"ntotal": vs.index.ntotal, "embedding_model": cfg.EMBEDDING_MODEL},
        "baseline": _metrics(rows, "baseline"),
        "agentic": _metrics(rows, "agentic"),
        "rows": rows,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": result["baseline"], "agentic": result["agentic"]}, ensure_ascii=False, indent=2))
    print(f"结果: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
