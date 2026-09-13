"""检索评测：hit@5 + MRR，逐条明细写 reports/eval_retrieval.md。

用法: uv run python evals/eval_retrieval.py
命中率 >= 80% 退出码 0，否则 1（可挂 CI）。
"""
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_research import configuration as cfg  # noqa: E402
from deep_research.tools.retrieval import get_vectorstore  # noqa: E402

K = 5
PASS_RATE = 0.80
DATASET_PATH = Path(__file__).parent / "qa_dataset.jsonl"
RESULT_PATH = Path(__file__).parent / "results" / "retrieval_latest.json"


def _git_metadata() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()

    return {
        "sha": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * pct))
    return ordered[index]


def is_hit(source: str, must_include: list[str]) -> bool:
    return any(m in source for m in must_include)


def main() -> int:
    dataset_text = DATASET_PATH.read_text(encoding="utf-8")
    dataset = [
        json.loads(line)
        for line in dataset_text.splitlines()
        if line.strip()
    ]
    vs = get_vectorstore()
    print(f"知识库共 {vs.index.ntotal} 块，评测 {len(dataset)} 条，k={K}")

    rows, latencies, t0 = [], [], time.time()
    for item in dataset:
        query_started = time.perf_counter()
        docs = vs.similarity_search(item["question"], k=K)
        latencies.append(time.perf_counter() - query_started)
        sources = [d.metadata.get("source", "?") for d in docs]
        rank = next((i + 1 for i, s in enumerate(sources) if is_hit(s, item["must_include"])), None)
        rows.append({**item, "sources": sources, "rank": rank})

    def section(qid: str) -> str:
        return qid.split("-")[0]

    by_sec = defaultdict(lambda: [0, 0])
    for r in rows:
        by_sec[section(r["id"])][0] += r["rank"] is not None
        by_sec[section(r["id"])][1] += 1

    hits = sum(r["rank"] is not None for r in rows)
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / len(rows)
    hit5 = hits / len(rows)
    elapsed = time.time() - t0
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "generated_at": generated_at,
        "git": _git_metadata(),
        "dataset": {
            "path": str(DATASET_PATH.relative_to(cfg.ROOT)),
            "sha256": hashlib.sha256(dataset_text.encode("utf-8")).hexdigest(),
            "samples": len(rows),
        },
        "index": {
            "path": str(cfg.FAISS_DIR.relative_to(cfg.ROOT)),
            "ntotal": vs.index.ntotal,
            "embedding_model": cfg.EMBEDDING_MODEL,
            "embedding_dims": cfg.EMBEDDING_DIMS,
            "chunk_size": cfg.CHUNK_SIZE,
            "chunk_overlap": cfg.CHUNK_OVERLAP,
        },
        "retrieval": {"k": K, "top_k_config": cfg.RAG_TOP_K},
        "metrics": {
            "hit_at_5": hit5,
            "mrr": mrr,
            "latency_p50_seconds": _percentile(latencies, 0.50),
            "latency_p95_seconds": _percentile(latencies, 0.95),
            "total_seconds": elapsed,
        },
        "rows": rows,
    }

    lines = [
        "# 检索评测报告",
        "",
        f"- 时间：{generated_at} | Git：`{meta['git']['sha'][:12]}` | dirty={meta['git']['dirty']}",
        f"- 数据集：`{meta['dataset']['path']}` | SHA-256：`{meta['dataset']['sha256'][:16]}...`",
        f"- 知识库：{vs.index.ntotal} 块 | embedding={cfg.EMBEDDING_MODEL} | 样本：{len(rows)} 条 | k={K}",
        f"- **hit@{K}：{hits}/{len(rows)} = {hit5:.1%}**",
        f"- **MRR：{mrr:.3f}**",
        f"- 延迟：p50={_percentile(latencies, 0.50):.3f}s | p95={_percentile(latencies, 0.95):.3f}s | total={elapsed:.1f}s",
        "",
        "| 板块 | 命中 | 占比 |",
        "|---|---|---|",
    ]
    for sec, (h, n) in sorted(by_sec.items()):
        lines.append(f"| {sec} | {h}/{n} | {h / n:.0%} |")
    lines += ["", "| ID | 问题 | 命中排名 | 命中来源 |", "|---|---|---|---|"]
    for r in rows:
        rank = r["rank"] or "-"
        hit_src = next((s for s in r["sources"] if is_hit(s, r["must_include"])), "-")
        lines.append(f"| {r['id']} | {r['question'][:30]} | {rank} | {hit_src[:60]} |")
    misses = [r for r in rows if r["rank"] is None]
    if misses:
        lines += ["", "## 未命中条目", ""]
        for r in misses:
            lines.append(f"- **{r['id']}** {r['question']}（期望 {r['must_include']}）→ 实际 top1: {r['sources'][0]}")

    report = cfg.REPORTS_DIR / "eval_retrieval.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"hit@{K}: {hits}/{len(rows)} = {hit5:.1%} | MRR: {mrr:.3f} | "
        f"p95: {_percentile(latencies, 0.95):.3f}s | JSON: {RESULT_PATH}"
    )
    return 0 if hit5 >= PASS_RATE else 1


if __name__ == "__main__":
    sys.exit(main())
