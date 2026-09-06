"""检索评测：hit@5 + MRR，逐条明细写 reports/eval_retrieval.md。

用法: uv run python evals/eval_retrieval.py
命中率 >= 80% 退出码 0，否则 1（可挂 CI）。
"""
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_research import configuration as cfg  # noqa: E402
from deep_research.tools.retrieval import get_vectorstore  # noqa: E402

K = 5
PASS_RATE = 0.80


def is_hit(source: str, must_include: list[str]) -> bool:
    return any(m in source for m in must_include)


def main() -> int:
    dataset = [
        json.loads(line)
        for line in (Path(__file__).parent / "qa_dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vs = get_vectorstore()
    print(f"知识库共 {vs.index.ntotal} 块，评测 {len(dataset)} 条，k={K}")

    rows, t0 = [], time.time()
    for item in dataset:
        docs = vs.similarity_search(item["question"], k=K)
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

    lines = [
        "# 检索评测报告",
        "",
        f"- 日期：{date.today().isoformat()} | 知识库：{vs.index.ntotal} 块 | 样本：{len(rows)} 条 | k={K}",
        f"- **hit@{K}：{hits}/{len(rows)} = {hit5:.1%}**",
        f"- **MRR：{mrr:.3f}**",
        f"- 耗时：{elapsed:.1f}s",
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
    print(f"hit@{K}: {hits}/{len(rows)} = {hit5:.1%} | MRR: {mrr:.3f} | 报告: {report}")
    return 0 if hit5 >= PASS_RATE else 1


if __name__ == "__main__":
    sys.exit(main())
