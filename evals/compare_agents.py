"""单 Agent 与完整多 Agent 的受控质量/成本/延迟对照。

用法: uv run python evals/compare_agents.py [--limit 3]
"""
import argparse
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepagents import create_deep_agent
from eval_retrieval import _git_metadata
from langchain.agents import create_agent
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from deep_research import configuration as cfg
from deep_research.events import final_content
from deep_research.llm import get_model
from deep_research.prompts import RAG_EXPERT_PROMPT
from deep_research.tools.retrieval import get_vectorstore

DATASET_PATH = Path(__file__).parent / "qa_dataset.jsonl"
RESULT_PATH = Path(__file__).parent / "results" / "agent_comparison_latest.json"
SINGLE_AGENT_PROMPT = """你是单 Agent 基线的研究助手。
只允许调用 benchmark_retrieve_docs 获取事实。最多检索 2 次，最终用中文简洁回答，并保留工具返回的来源标注。
知识库没有证据时明确说“知识库中未找到”，不得编造。"""
MULTI_AGENT_PROMPT = """你是研究主管。将用户问题完整交给 rag-expert 子智能体，收到结果后直接给出中文答案。
不要写文件，不要保存记忆，不要调用其他工具。最终答案必须保留 rag-expert 返回的来源标注。"""


@tool
def submit_quality_scores(single_score: int, multi_score: int, reason: str) -> str:
    """提交两个答案的 0-10 质量分和简短比较理由。"""
    return json.dumps(
        {"single_score": single_score, "multi_score": multi_score, "reason": reason},
        ensure_ascii=False,
    )


@tool
def benchmark_retrieve_docs(question: str) -> str:
    """用固定的单次向量检索返回证据，避免 CRAG 模型调用干扰 Agent 编排对照。"""
    docs = get_vectorstore().similarity_search(question, k=cfg.RAG_TOP_K)
    return "\n\n".join(
        f"[{i}] ({doc.metadata.get('source', 'unknown')})\n{doc.page_content}" for i, doc in enumerate(docs, 1)
    )


def _usage(callback) -> dict:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in callback.usage_metadata.values():
        for key in totals:
            totals[key] += usage.get(key) or 0
    return totals


def _invoke(runnable, question: str) -> tuple[str, float, dict]:
    started = time.perf_counter()
    with get_usage_metadata_callback() as callback:
        result = runnable.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={
                "configurable": {"thread_id": f"bench-{uuid.uuid4().hex[:8]}"},
                "recursion_limit": 12,
            },
        )
    return final_content(result["messages"]), time.perf_counter() - started, _usage(callback)


def _judge(question: str, expected: list[str], single: str, multi: str) -> dict | None:
    swap = int(uuid.uuid5(uuid.NAMESPACE_URL, question).hex[:2], 16) % 2 == 1
    answer_a, answer_b = (multi, single) if swap else (single, multi)
    prompt = f"""比较两个答案，按事实正确性、问题相关性和来源可信度各维度综合打 0-10 分。

问题：{question}
期望关键词：{"、".join(expected)}

答案 A：
{answer_a[:6000]}

答案 B：
{answer_b[:6000]}

只调用 submit_quality_scores。single_score 是单 Agent 答案得分，multi_score 是多 Agent 答案得分。"""
    try:
        response = get_model("research").bind_tools([submit_quality_scores]).invoke(prompt)
        call = next((c for c in response.tool_calls if c.get("name") == "submit_quality_scores"), None)
        if call is None:
            return None
        args = call.get("args") or {}
        return {
            "single_score": int(args["single_score"]),
            "multi_score": int(args["multi_score"]),
            "reason": str(args["reason"]),
            "swapped": swap,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _citation_count(answer: str) -> int:
    return len(re.findall(r"\[[0-9]+\]", answer))


def _quality_proxy(answer: str, keywords: list[str], must_include: list[str]) -> dict:
    """确定性代理指标：70% 期望词覆盖率 + 30% 是否引用期望文档。"""
    lowered = answer.lower()
    keyword_coverage = sum(keyword.lower() in lowered for keyword in keywords) / len(keywords)
    aliases = [source.split("__")[-1].removesuffix(".md").lower() for source in must_include]
    source_hit = any(alias in lowered for alias in aliases)
    return {
        "keyword_coverage": keyword_coverage,
        "source_hit": source_hit,
        "score": 0.7 * keyword_coverage + 0.3 * source_hit,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge", action="store_true", help="额外调用 LLM 做主观比较，结果仅作诊断")
    args = parser.parse_args()

    dataset = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        dataset = dataset[: args.limit]

    single_agent = create_agent(
        model=get_model("research"),
        tools=[benchmark_retrieve_docs],
        system_prompt=SINGLE_AGENT_PROMPT,
    )

    multi_agent = create_deep_agent(
        model=get_model("research"),
        system_prompt=MULTI_AGENT_PROMPT,
        subagents=[
            {
                "name": "rag-expert",
                "description": "根据固定向量检索结果回答 LangChain / LangGraph / DeepAgents / GLM 技术问题。",
                "system_prompt": RAG_EXPERT_PROMPT.replace("retrieve_docs", "benchmark_retrieve_docs"),
                "tools": [benchmark_retrieve_docs],
                "model": get_model("research"),
            }
        ],
        checkpointer=InMemorySaver(),
    )

    rows = []
    for item in dataset:
        print(f"{item['id']}: running single-agent", flush=True)
        single_answer, single_seconds, single_usage = _invoke(single_agent, item["question"])
        print(f"{item['id']}: running multi-agent", flush=True)
        multi_answer, multi_seconds, multi_usage = _invoke(multi_agent, item["question"])
        single_quality = _quality_proxy(single_answer, item["keywords"], item["must_include"])
        multi_quality = _quality_proxy(multi_answer, item["keywords"], item["must_include"])
        judge = _judge(item["question"], item["keywords"], single_answer, multi_answer) if args.judge else None
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "single": {
                    "answer": single_answer,
                    "seconds": single_seconds,
                    "usage": single_usage,
                    "citations": _citation_count(single_answer),
                    "quality": single_quality,
                },
                "multi": {
                    "answer": multi_answer,
                    "seconds": multi_seconds,
                    "usage": multi_usage,
                    "citations": _citation_count(multi_answer),
                    "quality": multi_quality,
                },
                "judge": judge,
            }
        )
        print(
            f"{item['id']}: single={single_seconds:.1f}s/{single_usage['total_tokens']}tok "
            f"q={single_quality['score']:.2f} "
            f"multi={multi_seconds:.1f}s/{multi_usage['total_tokens']}tok "
            f"q={multi_quality['score']:.2f} "
            f"score={judge.get('single_score') if judge else '?'}:"
            f"{judge.get('multi_score') if judge else '?'}",
            flush=True,
        )

    judged = [row for row in rows if row["judge"] and "multi_score" in row["judge"]]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_metadata(),
        "samples": len(rows),
        "retrieval_mode": "fixed_single_vector",
        "summary": {
            "single_quality_proxy": sum(row["single"]["quality"]["score"] for row in rows) / len(rows),
            "multi_quality_proxy": sum(row["multi"]["quality"]["score"] for row in rows) / len(rows),
            "single_judge_quality": (
                sum(row["judge"]["single_score"] for row in judged) / len(judged) if judged else None
            ),
            "multi_judge_quality": (
                sum(row["judge"]["multi_score"] for row in judged) / len(judged) if judged else None
            ),
            "single_seconds_p50": sorted(row["single"]["seconds"] for row in rows)[len(rows) // 2],
            "multi_seconds_p50": sorted(row["multi"]["seconds"] for row in rows)[len(rows) // 2],
            "single_total_tokens": sum(row["single"]["usage"]["total_tokens"] for row in rows),
            "multi_total_tokens": sum(row["multi"]["usage"]["total_tokens"] for row in rows),
        },
        "rows": rows,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"结果: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
