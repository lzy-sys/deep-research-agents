"""代码控制的 Agentic RAG 检索闭环测试，不访问模型或 Ollama。"""
from langchain_core.documents import Document

from deep_research.tools.retrieval import (
    MAX_RETRIEVAL_ATTEMPTS,
    _agentic_retrieve,
    _fallback_assessment,
)


def _hit(source: str, score: float = 0.8) -> tuple[Document, float]:
    return Document(page_content="relevant content", metadata={"source": source}), score


def test_rewrites_once_then_returns_evidence():
    calls: list[str] = []

    def search(query: str):
        calls.append(query)
        if query == "原始问题":
            return [_hit("wrong", 0.3)]
        return [_hit("right", 0.9)]

    def assess(query: str, hits: list[tuple]):
        return (False, "LangGraph checkpointer") if query == "原始问题" else (True, "")

    out = _agentic_retrieve("原始问题", search_fn=search, assess_fn=assess)

    assert calls == ["原始问题", "LangGraph checkpointer"]
    assert "right" in out
    assert "score=0.900" in out


def test_stops_after_max_attempts_and_falls_back():
    calls: list[str] = []

    def search(query: str):
        calls.append(query)
        return [_hit(f"wrong-{len(calls)}", 0.2)]

    def assess(query: str, hits: list[tuple]):
        return False, f"rewrite-{len(calls) + 1}"

    out = _agentic_retrieve("原始问题", search_fn=search, assess_fn=assess)

    assert len(calls) == MAX_RETRIEVAL_ATTEMPTS
    assert "知识库中未找到足够相关的证据" in out
    assert "web-researcher" in out


def test_fallback_requires_high_score():
    assert _fallback_assessment([_hit("high", 0.61)])[0] is True
    assert _fallback_assessment([_hit("low", 0.59)])[0] is False
    assert _fallback_assessment([])[0] is False
