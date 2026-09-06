"""rag-expert 子智能体：框架知识库问答（Agentic RAG）。"""
from deep_research.llm import get_model
from deep_research.prompts import RAG_EXPERT_PROMPT
from deep_research.tools.retrieval import retrieve_docs


def build_rag_expert() -> dict:
    return {
        "name": "rag-expert",
        "description": (
            "框架知识库专家。回答 LangChain / LangGraph / DeepAgents 的概念、API、用法、最佳实践问题，"
            "以及智谱 GLM 等国产模型的接入问题，答案基于本地文档知识库并标注来源。输入一个具体的技术问题。"
        ),
        "system_prompt": RAG_EXPERT_PROMPT,
        "tools": [retrieve_docs],
        "model": get_model("research"),
    }
