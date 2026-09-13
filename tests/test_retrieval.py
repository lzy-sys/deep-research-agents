"""知识库检索工具测试：离线确定性冒烟 + 可选真实索引集成测试。"""
import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from deep_research import configuration as cfg
from deep_research.tools.retrieval import INDEX_FILE


def test_fake_embedding_search_is_offline():
    docs = [
        Document(page_content="LangGraph checkpoint persistence", metadata={"source": "langgraph/checkpoint"}),
        Document(page_content="DeepAgents subagent delegation", metadata={"source": "deepagents/subagents"}),
    ]
    vs = FAISS.from_documents(docs, DeterministicFakeEmbedding(size=8))
    hits = vs.similarity_search("checkpoint", k=1)
    assert len(hits) == 1
    assert hits[0].metadata["source"]


@pytest.mark.skipif(
    not (cfg.FAISS_DIR / INDEX_FILE).exists(), reason="知识库未构建，先跑 scripts/ingest.py"
)
def test_load_and_search():
    from deep_research.tools.retrieval import get_vectorstore

    try:
        vs = get_vectorstore()
        docs = vs.similarity_search("create_deep_agent subagents", k=3)
    except Exception as e:  # Ollama 冷启动/模型缺失时跳过真实集成测试
        pytest.skip(f"Ollama/embedding 不可用，跳过: {type(e).__name__}: {e}")
    assert vs.index.ntotal > 1000
    assert len(docs) == 3
    assert all(d.metadata.get("source") for d in docs), "每条结果必须带来源"
