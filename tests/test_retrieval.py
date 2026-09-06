"""知识库检索工具测试（需已建库 + 本地 Ollama，未建库自动跳过）。"""
import pytest

from deep_research import configuration as cfg

pytestmark = pytest.mark.skipif(
    not (cfg.CHROMA_DIR / "index.faiss.bin").exists(), reason="知识库未构建，先跑 scripts/ingest.py"
)


def test_load_and_search():
    from deep_research.tools.retrieval import get_vectorstore

    vs = get_vectorstore()
    assert vs.index.ntotal > 1000
    docs = vs.similarity_search("create_deep_agent subagents", k=3)
    assert len(docs) == 3
    assert all(d.metadata.get("source") for d in docs), "每条结果必须带来源"
