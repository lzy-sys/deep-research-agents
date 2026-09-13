"""索引元数据与版本校验：保存 meta、模型/维度不匹配时拒绝加载。不触网。"""
import json

import faiss
import numpy as np
import pytest
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from deep_research import configuration as cfg
from deep_research.tools.retrieval import get_embeddings, load_vectorstore, save_vectorstore


def _make_vs() -> FAISS:
    index = faiss.IndexFlatIP(8)
    index.add(np.array([[1.0] * 8], dtype="float32"))
    docstore = InMemoryDocstore(
        {"0": Document(page_content="hello world", metadata={"source": "x/page", "url": "https://x/page"})}
    )
    return FAISS(get_embeddings(), index, docstore, {"0": "0"})


def test_save_writes_meta_and_loads_ok(tmp_path):
    save_vectorstore(_make_vs(), tmp_path)
    meta = json.loads((tmp_path / "index.meta.json").read_text(encoding="utf-8"))
    assert meta["embedding_model"] == cfg.EMBEDDING_MODEL
    assert meta["embedding_dims"] == cfg.EMBEDDING_DIMS
    assert meta["ntotal"] == 1

    out = load_vectorstore(tmp_path)
    assert out.index.ntotal == 1


def test_load_rejects_embedding_model_mismatch(tmp_path, monkeypatch):
    save_vectorstore(_make_vs(), tmp_path)
    monkeypatch.setattr(cfg, "EMBEDDING_MODEL", "another-embedding")
    with pytest.raises(RuntimeError, match="重建"):
        load_vectorstore(tmp_path)


def test_load_accepts_legacy_index_without_meta(tmp_path, monkeypatch):
    save_vectorstore(_make_vs(), tmp_path)
    (tmp_path / "index.meta.json").unlink()
    monkeypatch.setattr(cfg, "EMBEDDING_MODEL", "another-embedding")
    assert load_vectorstore(tmp_path).index.ntotal == 1  # 旧索引无 meta，不强制校验
