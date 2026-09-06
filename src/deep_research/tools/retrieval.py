"""FAISS 知识库检索：Ollama embedding + 本地持久化（Python 侧序列化）。

ponytail: 弃用 chromadb 1.5.9（Windows 上 HNSW 新进程加载必崩）与 faiss 自带文件 IO
（C++ fopen 不支持含中文的项目路径），改用 serialize_index + pickle，引擎仍是 FAISS。
"""
import pickle
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

from deep_research import configuration as cfg

_INDEX_FILE = "index.faiss.bin"
_META_FILE = "store.pkl"


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=cfg.EMBEDDING_MODEL, base_url=cfg.OLLAMA_BASE_URL)


def save_vectorstore(vs: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / _INDEX_FILE).write_bytes(faiss.serialize_index(vs.index).tobytes())
    with open(path / _META_FILE, "wb") as f:
        pickle.dump({"docstore": vs.docstore, "map": vs.index_to_docstore_id}, f)


def load_vectorstore(path: Path) -> FAISS:
    index = faiss.deserialize_index(np.frombuffer((path / _INDEX_FILE).read_bytes(), dtype=np.uint8))
    with open(path / _META_FILE, "rb") as f:
        d = pickle.load(f)
    return FAISS(get_embeddings(), index, d["docstore"], d["map"])


@lru_cache
def get_vectorstore() -> FAISS:
    if not (cfg.CHROMA_DIR / _INDEX_FILE).exists():
        raise RuntimeError("知识库不存在，先运行: uv run python scripts/ingest.py")
    return load_vectorstore(cfg.CHROMA_DIR)


@tool
def retrieve_docs(question: str) -> str:
    """检索 AI Agent 技术栈知识库（LangChain/LangGraph/DeepAgents 英文官方文档 + 智谱 GLM 中文文档），返回最相关的文档片段（含来源）。"""
    docs = get_vectorstore().similarity_search(question, k=cfg.RAG_TOP_K)
    if not docs:
        return "知识库中未找到相关内容。"
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('source', 'unknown')})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )
