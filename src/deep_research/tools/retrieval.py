"""FAISS 知识库检索：Ollama embedding + 本地持久化（Python 侧序列化）。

ponytail: 弃用 chromadb 1.5.9（Windows 上 HNSW 新进程加载必崩）与 faiss 自带文件 IO
（C++ fopen 不支持含中文的项目路径），改用 serialize_index + pickle，引擎仍是 FAISS。
"""
import json
import math
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

from deep_research import configuration as cfg

_INDEX_FILE = INDEX_FILE = "index.faiss.bin"  # INDEX_FILE 公开给 ingest/测试复用
_META_FILE = "store.pkl"
_META_JSON = "index.meta.json"
MAX_RETRIEVAL_ATTEMPTS = 3
MIN_ACCEPT_SCORE = 0.60


@dataclass(frozen=True)
class RetrievalResult:
    content: str
    queries: list[str]
    sources: list[str]
    relevant: bool


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=cfg.EMBEDDING_MODEL, base_url=cfg.OLLAMA_BASE_URL)


def save_vectorstore(vs: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / _INDEX_FILE).write_bytes(faiss.serialize_index(vs.index).tobytes())
    with open(path / _META_FILE, "wb") as f:
        pickle.dump({"docstore": vs.docstore, "map": vs.index_to_docstore_id}, f)
    (path / _META_JSON).write_text(
        json.dumps(
            {
                "embedding_model": cfg.EMBEDDING_MODEL,
                "embedding_dims": cfg.EMBEDDING_DIMS,
                "chunk_size": cfg.CHUNK_SIZE,
                "chunk_overlap": cfg.CHUNK_OVERLAP,
                "ntotal": vs.index.ntotal,
                "built_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_vectorstore(path: Path) -> FAISS:
    index = faiss.deserialize_index(np.frombuffer((path / _INDEX_FILE).read_bytes(), dtype=np.uint8))
    with open(path / _META_FILE, "rb") as f:
        d = pickle.load(f)
    meta_path = path / _META_JSON
    if meta_path.exists():  # 旧索引无 meta 视为 legacy，不做校验
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("embedding_model") != cfg.EMBEDDING_MODEL:
            raise RuntimeError(
                f"索引由 embedding 模型 {meta.get('embedding_model')} 构建，当前配置 {cfg.EMBEDDING_MODEL}；"
                "换模型后必须重建知识库（uv run python scripts/ingest.py --refresh）"
            )
        if meta.get("embedding_dims") != cfg.EMBEDDING_DIMS:
            raise RuntimeError(
                f"索引维度 {meta.get('embedding_dims')} 与当前配置 {cfg.EMBEDDING_DIMS} 不一致；请重建知识库"
            )
    return FAISS(get_embeddings(), index, d["docstore"], d["map"])


@lru_cache
def get_vectorstore() -> FAISS:
    if not (cfg.FAISS_DIR / _INDEX_FILE).exists():
        raise RuntimeError("知识库不存在，先运行: uv run python scripts/ingest.py")
    return load_vectorstore(cfg.FAISS_DIR)


@tool
def submit_retrieval_assessment(relevant: bool, reason: str, rewritten_query: str) -> str:
    """提交检索质量判断。relevant=false 时必须给出更好的 rewritten_query，否则传空字符串。"""
    return json.dumps(
        {"relevant": relevant, "reason": reason, "rewritten_query": rewritten_query},
        ensure_ascii=False,
    )


def _search_with_scores(question: str) -> list[tuple]:
    hits = get_vectorstore().similarity_search_with_score(question, k=cfg.RAG_TOP_K)
    return [
        # FAISS IndexFlat* 默认距离经 LangChain relevance 转换后再截断到 [0, 1]，
        # 避免负相关度触发库内警告，同时保持与 similarity_search 相同排序。
        (doc, max(0.0, min(1.0, 1.0 - float(score) / math.sqrt(2))))
        for doc, score in hits
    ]


def _assess_with_model(question: str, hits: list[tuple]) -> tuple[bool, str]:
    """用 function calling 判断候选片段能否回答问题，并在失败时生成新查询。"""
    if hits and float(hits[0][1]) >= MIN_ACCEPT_SCORE:
        return True, ""

    from deep_research.llm import get_model

    candidates = "\n\n".join(
        f"[{i}] score={float(score):.3f} source={doc.metadata.get('source', 'unknown')}\n"
        f"{doc.page_content[:1200]}"
        for i, (doc, score) in enumerate(hits, 1)
    )
    prompt = f"""判断下面的候选文档能否直接回答用户问题。只调用 submit_retrieval_assessment。

用户问题：{question}

候选文档：
{candidates or "（没有检索结果）"}

判断规则：
1. 候选内容必须包含回答问题所需的证据，不能只因出现相同关键词或主题相近就判为相关；但允许证据分散在多个候选片段中。
2. 问题询问“怎么用/如何/API/示例/参数”时，候选必须包含具体用法、接口或示例，不能在仅提及支持该能力时判为相关。
3. 问题询问“有哪些/对比/选型”时，候选必须包含具体列表、差异或选型说明。
4. 最高分低于 0.60 时，即使 relevant=true 也应给出更具体的 rewritten_query 做二次确认；确实无法改进时才传空字符串。
5. rewritten_query 必须保留原问题中的精确 API 名、缩写和关键词（如 SSE、function calling、Send），不得把它们泛化掉。
6. relevant=false 时，rewritten_query 必须非空，不能原样返回。
7. reason 简短说明证据是否充分。"""
    response = get_model("research").bind_tools([submit_retrieval_assessment]).invoke(prompt)
    call = next((c for c in response.tool_calls if c.get("name") == "submit_retrieval_assessment"), None)
    if call is None:
        raise RuntimeError("检索质量模型未返回结构化判断")
    args = call.get("args") or {}
    return bool(args.get("relevant")), str(args.get("rewritten_query") or "").strip()


def _fallback_assessment(hits: list[tuple]) -> tuple[bool, str]:
    top_score = float(hits[0][1]) if hits else 0.0
    return top_score >= MIN_ACCEPT_SCORE, ""


def _query_terms(question: str) -> list[str]:
    """轻量中英文词项：英文词权重 2，中文二元组权重 1。"""
    latin = re.findall(r"[a-z][a-z0-9_-]{1,}", question.lower())
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]+", question))
    bigrams = [chinese[i : i + 2] for i in range(max(0, len(chinese) - 1))]
    return [*latin, *latin, *bigrams]


def _lexical_score(question: str, doc) -> int:
    text = f"{doc.metadata.get('source', '')} {doc.page_content}".lower()
    return sum(term in text for term in _query_terms(question))


def _agentic_retrieve_result(
    question: str,
    search_fn: Callable[[str], list[tuple]] | None = None,
    assess_fn: Callable[[str, list[tuple]], tuple[bool, str]] | None = None,
) -> RetrievalResult:
    """代码控制的检索闭环：retrieve -> grade -> rewrite -> retry -> fallback。"""
    search = search_fn or _search_with_scores
    assess = assess_fn or _assess_with_model
    query = question
    attempted: list[str] = []
    attempts: list[list[tuple]] = []
    any_relevant = False

    for _ in range(MAX_RETRIEVAL_ATTEMPTS):
        attempted.append(query)
        hits = search(query)
        try:
            relevant, rewritten = assess(query, hits)
        except Exception:
            relevant, rewritten = _fallback_assessment(hits)
        if relevant and hits:
            any_relevant = True
        attempts.append(hits)
        if rewritten and rewritten not in attempted:
            query = rewritten
        else:
            break

    if any_relevant:
        # 多轮候选按词项覆盖 + RRF 融合，避免“高分但跑题”的后续查询覆盖正确证据。
        fused: dict[str, dict] = {}
        for hits in attempts:
            for rank, (doc, score) in enumerate(hits, 1):
                key = doc.id or f"{doc.metadata.get('source', 'unknown')}:{doc.page_content}"
                item = fused.setdefault(key, {"doc": doc, "score": float(score), "rrf": 0.0})
                item["rrf"] += 1.0 / (60 + rank)
                item["score"] = max(item["score"], float(score))
        ordered = sorted(
            fused.values(),
            key=lambda item: (_lexical_score(question, item["doc"]), item["rrf"], item["score"]),
            reverse=True,
        )
        selected = [(item["doc"], item["score"]) for item in ordered[: cfg.RAG_TOP_K]]
        content = "\n\n".join(
            f"[{i}] ({doc.metadata.get('source', 'unknown')}) score={float(score):.3f}\n{doc.page_content}"
            for i, (doc, score) in enumerate(selected, 1)
        )
        sources = [doc.metadata.get("source", "unknown") for doc, _ in selected]
        return RetrievalResult(content, attempted, sources, True)

    content = (
        "知识库中未找到足够相关的证据。"
        f"已尝试 {len(attempted)} 次查询：{'；'.join(attempted)}。"
        "建议转交 web-researcher 查询最新资料。"
    )
    return RetrievalResult(content, attempted, [], False)


def _agentic_retrieve(
    question: str,
    search_fn: Callable[[str], list[tuple]] | None = None,
    assess_fn: Callable[[str, list[tuple]], tuple[bool, str]] | None = None,
) -> str:
    return _agentic_retrieve_result(question, search_fn, assess_fn).content


@tool
def retrieve_docs(question: str) -> str:
    """检索 AI Agent 技术栈知识库；相关度不足时自动改写查询重试，最多 3 次。"""
    return _agentic_retrieve(question)
