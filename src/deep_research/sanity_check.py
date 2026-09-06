"""连通性自检：网关 LLM / 结构化输出 / Tavily / Ollama embedding。

用法: uv run python -m src.deep_research.sanity_check
"""
import os
import sys

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

results: list[tuple[str, bool, str]] = []


def record(name: str, err: Exception | None = None, note: str = "") -> None:
    ok = err is None
    msg = note if ok else f"{type(err).__name__}: {err}"
    results.append((name, ok, msg))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {msg}" if msg else ""))


def main() -> int:
    base_url = os.getenv("OPENCODE_BASE_URL", "")
    api_key = os.getenv("OPENCODE_API_KEY", "")
    model = os.getenv("MODEL_RESEARCH", "deepseek-v4-flash")
    missing = [k for k in ("OPENCODE_API_KEY", "TAVILY_API_KEY") if not os.getenv(k) or "your-key" in os.getenv(k, "")]
    if missing:
        print(f"!! .env 缺少有效 Key: {', '.join(missing)}（先填 .env 再跑）\n")

    # 1. 网关模型列表
    print("\n[1/5] 网关模型列表")
    models: list[str] = []
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        record("GET /models", note=f"{len(models)} 个模型: {', '.join(models[:10])}{' ...' if len(models) > 10 else ''}")
    except Exception as e:
        record("GET /models", e, f"检查 OPENCODE_BASE_URL={base_url}")

    # 2. LLM 对话
    print("\n[2/5] LLM 对话（research 档）")
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, timeout=60)
        reply = llm.invoke("回复两个字：正常")
        record(f"chat ({model})", note=repr(reply.content)[:80])
    except Exception as e:
        llm = None
        record("chat", e)

    # 3. 结构化输出
    print("\n[3/5] 结构化输出")
    try:
        class Ping(BaseModel):
            status: str
            score: int

        assert llm is not None, "依赖检查2"
        out = llm.with_structured_output(Ping).invoke("返回 status=ok, score=100 的 JSON")
        record("with_structured_output", note=f"{out.status}/{out.score}")
    except Exception as e:
        record("with_structured_output", e)

    # 4. Tavily 搜索
    print("\n[4/5] Tavily 搜索")
    try:
        from langchain_tavily import TavilySearch

        hit = TavilySearch(max_results=2).invoke({"query": "LangGraph multi-agent"})
        urls = [h.get("url", "") for h in hit.get("results", [])]
        if not urls:
            raise RuntimeError("返回 0 条结果（Key 无效或配额用尽）")
        record("TavilySearch", note=f"{len(urls)} 条结果: {urls[0]}")
    except Exception as e:
        record("TavilySearch", e)

    # 5. Ollama embedding
    print("\n[5/5] Ollama embedding")
    try:
        from langchain_ollama import OllamaEmbeddings

        emb = OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        vec = emb.embed_query("连通性测试")
        record("OllamaEmbeddings", note=f"维度={len(vec)}")
    except Exception as e:
        record("OllamaEmbeddings", e, "确认 Ollama 已启动且模型已 pull")

    failed = [r for r in results if not r[1]]
    print(f"\n===== 结果: {len(results) - len(failed)}/{len(results)} PASS =====")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
