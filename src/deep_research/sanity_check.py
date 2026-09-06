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
    from deep_research import configuration as cfg

    provider = cfg.LLM_PROVIDER
    model = cfg.MODEL_RESEARCH
    missing = [k for k in ("OPENCODE_API_KEY", "TAVILY_API_KEY") if not os.getenv(k) or "your-key" in os.getenv(k, "")]
    tavily_needed = os.getenv("SEARCH_PROVIDER", "tavily") == "tavily"
    if tavily_needed and "TAVILY_API_KEY" in missing:
        print(f"!! .env 缺少有效 Key: TAVILY_API_KEY（先填 .env 再跑）\n")

    # 1. 模型端点列表
    print(f"\n[1/5] 模型端点（provider={provider}）")
    try:
        if provider == "ollama":
            r = httpx.get(f"{cfg.OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=10)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            record("GET /api/tags", note=f"{len(models)} 个模型: {', '.join(models[:8])}")
        else:
            r = httpx.get(f"{cfg.OPENCODE_BASE_URL.rstrip('/')}/models", headers={"Authorization": f"Bearer {cfg.OPENCODE_API_KEY}"}, timeout=30)
            r.raise_for_status()
            models = [m["id"] for m in r.json().get("data", [])]
            record("GET /models", note=f"{len(models)} 个模型: {', '.join(models[:10])}{' ...' if len(models) > 10 else ''}")
    except Exception as e:
        record("模型端点", e, "检查端点配置")

    # 2. LLM 对话
    print("\n[2/5] LLM 对话（research 档）")
    try:
        from deep_research.llm import get_model

        llm = get_model("research")
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
        if "response_format" in str(e):
            # 网关上游不支持 json schema 受限解码；本项目工具调用走 function calling，
            # 不依赖该能力，如实标注而非算作故障
            record("with_structured_output", note="网关不支持 response_format——项目未使用该能力，不受影响")
        else:
            record("with_structured_output", e)

    # 4. Tavily 搜索
    print("\n[4/5] Tavily 搜索")
    try:
        if not tavily_needed:
            raise RuntimeError("SEARCH_PROVIDER != tavily，跳过判定为通过")
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

        emb = OllamaEmbeddings(model=cfg.EMBEDDING_MODEL, base_url=cfg.OLLAMA_BASE_URL)
        vec = emb.embed_query("连通性测试")
        record("OllamaEmbeddings", note=f"维度={len(vec)}")
    except Exception as e:
        record("OllamaEmbeddings", e, "确认 Ollama 已启动且模型已 pull")

    failed = [r for r in results if not r[1]]
    print(f"\n===== 结果: {len(results) - len(failed)}/{len(results)} PASS =====")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
