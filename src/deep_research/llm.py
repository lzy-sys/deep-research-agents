"""模型工厂：按档取模型（research/summarize/report），统一走 OpenCode 网关。"""
from functools import lru_cache

from langchain_openai import ChatOpenAI

from deep_research import configuration as cfg


@lru_cache
def get_model(role: str = "research") -> ChatOpenAI:
    """role: research(拆题/判断) | summarize(高频摘要) | report(最终报告)"""
    model_id = {
        "research": cfg.MODEL_RESEARCH,
        "summarize": cfg.MODEL_SUMMARIZE,
        "report": cfg.MODEL_REPORT,
    }[role]
    if cfg.LLM_PROVIDER == "ollama":
        # 本地 Ollama 的 OpenAI 兼容端点：零 API 成本；冷启动加载模型较慢，超时放宽
        base_url = cfg.OLLAMA_BASE_URL.rstrip("/") + "/v1"
        api_key = "ollama"
    else:
        base_url = cfg.OPENCODE_BASE_URL
        api_key = cfg.OPENCODE_API_KEY
    return ChatOpenAI(
        model=model_id,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        timeout=cfg.LLM_TIMEOUT,
        max_retries=2,
        # 强制非流式：网关晚高峰流式吞吐会塌到 ~18 token/s（200 字拖 60s+），
        # 涓涓细流让读超时永远不触发；非流式实测 2s 级整体返回，超时保护才能生效
        disable_streaming=True,
        # 禁用 keep-alive 复用：网关空闲期单方面掐断池内连接，长驻进程复用半死连接
        # 会把请求发进黑洞且超时不触发（反复"转圈卡死"的最终根因）
        default_headers={"Connection": "close"},
        # 关思考模式（实测 0.9s vs 5.1s）：thinking 产生的 reasoning_content 在多轮
        # 工具调用时必须回传，langchain 会剥掉 → 上游 400 "must be passed back"
        extra_body=cfg.LLM_EXTRA_BODY,
    )
