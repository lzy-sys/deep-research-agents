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
    return ChatOpenAI(
        model=model_id,
        base_url=cfg.OPENCODE_BASE_URL,
        api_key=cfg.OPENCODE_API_KEY,
        temperature=0,
        timeout=60,       # 网关偶发挂起连接，无超时会无限等待（页面转圈的根因）
        max_retries=2,
        # 强制非流式：网关晚高峰流式吞吐会塌到 ~18 token/s（200 字拖 60s+），
        # 涓涓细流让读超时永远不触发；非流式实测 2s 级整体返回，超时保护才能生效
        disable_streaming=True,
    )
