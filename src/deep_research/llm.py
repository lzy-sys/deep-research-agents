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
    return ChatOpenAI(model=model_id, base_url=cfg.OPENCODE_BASE_URL, api_key=cfg.OPENCODE_API_KEY, temperature=0)
