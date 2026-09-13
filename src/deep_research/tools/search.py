"""搜索工具：Tavily（basic/advanced 两档）。"""
from functools import lru_cache

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from deep_research import configuration as cfg


def _wrap_untrusted(source: str, content: object) -> str:
    return f"<untrusted_{source}_content>\n{content}\n</untrusted_{source}_content>"


@lru_cache
def _basic_search() -> TavilySearch:
    return TavilySearch(name="web_search_raw", max_results=5)


@lru_cache
def _deep_search() -> TavilySearch:
    return TavilySearch(
        name="web_search_deep_raw",
        max_results=5,
        search_depth="advanced",
        include_raw_content=True,
    )


@tool
def web_search(query: str) -> str:
    """快速搜索公开网络，返回 5 条候选结果。"""
    return _wrap_untrusted("web", _basic_search().invoke({"query": query}))


@tool
def web_search_deep(query: str) -> str:
    """深搜公开网络并包含原文，用于快搜不足或信息冲突时。"""
    return _wrap_untrusted("web", _deep_search().invoke({"query": query}))


def get_search_tools() -> list:
    """返回搜索工具列表，供 web_researcher 使用。

    - web_search: basic 快搜，5 条结果，首轮用
    - web_search_deep: advanced 深搜 + 原文内容，信息不足时升级用
    """
    if cfg.SEARCH_PROVIDER != "tavily":
        raise ValueError(f"未知 SEARCH_PROVIDER: {cfg.SEARCH_PROVIDER}（当前仅支持 tavily）")
    return [web_search, web_search_deep]
