"""搜索工具：Tavily（basic/advanced 两档）。"""
from langchain_tavily import TavilySearch

from deep_research import configuration as cfg


def get_search_tools() -> list:
    """返回搜索工具列表，供 web_researcher 使用。

    - web_search: basic 快搜，5 条结果，首轮用
    - web_search_deep: advanced 深搜 + 原文内容，信息不足时升级用
    """
    if cfg.SEARCH_PROVIDER != "tavily":
        raise ValueError(f"未知 SEARCH_PROVIDER: {cfg.SEARCH_PROVIDER}（当前仅支持 tavily）")
    return [
        TavilySearch(name="web_search", max_results=5),
        TavilySearch(
            name="web_search_deep",
            max_results=5,
            search_depth="advanced",
            include_raw_content=True,
        ),
    ]
