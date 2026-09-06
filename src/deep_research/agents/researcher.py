"""web-researcher 子智能体：外部网络信息调研。"""
from deep_research.llm import get_model
from deep_research.prompts import RESEARCHER_PROMPT
from deep_research.tools.search import get_search_tools


def build_researcher() -> dict:
    return {
        "name": "web-researcher",
        "description": (
            "外部网络信息调研员。负责时效性、行业动态、行情数据、公开网络信息等研究任务。"
            "输入一个自包含的调研任务描述，返回压缩后的结论要点与来源 URL 列表。"
        ),
        "system_prompt": RESEARCHER_PROMPT,
        "tools": get_search_tools(),
        "model": get_model("summarize"),
    }
