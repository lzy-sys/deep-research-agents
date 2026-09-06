"""图组装入口：编译多智能体（默认 InMemorySaver 供多轮会话）。"""
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.agents.supervisor import build_research_agent


def build(checkpointer=None):
    return build_research_agent(checkpointer=checkpointer or InMemorySaver())
