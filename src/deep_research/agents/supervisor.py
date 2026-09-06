"""主智能体（Supervisor）：deepagents harness + 3 个声明式 subagent。"""
from datetime import date

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from deep_research import configuration as cfg
from deep_research.agents.rag_expert import build_rag_expert
from deep_research.agents.researcher import build_researcher
from deep_research.agents.sql_expert import build_sql_expert
from deep_research.llm import get_model
from deep_research.memory.store import read_memory
from deep_research.prompts import MEMORY_EXTRACT_HINT, SUPERVISOR_PROMPT


def build_research_agent(checkpointer=None):
    """编译完整多智能体图。checkpointer 供多轮会话使用。"""
    return create_deep_agent(
        model=get_model("research"),
        system_prompt=SUPERVISOR_PROMPT.format(user_memory=read_memory(), today=date.today().isoformat()),
        subagents=[build_researcher(), build_rag_expert(), build_sql_expert()],
        backend=FilesystemBackend(root_dir=str(cfg.ROOT)),  # write_file 直接落盘项目目录
        memory=["AGENTS.md"],  # 长期记忆：跨会话持久文件
        checkpointer=checkpointer,
    )
