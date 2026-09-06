"""主智能体（Supervisor）：deepagents harness + 3 个声明式 subagent + 记忆工具。"""
from datetime import date

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from deep_research import configuration as cfg
from deep_research.agents.rag_expert import build_rag_expert
from deep_research.agents.researcher import build_researcher
from deep_research.agents.sql_expert import build_sql_expert
from deep_research.llm import get_model
from deep_research.memory.store import format_memory
from deep_research.memory.tools import delete_memory, save_memory
from deep_research.prompts import SUPERVISOR_PROMPT


def build_research_agent(checkpointer=None):
    """编译完整多智能体图。checkpointer 供多轮会话使用。"""
    return create_deep_agent(
        model=get_model("research"),
        system_prompt=SUPERVISOR_PROMPT.format(user_memory=format_memory(), today=date.today().isoformat()),
        subagents=[build_researcher(), build_rag_expert(), build_sql_expert()],
        # 长期记忆：SQLite 结构化条目（memory/store.py），读取侧烘进提示词，
        # 写入侧走 save_memory/delete_memory 工具（勿用 deepagents memory= 中间件，会双重注入）
        tools=[save_memory, delete_memory],
        backend=FilesystemBackend(root_dir=str(cfg.ROOT)),  # write_file 用于报告落盘
        checkpointer=checkpointer,
    )
