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
from deep_research.prompts import SUPERVISOR_PROMPT


def build_research_agent(checkpointer=None):
    """编译完整多智能体图。checkpointer 供多轮会话使用。"""
    return create_deep_agent(
        model=get_model("research"),
        system_prompt=SUPERVISOR_PROMPT.format(user_memory=read_memory(), today=date.today().isoformat()),
        subagents=[build_researcher(), build_rag_expert(), build_sql_expert()],
        backend=FilesystemBackend(root_dir=str(cfg.ROOT)),  # write_file 直接落盘项目目录
        # 记忆只走 system_prompt 注入这一条路（勿再开 deepagents memory= 中间件，会双重注入）；
        # 智能体更新记忆靠 write_file/edit_file 写 AGENTS.md
        checkpointer=checkpointer,
    )
