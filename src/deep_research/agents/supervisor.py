"""主智能体（Supervisor）：deepagents harness + 3 个声明式 subagent + 记忆工具。"""
from datetime import date

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware

from deep_research import configuration as cfg
from deep_research.agents.rag_expert import build_rag_expert
from deep_research.agents.researcher import build_researcher
from deep_research.agents.sql_expert import build_sql_expert
from deep_research.llm import get_model
from deep_research.memory import store
from deep_research.memory.tools import delete_memory, save_memory
from deep_research.prompts import SUPERVISOR_PROMPT

_CACHE: dict = {"agent": None, "mem_version": None, "checkpointer": None}


def build_research_agent(checkpointer=None):
    """编译完整多智能体图。checkpointer 供多轮会话使用。

    编译结果按「记忆版本 + checkpointer 实例」缓存：记忆烘在 system_prompt 里，
    仅当记忆库版本变化（或 checkpointer 变更）时才重建，避免每请求重复编译。
    """
    version = store.version()
    if (
        _CACHE["agent"] is not None
        and _CACHE["mem_version"] == version
        and _CACHE["checkpointer"] is checkpointer
    ):
        return _CACHE["agent"]

    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fs_backend = FilesystemBackend(root_dir=str(cfg.REPORTS_DIR))
    agent = create_deep_agent(
        model=get_model("research"),
        system_prompt=SUPERVISOR_PROMPT.format(user_memory=store.format_memory(), today=date.today().isoformat()),
        subagents=[build_researcher(), build_rag_expert(), build_sql_expert()],
        # 长期记忆：SQLite 结构化条目（memory/store.py），读取侧烘进提示词，
        # 写入侧走 save_memory/delete_memory 工具（勿用 deepagents memory= 中间件，会双重注入）
        tools=[save_memory, delete_memory],
        # 文件系统根目录严格限制为 reports/，且只暴露 read_file/write_file
        # （FilesystemMiddleware 强制要求 read_file 随附）：模型无法浏览目录、
        # 编辑或删除 reports 之外的内容（含源码、配置、密钥）。
        backend=fs_backend,
        middleware=[FilesystemMiddleware(backend=fs_backend, tools=["read_file", "write_file"])],
        checkpointer=checkpointer,
    )
    _CACHE.update(agent=agent, mem_version=version, checkpointer=checkpointer)
    return agent
