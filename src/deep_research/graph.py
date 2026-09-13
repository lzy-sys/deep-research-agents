"""图组装入口：编译多智能体 + 共享 SqliteSaver（API/CLI 会话持久化）。"""
import sqlite3
import threading

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from deep_research import configuration as cfg
from deep_research.agents.supervisor import build_research_agent

_CHECKPOINTER: SqliteSaver | None = None
_CHECKPOINTER_LOCK = threading.Lock()


def get_checkpointer() -> SqliteSaver:
    """共享的 SQLite checkpointer（data/checkpoints.sqlite），API 与 CLI 共用。"""
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        with _CHECKPOINTER_LOCK:
            if _CHECKPOINTER is None:
                cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(
                    cfg.DATA_DIR / "checkpoints.sqlite",
                    timeout=10,
                    check_same_thread=False,
                )
                conn.execute("PRAGMA busy_timeout=10000")
                _CHECKPOINTER = SqliteSaver(conn)
    return _CHECKPOINTER


def build(checkpointer=None):
    return build_research_agent(checkpointer=checkpointer or InMemorySaver())
