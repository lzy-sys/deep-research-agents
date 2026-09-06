"""AGENTS.md 记忆文件读写：查看 / 重置 / 初始引导。"""
from deep_research import configuration as cfg
from deep_research.prompts import MEMORY_EXTRACT_HINT

MEMORY_PATH = cfg.ROOT / "AGENTS.md"

INITIAL_CONTENT = f"""# AGENTS.md — 长期记忆

{MEMORY_EXTRACT_HINT}

## 用户偏好

（暂无，会话中得知后自动更新）

## 历史研究结论

（暂无）
"""


def read_memory() -> str:
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text(INITIAL_CONTENT, encoding="utf-8")
    return MEMORY_PATH.read_text(encoding="utf-8")


def reset_memory() -> None:
    MEMORY_PATH.write_text(INITIAL_CONTENT, encoding="utf-8")
