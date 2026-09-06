"""节点事件与最终回复的统一抽取：CLI 与 API 服务共用，消除双份逻辑。"""
from uuid import uuid4

TOOL_LABELS = {
    "task": lambda a: f"派发 → {a.get('subagent_type', '?')}",
    "web_search": lambda a: f"web 搜索 → {a.get('query', '')}",
    "web_search_deep": lambda a: f"深搜 → {a.get('query', '')}",
    "retrieve_docs": lambda a: f"知识库检索 → {a.get('question', '')}",
    "run_sql": lambda a: f"SQL → {str(a.get('sql', ''))[:80]}",
    "write_todos": lambda a: "更新任务清单",
    "write_file": lambda a: f"写文件 → {a.get('path', a.get('file_path', ''))}",
    "edit_file": lambda a: f"编辑文件 → {a.get('path', a.get('file_path', ''))}",
    "save_memory": lambda a: f"记忆保存（{a.get('category', '?')}）",
    "delete_memory": lambda a: f"记忆删除 #{a.get('memory_id', '?')}",
}


def new_thread_id() -> str:
    return f"r-{uuid4().hex[:8]}"


def iter_tool_labels(update: dict):
    """从单个节点的 update 中产出工具调用的展示标签。"""
    for m in update.get("messages", []):
        for tc in getattr(m, "tool_calls", None) or []:
            label = TOOL_LABELS.get(tc.get("name"), lambda a: tc.get("name"))
            yield label(tc.get("args", {}))


def final_content(messages: list) -> str:
    """取最后一条有内容的 AI 消息文本；content 为分块列表时拼接文本块。"""
    final = next(
        (m for m in reversed(messages) if getattr(m, "type", "") == "ai" and m.content),
        None,
    )
    if final is None:
        return ""
    if isinstance(final.content, str):
        return final.content
    return "\n".join(p.get("text", "") for p in final.content if isinstance(p, dict))
