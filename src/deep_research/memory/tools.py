"""Supervisor 的长期记忆工具：记忆读写从"编辑文件"收拢为结构化数据库操作。

这两个工具通过 create_deep_agent(tools=[...]) 挂到 supervisor 上，
容量上限与自动淘汰策略在 memory/store.py 内实现。
"""
from langchain_core.tools import tool

from deep_research.memory import store


@tool
def save_memory(category: str, content: str) -> str:
    """保存一条长期记忆，跨会话自动生效。category 只能是 "偏好" 或 "历史研究结论"。

    content 必须是一条完整、自包含的记录（用户偏好或可复用的重要研究结论），
    不要写流水账；过时或被推翻的旧记录应配合 delete_memory 清理。
    """
    try:
        entry_id, note = store.add_entry(category, content)
        return f"已保存记忆 #{entry_id}（{category}）{note}"
    except (ValueError, RuntimeError) as e:
        return f"保存失败：{e}"


@tool
def delete_memory(memory_id: int) -> str:
    """删除指定编号的长期记忆条目（编号即 <user_memory> 中各条目前面的 [id]）。"""
    return f"已删除记忆 #{memory_id}" if store.delete_entry(memory_id) else f"记忆 #{memory_id} 不存在"
