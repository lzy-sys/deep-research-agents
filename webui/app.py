"""Streamlit Web UI：会话 + 节点进度 + 报告预览 + 记忆查看。

启动: uv run streamlit run webui/app.py --server.port 8501
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from deep_research.graph import build
from deep_research.main import TOOL_LABELS
from deep_research.memory.store import read_memory

st.set_page_config(page_title="Deep Research Agents", page_icon="🔬", layout="wide")

if "thread_id" not in st.session_state:
    import uuid

    st.session_state.thread_id = f"r-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []
    st.session_state.agent = build()  # 进程内复用同一 agent + checkpointer

AGENT = st.session_state.agent
CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("🔬 深度研究多智能体系统")
st.caption(f"thread: {st.session_state.thread_id} | supervisor + web-researcher / rag-expert / sql-expert")

with st.sidebar:
    st.header("控制台")
    if st.button("🆕 开启新研究（新会话）"):
        import uuid

        st.session_state.thread_id = f"r-{uuid.uuid4().hex[:8]}"
        st.session_state.agent = build()
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.subheader("🧠 长期记忆 (AGENTS.md)")
    st.text(read_memory()[:800])
    if st.button("🗑️ 重置长期记忆"):
        from deep_research.memory.store import reset_memory

        reset_memory()
        st.session_state.agent = build()
        st.success("已重置")
    st.divider()
    st.subheader("📄 研究报告")
    for f in sorted(Path("reports").glob("*.md")):
        st.write(f.name)

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("输入研究主题，或针对当前研究追问…"):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("supervisor 规划中…", expanded=True)
        answer_box = st.empty()
        final_content = ""
        try:
            for chunk in AGENT.stream({"messages": [("user", prompt)]}, CONFIG, stream_mode="updates"):
                for node, update in chunk.items():
                    if node != "model":
                        continue
                    for m in update.get("messages", []):
                        for tc in getattr(m, "tool_calls", None) or []:
                            label = TOOL_LABELS.get(tc.get("name"), lambda a: tc.get("name"))(tc.get("args", {}))
                            status.update(label=f"  • {label}")
            state = AGENT.get_state(CONFIG)
            final = next(
                (m for m in reversed(state.values.get("messages", [])) if getattr(m, "type", "") == "ai" and m.content),
                None,
            )
            final_content = final.content if final and isinstance(final.content, str) else "（没有得到回复）"
        except Exception as e:
            final_content = f"**出错了**：{type(e).__name__}: {e}"
        status.update(label="完成 ✅", state="complete", expanded=False)
        answer_box.markdown(final_content)
        st.session_state.messages.append(("assistant", final_content))
