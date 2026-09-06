"""Streamlit Web UI：会话 + 节点进度 + 报告预览 + 记忆查看。

启动: uv run streamlit run webui/app.py --server.port 8501
"""
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from deep_research.graph import build
from deep_research.main import TOOL_LABELS
from deep_research.memory.store import read_memory

st.set_page_config(page_title="Deep Research Agents", page_icon="🔬", layout="wide")


def new_thread() -> None:
    st.session_state.thread_id = f"r-{uuid.uuid4().hex[:8]}"
    st.session_state.agent = build()
    st.session_state.messages = []


if "thread_id" not in st.session_state:
    new_thread()

AGENT = st.session_state.agent
CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("🔬 深度研究多智能体系统")
st.caption(f"thread: {st.session_state.thread_id} | supervisor + web-researcher / rag-expert / sql-expert")

with st.sidebar:
    st.header("控制台")
    if st.button("🆕 开启新研究（新会话）"):
        new_thread()
        st.rerun()
    st.caption("⚠️ 运行中请勿刷新页面，否则当前任务会中断")
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
        if f.name[0].isdigit():  # 评测报告（eval_*）不列入
            st.write(f.name)

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)


def run_in_background(prompt: str, q: queue.Queue) -> None:
    """在后台线程跑一轮研究，事件与最终回复推入队列（UI 每秒轮询，避免长时间静默）。"""
    try:
        for chunk in AGENT.stream({"messages": [("user", prompt)]}, CONFIG, stream_mode="updates"):
            for node, update in chunk.items():
                if node != "model":
                    continue
                for m in update.get("messages", []):
                    for tc in getattr(m, "tool_calls", None) or []:
                        label = TOOL_LABELS.get(tc.get("name"), lambda a: tc.get("name"))(tc.get("args", {}))
                        q.put(("tool", label))
        state = AGENT.get_state(CONFIG)
        final = next(
            (m for m in reversed(state.values.get("messages", [])) if getattr(m, "type", "") == "ai" and m.content),
            None,
        )
        content = final.content if final and isinstance(final.content, str) else "（没有得到回复）"
        q.put(("final", content))
    except Exception as e:
        q.put(("error", f"**出错了**：{type(e).__name__}: {e}"))


if prompt := st.chat_input("输入研究主题，或针对当前研究追问…"):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        q: queue.Queue = queue.Queue()
        threading.Thread(target=run_in_background, args=(prompt, q), daemon=True).start()
        start = time.time()
        deadline = start + 480  # 总保险丝：内部 HTTP 客户端偶发绕过超时，8 分钟强制终止
        with st.status("supervisor 规划中…", expanded=True) as status:
            final_content = None
            timed_out = False
            while True:
                try:
                    kind, payload = q.get(timeout=1)
                except queue.Empty:
                    elapsed = int(time.time() - start)
                    if time.time() > deadline:
                        timed_out = True
                        final_content = (
                            "**任务超时终止**：内部调用长时间无响应（通常是网关拥堵时段）。"
                            "请点击重试或稍后再试；也可切换 .env 里的模型。"
                        )
                        break
                    status.update(label=f"子智能体执行中… 已 {elapsed}s（检索/生成期间无中间事件，非卡死）")
                    continue
                if kind == "tool":
                    status.write(f"• {payload}")
                elif kind == "final":
                    final_content = payload
                    break
                elif kind == "error":
                    final_content = payload
                    status.update(label=f"⚠️ 调用失败（用时 {int(time.time() - start)}s）", state="error", expanded=True)
                    break
            if not timed_out and final_content and not final_content.startswith("**出错了**"):
                status.update(
                    label=f"完成 ✅（用时 {int(time.time() - start)}s）",
                    state="complete",
                    expanded=False,
                )
        st.markdown(final_content)
        st.session_state.messages.append(("assistant", final_content))
