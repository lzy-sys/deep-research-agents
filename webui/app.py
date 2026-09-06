"""Streamlit Web UI：纯前端客户端，执行全部委托给 FastAPI 服务（进程隔离，UI 不受 SDK 不稳定影响）。

启动: uv run streamlit run webui/app.py --server.port 8501
依赖: API 服务已启动（uv run uvicorn src.deep_research.api.app:app --port 18000 或 docker compose up）
"""
import json
import os
import queue
import sys
import threading
import time

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import httpx
import streamlit as st

from deep_research import configuration as cfg

API_BASE = os.getenv("API_BASE_URL", "http://localhost:18000")

st.set_page_config(page_title="Deep Research Agents", page_icon="🔬", layout="wide")


def api_get(path: str, timeout: float = 10):
    try:
        return httpx.get(f"{API_BASE}{path}", timeout=timeout).json()
    except Exception:
        return None


def api_post(path: str, payload: dict | None = None, timeout: float = 600) -> httpx.Response:
    return httpx.post(f"{API_BASE}{path}", json=payload, timeout=timeout)


@st.cache_data(ttl=30, show_spinner=False)
def list_reports_cached() -> list | None:
    try:
        return httpx.get(f"{API_BASE}/api/reports", timeout=5).json()
    except Exception:
        return None


if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
    st.session_state.messages = []


def new_thread() -> None:
    st.session_state.thread_id = None  # 首条消息提交时由 API 创建
    st.session_state.messages = []


st.title("🔬 深度研究多智能体系统")
thread_label = st.session_state.thread_id or "（首次提问时创建）"
st.caption(f"thread: {thread_label} | supervisor + web-researcher / rag-expert / sql-expert | API: {API_BASE}")

with st.sidebar:
    st.header("控制台")
    if st.button("🆕 开启新研究（新会话）"):
        new_thread()
        st.rerun()
    st.caption("任务在 API 服务进程执行，刷新页面不会中断；页面仅展示结果")
    st.divider()
    st.subheader("🧠 长期记忆 (AGENTS.md)")
    mem = api_get("/api/memory")
    st.text(mem["content"][:800] if mem else "（API 未启动，无法读取）")
    if st.button("🗑️ 重置长期记忆"):
        st.session_state.confirm_reset = True
    if st.session_state.get("confirm_reset"):
        st.warning("确定清空全部长期记忆？此操作不可恢复。")
        c1, c2 = st.columns(2)
        if c1.button("⚠️ 确认清空"):
            try:
                api_post("/api/memory/reset", {}, timeout=10)
                st.success("已重置")
            except Exception:
                st.error("API 未启动，重置失败")
            st.session_state.confirm_reset = False
        if c2.button("取消"):
            st.session_state.confirm_reset = False
            st.rerun()
    st.divider()
    st.subheader("📄 研究报告")
    reports = list_reports_cached()  # 缓存 30s：侧栏每次 rerun 不再同步请求 API
    if reports is None:
        st.error("API 服务未启动\n\n请先运行:\nuv run uvicorn src.deep_research.api.app:app --port 18000")
    elif reports:
        names = [r["name"] for r in reports]
        st.caption(f"共 {len(names)} 份")
        with st.expander("📖 预览报告"):
            pick = st.selectbox("选择报告", names)
            detail = api_get(f"/api/reports/{pick}")
            if detail:
                st.markdown(detail["content"][:4000])
                if "localhost" in API_BASE or "127.0.0.1" in API_BASE:
                    # docker 部署时 API_BASE 是容器内地址，浏览器不可达，仅本地模式提供直链
                    st.markdown(f"[⬇️ 下载原文件]({API_BASE}/api/reports/{pick}/raw)")
    else:
        st.caption("还没有报告")


for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)


def sse_reader(thread_id: str, q: queue.Queue) -> None:
    """后台线程：读 API 的 SSE 事件流，推入队列（UI 每秒轮询渲染心跳）。"""
    terminal = False
    try:
        with httpx.stream("GET", f"{API_BASE}/api/research/{thread_id}/stream", timeout=cfg.STREAM_IDLE_TIMEOUT) as s:
            for line in s.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    terminal = True
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "tool":
                    q.put(("tool", event["label"]))
                elif event.get("type") == "final":
                    q.put(("final", event["content"]))
                    terminal = True
                    break
                elif event.get("type") == "error":
                    q.put(("error", f"**任务失败**：{event['content']}"))
                    terminal = True
                    break
                elif event.get("type") == "timeout":
                    q.put(("error", "**任务超时**：API 等待事件超时，请重试"))
                    terminal = True
                    break
                elif event.get("type") == "ping":
                    continue  # API 心跳：仅保活连接，UI 本地每秒刷新计时
    except Exception as e:
        q.put(("error", f"**连接 API 失败**：{type(e).__name__}: {e}\n\n请确认 API 服务已启动。"))
        return
    if not terminal:  # 流被静默掐断（如服务重启）：必须终止 UI 循环，否则永远转圈
        q.put(("error", "**连接中断**：事件流意外断开（任务可能仍在服务端执行）。请点「开启新研究」重试。"))


def run_research(prompt: str):
    """新研究：创建任务 + SSE 渲染。返回最终内容。"""
    with st.status("创建研究任务…", expanded=True) as status:
        try:
            r = api_post("/api/research", {"topic": prompt}, timeout=30)
            r.raise_for_status()
        except Exception as e:
            status.update(label="⚠️ 无法连接 API 服务", state="error", expanded=True)
            st.markdown(
                f"**连接失败**：{type(e).__name__}: {e}\n\n"
                f"请先启动 API 服务：\n`uv run uvicorn src.deep_research.api.app:app --port 18000`"
            )
            return None
        thread_id = r.json()["thread_id"]
        st.session_state.thread_id = thread_id
        status.write(f"任务已创建: {thread_id}")

    q: queue.Queue = queue.Queue()
    threading.Thread(target=sse_reader, args=(thread_id, q), daemon=True).start()
    start = time.time()
    with st.status("子智能体执行中…", expanded=True) as status:
        final_content = None
        while True:
            try:
                kind, payload = q.get(timeout=1)
            except queue.Empty:
                elapsed = int(time.time() - start)
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
        status.update(label=f"完成 ✅（用时 {int(time.time() - start)}s）", state="complete", expanded=False)
    return final_content


def run_followup(prompt: str):
    """追问：同 thread 同步调用（API 进程内执行，UI 只等结果）。"""
    with st.status("回答中…（基于当前研究上下文）", expanded=False) as status:
        try:
            r = api_post("/api/chat", {"thread_id": st.session_state.thread_id, "message": prompt})
            r.raise_for_status()
            content = r.json()["answer"]
            status.update(label="完成 ✅", state="complete", expanded=False)
        except Exception as e:
            content = f"**调用失败**：{type(e).__name__}: {e}\n\n请确认 API 服务已启动。"
            status.update(label="⚠️ 调用失败", state="error", expanded=True)
    return content


if prompt := st.chat_input("输入研究主题，或针对当前研究追问…"):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.thread_id is None:
            answer = run_research(prompt)
        else:
            answer = run_followup(prompt)
        st.markdown(answer)
        st.session_state.messages.append(("assistant", answer))
