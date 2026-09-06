"""FastAPI 服务层：研究任务后台执行 + SSE 事件流 + 追问 + 报告下载。

启动: uv run uvicorn src.deep_research.api.app:app --port 8000
"""
import json
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from deep_research import configuration as cfg
from deep_research.graph import build
from deep_research.main import TOOL_LABELS

app = FastAPI(title="Deep Research Agents", version="0.1.0")

CHECKPOINTER = InMemorySaver()  # 进程级共享：同一 thread_id 的研究与会话上下文互通
RUNS: dict[str, dict] = {}  # thread_id -> {"queue": Queue, "done": bool}


class TopicIn(BaseModel):
    topic: str


class ChatIn(BaseModel):
    thread_id: str
    message: str


def _drain_events(agent, config: dict, q: queue.Queue) -> None:
    """跑一轮研究，节点事件与最终回复推入队列；结束时推 None 哨兵。"""
    try:
        for chunk in agent.stream({"messages": [("user", config["topic"])]}, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node != "model":
                    continue
                for m in update.get("messages", []):
                    for tc in getattr(m, "tool_calls", None) or []:
                        label = TOOL_LABELS.get(tc.get("name"), lambda a: tc.get("name"))(tc.get("args", {}))
                        q.put({"type": "tool", "label": label})
        state = agent.get_state(config)
        final = next(
            (m for m in reversed(state.values.get("messages", [])) if getattr(m, "type", "") == "ai" and m.content),
            None,
        )
        content = final.content if final and isinstance(final.content, str) else ""
        q.put({"type": "final", "content": content})
    except Exception as e:
        q.put({"type": "error", "content": f"{type(e).__name__}: {e}"})
    finally:
        q.put(None)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "runs": len(RUNS)}


@app.post("/api/research")
def start_research(body: TopicIn) -> dict:
    """创建研究任务：立即返回 thread_id，事件通过 /stream 获取。"""
    thread_id = f"r-{uuid.uuid4().hex[:8]}"
    RUNS[thread_id] = {"queue": queue.Queue(), "done": False}
    agent = build(CHECKPOINTER)
    config = {"configurable": {"thread_id": thread_id}, "topic": body.topic}
    threading.Thread(target=_drain_events, args=(agent, config, RUNS[thread_id]["queue"]), daemon=True).start()
    return {"thread_id": thread_id, "status": "started"}


@app.get("/api/research/{thread_id}/stream")
def stream_research(thread_id: str):
    """SSE：逐条推送节点事件，final 为最终回复，error 为失败信息。"""
    run = RUNS.get(thread_id)
    if run is None:
        raise HTTPException(404, "任务不存在")

    def gen():
        while True:
            try:
                item = run["queue"].get(timeout=300)
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'timeout'}, ensure_ascii=False)}\n\n"
                break
            if item is None:
                yield "data: [DONE]\n\n"
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        run["done"] = True

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/chat")
def chat(body: ChatIn) -> dict:
    """同 thread 同步追问：checkpointer 延续上下文，直接返回最终回答。"""
    if body.thread_id not in RUNS:
        raise HTTPException(404, "会话不存在")
    agent = build(CHECKPOINTER)
    config = {"configurable": {"thread_id": body.thread_id}}
    agent.invoke({"messages": [("user", body.message)]}, config)
    state = agent.get_state(config)
    final = next(
        (m for m in reversed(state.values.get("messages", [])) if getattr(m, "type", "") == "ai" and m.content),
        None,
    )
    if final is None:
        raise HTTPException(502, "没有得到回复")
    content = final.content if isinstance(final.content, str) else ""
    return {"thread_id": body.thread_id, "answer": content}


@app.get("/api/reports")
def list_reports() -> list[dict]:
    return [
        {"name": f.name, "size_kb": f.stat().st_size // 1024}
        for f in sorted(cfg.REPORTS_DIR.glob("*.md"))
    ]


@app.get("/api/reports/{name}")
def get_report(name: str) -> dict:
    if "/" in name or ".." in name:  # 路径穿越防护
        raise HTTPException(400, "非法文件名")
    path = cfg.REPORTS_DIR / name
    if not path.is_file():
        raise HTTPException(404, "报告不存在")
    return {"name": name, "content": path.read_text(encoding="utf-8")}
