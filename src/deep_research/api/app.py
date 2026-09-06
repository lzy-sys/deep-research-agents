"""FastAPI 服务层：研究任务后台执行 + SSE 事件流 + 追问 + 报告下载。

启动: uv run uvicorn src.deep_research.api.app:app --port 8000
"""
import json
import queue
import threading
import time
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from deep_research import configuration as cfg
from deep_research.events import final_content, iter_tool_labels, new_thread_id
from deep_research.graph import build
from deep_research.memory import store

app = FastAPI(title="Deep Research Agents", version="0.1.0")

CHECKPOINTER = InMemorySaver()  # 进程级共享：同一 thread_id 的研究与会话上下文互通
RUN_STALE_SECONDS = 1800  # 已结束的任务记录保留 30 分钟，过期清理防 RUNS 无限增长


@dataclass
class Run:
    """一次研究任务：事件队列 + 主题 + 生命周期标记。"""

    queue: queue.Queue
    topic: str
    done: bool = False
    done_at: float | None = None  # 图执行结束时间，用于过期清理


RUNS: dict[str, Run] = {}


class TopicIn(BaseModel):
    topic: str


class ChatIn(BaseModel):
    thread_id: str
    message: str


def _purge_stale_runs() -> None:
    now = time.time()
    for tid in [t for t, r in RUNS.items() if r.done_at and now - r.done_at > RUN_STALE_SECONDS]:
        del RUNS[tid]


def _drain_events(agent, config: dict, run: Run) -> None:
    """跑一轮研究，节点事件与最终回复推入队列；结束时推 None 哨兵。"""
    try:
        for chunk in agent.stream({"messages": [("user", run.topic)]}, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node != "model":
                    continue
                for label in iter_tool_labels(update):
                    run.queue.put({"type": "tool", "label": label})
        run.queue.put(
            {"type": "final", "content": final_content(agent.get_state(config).values.get("messages", []))}
        )
    except Exception as e:
        run.queue.put({"type": "error", "content": f"{type(e).__name__}: {e}"})
    finally:
        run.done_at = time.time()
        run.queue.put(None)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "runs": len(RUNS)}


@app.post("/api/research")
def start_research(body: TopicIn) -> dict:
    """创建研究任务：立即返回 thread_id，事件通过 /stream 获取。"""
    _purge_stale_runs()
    thread_id = new_thread_id()
    run = Run(queue=queue.Queue(), topic=body.topic)
    RUNS[thread_id] = run
    agent = build(CHECKPOINTER)
    config = {"configurable": {"thread_id": thread_id}}
    threading.Thread(target=_drain_events, args=(agent, config, run), daemon=True).start()
    return {"thread_id": thread_id, "status": "started"}


@app.get("/api/research/{thread_id}/stream")
def stream_research(thread_id: str):
    """SSE：逐条推送节点事件，final 为最终回复，error 为失败信息。"""
    run = RUNS.get(thread_id)
    if run is None:
        raise HTTPException(404, "任务不存在")

    def gen():
        idle = 0
        while True:
            try:
                item = run.queue.get(timeout=15)
                idle = 0
            except queue.Empty:
                idle += 15
                if idle >= cfg.STREAM_IDLE_TIMEOUT:  # 长时间无事件才判超时（LLM 慢≠死，ping 保活）
                    yield f"data: {json.dumps({'type': 'timeout'}, ensure_ascii=False)}\n\n"
                    break
                yield f"data: {json.dumps({'type': 'ping'}, ensure_ascii=False)}\n\n"
                continue
            if item is None:
                yield "data: [DONE]\n\n"
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        run.done = True

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/chat")
def chat(body: ChatIn) -> dict:
    """同 thread 同步追问：checkpointer 延续上下文，直接返回最终回答。"""
    if body.thread_id not in RUNS:
        raise HTTPException(404, "会话不存在")
    agent = build(CHECKPOINTER)
    config = {"configurable": {"thread_id": body.thread_id}}
    agent.invoke({"messages": [("user", body.message)]}, config)
    answer = final_content(agent.get_state(config).values.get("messages", []))
    if not answer:
        raise HTTPException(502, "没有得到回复")
    return {"thread_id": body.thread_id, "answer": answer}


@app.get("/api/memory")
def get_memory() -> dict:
    """长期记忆条目（SQLite 结构化存储），UI 逐条展示与删除。"""
    entries = store.list_entries()
    return {"entries": entries, "total": len(entries)}


@app.delete("/api/memory/{entry_id}")
def remove_memory(entry_id: int) -> dict:
    if not store.delete_entry(entry_id):
        raise HTTPException(404, "条目不存在")
    return {"status": "deleted", "id": entry_id}


@app.post("/api/memory/reset")
def memory_reset() -> dict:
    removed = store.reset_memory()
    return {"status": "reset", "removed": removed}


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


@app.get("/api/reports/{name}/raw")
def download_report(name: str) -> FileResponse:
    """Markdown 原文件下载。"""
    if "/" in name or ".." in name:
        raise HTTPException(400, "非法文件名")
    path = cfg.REPORTS_DIR / name
    if not path.is_file():
        raise HTTPException(404, "报告不存在")
    return FileResponse(path, filename=name, media_type="text/markdown")
