"""FastAPI 服务层：研究任务后台执行 + SSE 事件流 + 追问 + 报告下载。

启动: uv run uvicorn src.deep_research.api.app:app --port 8000
"""
import json
import logging
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from deep_research import configuration as cfg
from deep_research.api import run_store
from deep_research.events import final_content, iter_tool_labels, new_thread_id
from deep_research.graph import build, get_checkpointer
from deep_research.memory import store

RUN_STALE_SECONDS = 1800  # 已结束的任务记录保留 30 分钟，避免 runs/events 无限增长
RUN_PURGE_SECONDS = 300  # 后台清理线程周期
EVENT_POLL_SECONDS = 0.25
RUNS_LOCK = threading.RLock()
logger = logging.getLogger("deep_research.api")


def _purge_loop(stop: threading.Event) -> None:
    while not stop.wait(RUN_PURGE_SECONDS):
        run_store.purge_stale_runs(RUN_STALE_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时恢复未完成任务，并定时清理过期 run/event 记录。"""
    run_store.init_db()
    run_store.recover_running_runs()
    stop = threading.Event()
    thread = threading.Thread(target=_purge_loop, args=(stop,), daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


app = FastAPI(title="Deep Research Agents", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    if not cfg.API_KEY or not request.url.path.startswith("/api"):
        return await call_next(request)
    supplied = request.headers.get("X-API-Key", "")
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not secrets.compare_digest(supplied, cfg.API_KEY):
        return JSONResponse(status_code=401, content={"detail": "API Key 无效"})
    return await call_next(request)


class TopicIn(BaseModel):
    topic: str = Field(min_length=1, max_length=5000)

    @field_validator("topic")
    @classmethod
    def _strip_topic(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("topic 不能为空")
        return v


class ChatIn(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=5000)

    @field_validator("message")
    @classmethod
    def _strip_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message 不能为空")
        return v


def _report_path(name: str) -> Path:
    """解析报告文件名，拒绝绝对路径、目录跳转和非 Markdown 文件。"""
    candidate_name = Path(name)
    if (
        "\\" in name  # Windows 分隔符（反斜杠不受 Path.name 与 / 校验覆盖）
        or not name
        or candidate_name.name != name
        or candidate_name.suffix.lower() != ".md"
    ):
        raise HTTPException(400, "非法报告文件名")

    root = cfg.REPORTS_DIR.resolve()
    path = (root / candidate_name).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise HTTPException(400, "非法报告路径") from None
    return path


def _drain_events(agent, config: dict, thread_id: str, topic: str) -> None:
    """跑一轮研究并把节点事件持久化；SSE 消费者从事件表按 ID 重放。"""
    started = time.time()
    steps = input_tokens = output_tokens = 0
    error = None
    cancelled = False
    try:
        for chunk in agent.stream({"messages": [("user", topic)]}, config, stream_mode="updates"):
            if run_store.is_cancel_requested(thread_id):
                cancelled = True
                run_store.append_event(thread_id, {"type": "cancelled", "content": "任务已取消"})
                break
            for node, update in chunk.items():
                if node != "model":
                    continue
                steps += 1
                for m in update.get("messages", []):
                    usage = getattr(m, "usage_metadata", None) or {}
                    input_tokens += usage.get("input_tokens") or 0
                    output_tokens += usage.get("output_tokens") or 0
                for label in iter_tool_labels(update):
                    run_store.append_event(thread_id, {"type": "tool", "label": label})
            if run_store.is_cancel_requested(thread_id):
                cancelled = True
                run_store.append_event(thread_id, {"type": "cancelled", "content": "任务已取消"})
                break
        if not cancelled:
            content = final_content(agent.get_state(config).values.get("messages", []))
            run_store.append_event(thread_id, {"type": "final", "content": content})
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        run_store.append_event(thread_id, {"type": "error", "content": error})
    finally:
        status = "error" if error else "cancelled" if cancelled else "done"
        run_store.finish_run(
            thread_id,
            status,
            steps=steps,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )
        logger.info(
            "run_finished thread_id=%s status=%s seconds=%.2f steps=%d input_tokens=%d output_tokens=%d error=%s",
            thread_id,
            status,
            time.time() - started,
            steps,
            input_tokens,
            output_tokens,
            error or "",
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "runs": run_store.stats()["active_runs"]}


@app.get("/api/stats")
def stats() -> dict:
    """持久化运行统计：任务数/工具调用/LLM token/耗时/错误。"""
    return run_store.stats()


@app.post("/api/research")
def start_research(
    body: TopicIn,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict:
    """创建研究任务：立即返回 thread_id，事件通过 /stream 获取。"""
    run_store.purge_stale_runs(RUN_STALE_SECONDS)
    agent = build(get_checkpointer())
    proposed_id = new_thread_id()
    with RUNS_LOCK:
        thread_id, created = run_store.create_run(proposed_id, body.topic, idempotency_key)
        if not created:
            existing = run_store.get_run(thread_id)
            if existing is not None and existing.topic != body.topic:
                raise HTTPException(409, "Idempotency-Key 已用于其他 topic")
            return {"thread_id": thread_id, "status": existing.status if existing else "unknown", "idempotent": True}
    config = {"configurable": {"thread_id": thread_id}}
    threading.Thread(target=_drain_events, args=(agent, config, thread_id, body.topic), daemon=True).start()
    logger.info("run_started thread_id=%s topic_chars=%d", thread_id, len(body.topic))
    return {"thread_id": thread_id, "status": "started", "idempotent": False}


@app.get("/api/research/{thread_id}/stream")
def stream_research(
    thread_id: str,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    after: int = 0,
):
    """SSE：事件带递增 ID；断线后可通过 Last-Event-ID 从断点继续。"""
    if run_store.get_run(thread_id) is None:
        raise HTTPException(404, "任务不存在")
    try:
        cursor = int(last_event_id or after or 0)
    except ValueError:
        raise HTTPException(400, "Last-Event-ID 必须为整数") from None

    def gen():
        nonlocal cursor
        last_activity = last_ping = time.monotonic()
        while True:
            events = run_store.list_events(thread_id, after_id=cursor)
            if events:
                for event in events:
                    cursor = event["id"]
                    payload = {key: value for key, value in event.items() if key != "id"}
                    yield f"id: {cursor}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_activity = last_ping = time.monotonic()
                continue

            run = run_store.get_run(thread_id)
            if run is None:
                break
            if run.status != "running":
                yield "data: [DONE]\n\n"
                break

            now = time.monotonic()
            if now - last_activity >= cfg.STREAM_IDLE_TIMEOUT:
                yield f"data: {json.dumps({'type': 'timeout'}, ensure_ascii=False)}\n\n"
                break
            if now - last_ping >= 15:
                yield f"data: {json.dumps({'type': 'ping'}, ensure_ascii=False)}\n\n"
                last_ping = now
            time.sleep(EVENT_POLL_SECONDS)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/research/{thread_id}")
def get_run_status(thread_id: str) -> dict:
    run = run_store.get_run(thread_id)
    if run is None:
        raise HTTPException(404, "任务不存在")
    return {
        "thread_id": run.thread_id,
        "topic": run.topic,
        "status": run.status,
        "steps": run.steps,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "error": run.error,
        "cancel_requested": run.cancel_requested,
    }


@app.post("/api/research/{thread_id}/cancel")
def cancel_research(thread_id: str) -> dict:
    if run_store.get_run(thread_id) is None:
        raise HTTPException(404, "任务不存在")
    if not run_store.request_cancel(thread_id):
        raise HTTPException(409, "任务已结束，无法取消")
    return {"thread_id": thread_id, "status": "cancel_requested"}


@app.post("/api/chat")
def chat(body: ChatIn) -> dict:
    """同 thread 追问：转后台执行，复用 /stream 事件链路（心跳/进度/断流保护全生效）。

    会话存在性以 checkpointer 为准（历史消息在即有效），不受 run 记录清理影响。
    同一 thread 的上一轮任务执行中时拒绝并发追问，避免并发写入同一图状态。
    """
    agent = build(get_checkpointer())
    config = {"configurable": {"thread_id": body.thread_id}}
    if not agent.get_state(config).values.get("messages"):
        raise HTTPException(404, "会话不存在（该线程没有历史上下文）")
    run_store.purge_stale_runs(RUN_STALE_SECONDS)
    with RUNS_LOCK:
        existing = run_store.get_run(body.thread_id)
        if existing and existing.status == "running":
            raise HTTPException(409, "该会话仍有任务执行中")
        run_store.create_run(body.thread_id, body.message)
    threading.Thread(target=_drain_events, args=(agent, config, body.thread_id, body.message), daemon=True).start()
    logger.info("run_started thread_id=%s topic_chars=%d", body.thread_id, len(body.message))
    return {"thread_id": body.thread_id, "status": "started"}


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
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        {"name": f.name, "size_kb": f.stat().st_size // 1024}
        for f in sorted(cfg.REPORTS_DIR.glob("*.md"))
    ]


@app.get("/api/reports/{name}")
def get_report(name: str) -> dict:
    path = _report_path(name)
    if not path.is_file():
        raise HTTPException(404, "报告不存在")
    return {"name": name, "content": path.read_text(encoding="utf-8")}


@app.get("/api/reports/{name}/raw")
def download_report(name: str) -> FileResponse:
    """Markdown 原文件下载。"""
    path = _report_path(name)
    if not path.is_file():
        raise HTTPException(404, "报告不存在")
    return FileResponse(path, filename=name, media_type="text/markdown")
