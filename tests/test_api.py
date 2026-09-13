"""API 服务层测试：持久化 run/event、SSE 重放、追问、鉴权、记忆与报告端点。

用 FakeAgent 替换真实 deep agent，全程不触网、不编译图。
"""
import time

import pytest
from fastapi.testclient import TestClient

from deep_research import configuration as cfg
from deep_research.api import app as api_app
from deep_research.api import run_store
from deep_research.memory import store


class FakeMsg:
    def __init__(self, tool_calls=None, content="", type="ai"):
        self.tool_calls = tool_calls or []
        self.content = content
        self.type = type


class FakeState:
    def __init__(self, messages):
        self.values = {"messages": messages}


class FakeAgent:
    """替身 agent：stream 产出固定事件，get_state 由 known 控制会话存在性。"""

    def __init__(self, known: bool = True):
        self.known = known
        self.stream_inputs: list = []

    def stream(self, inp, config, stream_mode=None):
        self.stream_inputs.append((inp, config))
        yield {"model": {"messages": [FakeMsg(tool_calls=[{"name": "run_sql", "args": {"sql": "SELECT 1"}}])]}}
        yield {"model": {"messages": [FakeMsg(content="最终结论")]}}

    def get_state(self, config):
        if self.known:
            return FakeState([FakeMsg(content="历史消息"), FakeMsg(content="最终结论")])
        return FakeState([])  # 未知线程：空状态 → chat 应 404


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "memory.sqlite")
    monkeypatch.setattr(store, "LEGACY_MD_PATH", tmp_path / "AGENTS.md")
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "runs.sqlite")
    fake = FakeAgent()
    monkeypatch.setattr(api_app, "build", lambda checkpointer: fake)
    monkeypatch.setattr(api_app, "get_checkpointer", lambda: None)
    with TestClient(api_app.app) as client:
        yield client, fake


def _sse_events(client: TestClient, thread_id: str) -> list[dict]:
    with client.stream("GET", f"/api/research/{thread_id}/stream") as resp:
        assert resp.status_code == 200
        events = []
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                events.append({"type": "[DONE]"})
                break
            import json

            events.append(json.loads(payload))
    return events


def test_health(env):
    client, _ = env
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_research_stream_flow(env):
    client, fake = env
    thread_id = client.post("/api/research", json={"topic": "测试主题"}).json()["thread_id"]

    events = _sse_events(client, thread_id)

    assert events[0] == {"type": "tool", "label": "SQL → SELECT 1"}
    assert events[1] == {"type": "final", "content": "最终结论"}
    assert events[-1] == {"type": "[DONE]"}
    assert run_store.get_run(thread_id).status == "done"
    # 追问消息进入图输入
    assert fake.stream_inputs[0][1]["configurable"]["thread_id"] == thread_id


def test_research_purges_stale_runs(env):
    client, _ = env
    old_id = client.post("/api/research", json={"topic": "旧任务"}).json()["thread_id"]
    con = run_store._connect()
    try:
        con.execute(
            "UPDATE runs SET status='done', done_at=? WHERE thread_id=?",
            (time.time() - api_app.RUN_STALE_SECONDS - 1, old_id),
        )
        con.commit()
    finally:
        con.close()

    new_id = client.post("/api/research", json={"topic": "新任务"}).json()["thread_id"]

    assert run_store.get_run(old_id) is None  # 过期条目被清理
    assert run_store.get_run(new_id) is not None


def test_chat_unknown_thread_404(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "m.sqlite")
    monkeypatch.setattr(store, "LEGACY_MD_PATH", tmp_path / "AGENTS.md")
    fake = FakeAgent(known=False)
    monkeypatch.setattr(api_app, "build", lambda checkpointer: fake)
    with TestClient(api_app.app) as client:
        r = client.post("/api/chat", json={"thread_id": "r-ghost", "message": "追问"})
    assert r.status_code == 404
    assert "会话不存在" in r.json()["detail"]


def test_chat_known_thread_streams(env):
    client, fake = env
    thread_id = client.post("/api/research", json={"topic": "首次研究"}).json()["thread_id"]
    _sse_events(client, thread_id)  # 消费掉第一次运行的事件

    r = client.post("/api/chat", json={"thread_id": thread_id, "message": "追问内容"})
    assert r.status_code == 200
    assert r.json()["thread_id"] == thread_id

    events = _sse_events(client, thread_id)  # 追问复用同一 SSE 链路
    assert events[1] == {"type": "final", "content": "最终结论"}
    assert fake.stream_inputs[-1][0] == {"messages": [("user", "追问内容")]}


def test_chat_replaces_finished_run(env):
    client, _ = env
    thread_id = client.post("/api/research", json={"topic": "首次"}).json()["thread_id"]
    _sse_events(client, thread_id)

    client.post("/api/chat", json={"thread_id": thread_id, "message": "追问"})

    run = run_store.get_run(thread_id)
    assert run.status == "running" and run.topic == "追问"  # 新一轮事件接管该线程
    _sse_events(client, thread_id)


def test_memory_endpoints(env):
    client, _ = env
    assert client.get("/api/memory").json()["total"] == 0

    entry_id, _ = store.add_entry("偏好", "测试偏好条目")
    data = client.get("/api/memory").json()
    assert data["total"] == 1 and data["entries"][0]["content"] == "测试偏好条目"

    assert client.post("/api/memory/reset").json()["removed"] == 1

    entry_id, _ = store.add_entry("历史研究结论", "待删除条目")
    assert client.delete(f"/api/memory/{entry_id}").status_code == 200
    assert client.delete(f"/api/memory/{entry_id}").status_code == 404  # 二次删除报不存在


def test_report_endpoints_guard(env):
    client, _ = env
    assert client.get("/api/reports/不存在的报告.md").status_code == 404
    assert client.get("/api/reports/..%2Fsecret.md").status_code in (400, 404)  # 路径穿越被拒


def test_chat_rejects_concurrent_run(env):
    client, _ = env
    thread_id = client.post("/api/research", json={"topic": "首次研究"}).json()["thread_id"]
    response = client.post("/api/chat", json={"thread_id": thread_id, "message": "并发追问"})
    assert response.status_code == 409


def test_report_rejects_windows_absolute_path(env):
    client, _ = env
    assert client.get("/api/reports/C:%5CWindows%5Cwin.ini").status_code == 400
    assert client.get("/api/reports/foo%5Cbar.md").status_code == 400


def test_stats_endpoint(env):
    client, _ = env
    thread_id = client.post("/api/research", json={"topic": "统计任务"}).json()["thread_id"]
    _sse_events(client, thread_id)

    s = client.get("/api/stats").json()
    assert s["runs"] >= 1
    assert s["steps"] >= 2
    assert s["tools"].get("SQL → SELECT 1", 0) >= 1
    assert s["seconds"] >= 0


def test_stream_replay_after_last_event_id(env):
    client, _ = env
    thread_id = client.post("/api/research", json={"topic": "重放任务"}).json()["thread_id"]
    _sse_events(client, thread_id)

    with client.stream(
        "GET",
        f"/api/research/{thread_id}/stream",
        headers={"Last-Event-ID": "1"},
    ) as resp:
        events = [line[5:].strip() for line in resp.iter_lines() if line.startswith("data:")]
    assert any("最终结论" in event for event in events)
    assert all("SELECT 1" not in event for event in events)  # 已消费的事件不重复


def test_idempotency_key_returns_same_thread(env):
    client, _ = env
    headers = {"Idempotency-Key": "same-request"}
    first = client.post("/api/research", json={"topic": "幂等任务"}, headers=headers).json()
    second = client.post("/api/research", json={"topic": "幂等任务"}, headers=headers).json()
    assert first["thread_id"] == second["thread_id"]
    assert second["idempotent"] is True
    _sse_events(client, first["thread_id"])


def test_cancel_running_run(env, monkeypatch, tmp_path):
    import threading

    class BlockingAgent(FakeAgent):
        release = threading.Event()

        def stream(self, inp, config, stream_mode=None):
            yield {"model": {"messages": [FakeMsg(content="开始")]}}
            self.release.wait(timeout=3)
            yield {"model": {"messages": [FakeMsg(content="结束")]}}

    blocking = BlockingAgent()
    monkeypatch.setattr(api_app, "build", lambda checkpointer: blocking)
    client, _ = env
    monkeypatch.setattr(api_app, "build", lambda checkpointer: blocking)
    thread_id = client.post("/api/research", json={"topic": "取消任务"}).json()["thread_id"]
    time.sleep(0.05)

    response = client.post(f"/api/research/{thread_id}/cancel")
    assert response.status_code == 200
    blocking.release.set()
    events = _sse_events(client, thread_id)

    assert any(event["type"] == "cancelled" for event in events)
    assert run_store.get_run(thread_id).status == "cancelled"


def test_recover_running_runs(env):
    client, _ = env
    thread_id = client.post("/api/research", json={"topic": "恢复任务"}).json()["thread_id"]
    _sse_events(client, thread_id)
    con = run_store._connect()
    try:
        con.execute("UPDATE runs SET status='running', done_at=NULL WHERE thread_id=?", (thread_id,))
        con.commit()
    finally:
        con.close()

    assert run_store.recover_running_runs() == 1
    run = run_store.get_run(thread_id)
    assert run.status == "error" and "重启" in run.error


def test_api_key_guard(env, monkeypatch):
    client, _ = env
    monkeypatch.setattr(cfg, "API_KEY", "secret")
    assert client.get("/api/memory").status_code == 401
    assert client.get("/api/memory", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/api/memory", headers={"Authorization": "Bearer secret"}).status_code == 200
