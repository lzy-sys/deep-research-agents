"""API 服务层测试：RUNS 生命周期 / SSE 事件流 / 追问 / 记忆与报告端点。

用 FakeAgent 替换真实 deep agent，全程不触网、不编译图。
"""
import time

import pytest
from fastapi.testclient import TestClient

from deep_research.api import app as api_app
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
    fake = FakeAgent()
    monkeypatch.setattr(api_app, "build", lambda checkpointer: fake)
    api_app.RUNS.clear()
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
    assert api_app.RUNS[thread_id].done is True
    # 追问消息进入图输入
    assert fake.stream_inputs[0][1]["configurable"]["thread_id"] == thread_id


def test_research_purges_stale_runs(env):
    client, _ = env
    old_id = client.post("/api/research", json={"topic": "旧任务"}).json()["thread_id"]
    api_app.RUNS[old_id].done_at = time.time() - api_app.RUN_STALE_SECONDS - 1

    new_id = client.post("/api/research", json={"topic": "新任务"}).json()["thread_id"]

    assert old_id not in api_app.RUNS  # 过期条目被清理
    assert new_id in api_app.RUNS


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

    run = api_app.RUNS[thread_id]
    assert run.done is False and run.topic == "追问"  # 新的 Run 对象接管该线程


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
