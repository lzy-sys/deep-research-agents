"""诊断脚本：对 API 发起一次研究并打印 SSE 每个事件的到达时间线。

用法: uv run python scripts/e2e_stream_probe.py [topic]
"""
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API = "http://localhost:18000"
topic = sys.argv[1] if len(sys.argv) > 1 else "用三句话介绍 LangGraph 的 StateGraph 与 Checkpointer 的作用"

t0 = time.time()
r = httpx.post(f"{API}/api/research", json={"topic": topic}, timeout=30)
thread_id = r.json()["thread_id"]
print(f"[{time.time()-t0:6.1f}s] 任务已创建: {thread_id}")

with httpx.stream("GET", f"{API}/api/research/{thread_id}/stream", timeout=700) as s:
    for line in s.iter_lines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            print(f"[{time.time()-t0:6.1f}s] [DONE]")
            break
        ev = json.loads(payload)
        etype = ev.get("type")
        if etype == "tool":
            print(f"[{time.time()-t0:6.1f}s] tool: {ev['label']}")
        elif etype == "final":
            content = ev.get("content", "")
            print(f"[{time.time()-t0:6.1f}s] final ({len(content)} 字)")
        elif etype == "error":
            print(f"[{time.time()-t0:6.1f}s] error: {ev.get('content')}")
        else:
            print(f"[{time.time()-t0:6.1f}s] {etype}")

print(f"总耗时 {time.time()-t0:.1f}s")
