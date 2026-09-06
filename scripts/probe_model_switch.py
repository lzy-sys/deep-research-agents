"""探测目标模型在网关上的可用性：延迟 / 思考模式参数 / 工具调用。

用法: uv run python scripts/probe_model_switch.py <model_id>
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from deep_research import configuration as cfg

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.7-plus"
client = OpenAI(base_url=cfg.OPENCODE_BASE_URL, api_key=cfg.OPENCODE_API_KEY, timeout=90, max_retries=0)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_movie_rating",
            "description": "查询豆瓣电影的评分",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string", "description": "电影名"}},
                "required": ["title"],
            },
        },
    }
]


def chat(tag: str, **kw) -> None:
    t = time.time()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "1+1等于几？只回答数字"}],
            temperature=0,
            **kw,
        )
        dt = time.time() - t
        msg = r.choices[0].message
        reason = bool(getattr(msg, "reasoning_content", None) or (msg.model_extra or {}).get("reasoning_content"))
        print(f"{tag:28s} {dt:5.1f}s  思考={'有' if reason else '无'}  answer={str(msg.content)[:25]!r}")
        return msg
    except Exception as e:
        print(f"{tag:28s} FAIL {time.time()-t:5.1f}s  {type(e).__name__}: {str(e)[:130]}")
        return None


print(f"===== 模型: {model} =====")
m1 = chat("baseline(无参数)")
m2 = chat("thinking=disabled", extra_body={"thinking": {"type": "disabled"}})
m3 = chat("enable_thinking=False", extra_body={"enable_thinking": False})

print("----- 工具调用测试（用可用的最简参数）-----")
best = {"thinking": {"type": "disabled"}} if (m2 is not None) else ({"enable_thinking": False} if m3 is not None else {})
t = time.time()
try:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "帮我查一下《肖申克的救赎》的豆瓣评分"}],
        tools=TOOLS,
        temperature=0,
        **({"extra_body": best} if False else {}),
    )
    dt = time.time() - t
    calls = r.choices[0].message.tool_calls
    if calls:
        print(f"tool_calls             {dt:5.1f}s  ✅ {calls[0].function.name}({calls[0].function.arguments[:60]})")
    else:
        reason = bool((r.choices[0].message.model_extra or {}).get("reasoning_content"))
        print(f"tool_calls             {dt:5.1f}s  ⚠️ 未调用工具，直接文本回答（思考={'有' if reason else '无'}）: {str(r.choices[0].message.content)[:60]!r}")
except Exception as e:
    print(f"tool_calls             FAIL {time.time()-t:5.1f}s  {type(e).__name__}: {str(e)[:130]}")
