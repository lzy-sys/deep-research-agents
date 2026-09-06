"""探测网关 deepseek-v4-flash 支持的思考模式关闭参数（只测同一模型的不同参数）。

用法: uv run python scripts/probe_thinking.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from deep_research import configuration as cfg

client = OpenAI(base_url=cfg.OPENCODE_BASE_URL, api_key=cfg.OPENCODE_API_KEY, timeout=90, max_retries=0)
model = cfg.MODEL_RESEARCH


def call(tag: str, **kw) -> None:
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
        rt = getattr(getattr(r.usage, "completion_tokens_details", None), "reasoning_tokens", None)
        print(f"{tag:24s} {dt:5.1f}s  思考内容={'有' if reason else '无'}  reasoning_tokens={rt}  answer={str(msg.content)[:30]!r}")
    except Exception as e:
        print(f"{tag:24s} FAIL {time.time()-t:5.1f}s  {type(e).__name__}: {str(e)[:120]}")


print(f"model={model}")
call("baseline(默认)")
call("enable_thinking=False", extra_body={"enable_thinking": False})
call("thinking=disabled", extra_body={"thinking": {"type": "disabled"}})
call("chat_template_kwargs", extra_body={"chat_template_kwargs": {"thinking": False}})
call("reasoning_effort=none", extra_body={"reasoning_effort": "none"})
