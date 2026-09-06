"""模型基准测试：对比套餐内各模型的带工具请求延迟。用完即删。"""
import time

from deep_research import configuration as cfg
from deep_research.llm import get_model
from langchain_core.tools import tool


@tool
def dummy_tool(x: str) -> str:
    """测试工具"""
    return x


for mid in ["deepseek-v4-flash", "glm-5.3-flash", "kimi-k2.5", "minimax-m2.5"]:
    t0 = time.time()
    try:
        from langchain_openai import ChatOpenAI

        m = ChatOpenAI(
            model=mid,
            base_url=cfg.OPENCODE_BASE_URL,
            api_key=cfg.OPENCODE_API_KEY,
            temperature=0,
            timeout=45,
            max_retries=0,
        ).bind_tools([dummy_tool])
        r = m.invoke("用工具处理输入: hello")
        has_tc = bool(getattr(r, "tool_calls", None))
        print(f"{mid:20s} 成功 {time.time()-t0:5.1f}s  tool_calls={has_tc}", flush=True)
    except Exception as e:
        print(f"{mid:20s} 失败 {time.time()-t0:5.1f}s  {type(e).__name__}: {str(e)[:80]}", flush=True)
