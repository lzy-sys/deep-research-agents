"""测量网关在不同 payload 大小下的单次调用延迟（定位总耗时瓶颈）。

用法: uv run python scripts/probe_latency.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from deep_research import configuration as cfg

client = OpenAI(base_url=cfg.OPENCODE_BASE_URL, api_key=cfg.OPENCODE_API_KEY, timeout=120, max_retries=0)
model = cfg.MODEL_RESEARCH

# 模拟 sql-expert 真实体积：系统提示 + schema 说明 ≈ 5KB
big_system = "你是数据分析专家，通过只读 SQL 查询豆瓣电影数据库回答统计问题。\n数据库 Schema 与规范：\n" + (
    "movies 表包含 movie_id(主键), title(片名), rating(评分), total_ratings(评价人数), "
    "release_year(年代), runtime_min(片长), directors(导演), countries(国家/地区), "
    "languages(语言), link(链接) 等字段；movie_genres 与 movie_tags 为一对多维度表，"
    "通过 movie_id 关联；聚合查询注意去重与 NULL 处理；按评分排序需带评价人数防小样本误导；"
    "仅允许 SELECT/WITH 只读查询，强制 LIMIT 50。\n" * 25
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": "执行只读 SQL 查询豆瓣电影数据库",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "SQL 语句"}},
            "required": ["sql"],
        },
    },
}]


def timed(tag: str, messages: list, **kw) -> None:
    t = time.time()
    try:
        r = client.chat.completions.create(
            model=model, messages=messages, temperature=0,
            extra_body=cfg.LLM_EXTRA_BODY, **kw,
        )
        n = len(str(r.choices[0].message.content or ""))
        print(f"{tag:26s} {time.time()-t:6.1f}s  返回{n}字")
    except Exception as e:
        print(f"{tag:26s} FAIL {time.time()-t:6.1f}s  {type(e).__name__}: {str(e)[:100]}")


print(f"===== {model} @ {time.strftime('%H:%M:%S')} =====")
for i in range(2):
    timed(f"小请求(~0.1KB) #{i}", [{"role": "user", "content": "1+1等于几？只回答数字"}])
for i in range(2):
    timed(f"大请求(~5KB+工具) #{i}",
          [{"role": "system", "content": big_system},
           {"role": "user", "content": "评分8.5以上且评价人数超50万的电影有哪些？需要查库请调用工具"}],
          tools=TOOLS)
