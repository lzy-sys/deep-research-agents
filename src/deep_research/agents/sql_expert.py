"""sql-expert 子智能体：豆瓣电影数据库统计分析（text2sql）。"""
from deep_research.llm import get_model
from deep_research.prompts import SQL_EXPERT_PROMPT
from deep_research.tools.sql_tools import get_schema_summary, run_sql


def build_sql_expert() -> dict:
    return {
        "name": "sql-expert",
        "description": (
            "数据分析专家。对豆瓣电影数据库（约 2000 部电影：评分、评价人数、类型、年代、国家、片长）"
            "执行只读 SQL 查询，回答评分分布、类型对比、年代趋势等统计类问题，返回表格数据与结论。"
            "输入一个具体的数据问题。"
        ),
        "system_prompt": SQL_EXPERT_PROMPT.format(schema=get_schema_summary()),
        "tools": [run_sql],
        "model": get_model("research"),
    }
