"""豆瓣电影库只读 SQL 工具：纯代码安全闸门（仅 SELECT/WITH、单语句、强制 LIMIT）。"""
import re
import sqlite3

from langchain_core.tools import tool

from deep_research import configuration as cfg

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|GRANT|REVOKE)\b",
    re.I,
)
MAX_ROWS = 50


def check_sql(sql: str) -> tuple[str, str | None]:
    """校验并规范化 SQL。返回 (规范化后 SQL, 错误信息)；通过时错误为 None。"""
    sql = sql.strip().rstrip(";")
    if not sql:
        return sql, "SQL 为空"
    if ";" in sql:
        return sql, "禁止多语句"
    if _FORBIDDEN.search(sql):
        return sql, "仅允许 SELECT 查询"
    if not sql.upper().startswith(("SELECT", "WITH")):
        return sql, "仅允许 SELECT/WITH 开头的查询"
    if not re.search(r"\bLIMIT\s+\d+\b", sql, re.I):
        sql += f" LIMIT {MAX_ROWS}"  # 强制限行，防全表拉爆
    return sql, None


def get_schema_summary() -> str:
    """动态读取库表结构生成 schema 摘要，注入 sql_expert 的系统提示词（换库自动适配）。"""
    con = _connect()
    try:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        parts = []
        for (t,) in rows:
            cols = [f"{r[1]}({r[2]})" for r in con.execute(f"PRAGMA table_info({t})")]
            parts.append(f"{t}: {', '.join(cols)}")
        return "\n".join(parts)
    finally:
        con.close()


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{cfg.DB_PATH}?mode=ro", uri=True)  # 只读 URI，物理防写


@tool
def run_sql(sql: str) -> str:
    """对豆瓣电影数据库执行只读 SELECT 查询，返回 Markdown 表格结果。
    表：movies（电影主表：评分/评价人数/年代/片长/国家）、movie_genres（类型维度表）、movie_tags（标签维度表）。
    """
    sql, err = check_sql(sql)
    if err:
        return f"SQL 被安全闸门拒绝：{err}\n原 SQL: {sql}"
    con = _connect()
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS)
    except sqlite3.Error as e:
        return f"SQL 执行错误：{e}\n请根据错误信息修正 SQL 后重试。"
    finally:
        con.close()
    if not rows:
        return "查询成功，结果为空。"
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}\n\n(共 {len(rows)} 行)"
