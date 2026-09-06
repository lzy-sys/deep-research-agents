"""SQL 安全闸门测试：拦截矩阵、LIMIT 补全、真实查询（豆瓣电影库）、只读物理防线。"""
import sqlite3

import pytest

from deep_research.tools.sql_tools import _connect, check_sql, run_sql

BAD_SQL = [
    "DROP TABLE Album",
    "DELETE FROM Album",
    "INSERT INTO Album VALUES (1)",
    "UPDATE Album SET Title = 'x'",
    "PRAGMA table_info(Album)",
    "ATTACH DATABASE 'x.db' AS x",
    "SELECT 1; SELECT 2",  # 多语句
    "",  # 空
    "VACUUM",
]


@pytest.mark.parametrize("sql", BAD_SQL)
def test_blocked(sql):
    _, err = check_sql(sql)
    assert err, f"{sql!r} 应被拦截"


GOOD_SQL = [
    "SELECT title FROM movies",
    "WITH top AS (SELECT movie_id FROM movies LIMIT 3) SELECT * FROM top",
    "select title from movies limit 3",  # 小写
]


@pytest.mark.parametrize("sql", GOOD_SQL)
def test_allowed(sql):
    out, err = check_sql(sql)
    assert err is None, f"{sql!r} 不应被拦截: {err}"
    assert "LIMIT" in out.upper()


def test_limit_autocomplete_and_preserve():
    out, _ = check_sql("SELECT title FROM movies")
    assert out.upper().endswith("LIMIT 50")
    out, _ = check_sql("SELECT title FROM movies LIMIT 3")
    assert out.upper().count("LIMIT") == 1  # 已有限制不重复追加


def test_run_sql_returns_markdown_table():
    out = run_sql.invoke({"sql": "SELECT title, rating FROM movies ORDER BY total_ratings DESC LIMIT 2"})
    assert out.startswith("|"), "应输出 Markdown 表格"
    assert "title" in out.lower()


def test_run_sql_blocked_path():
    out = run_sql.invoke({"sql": "DROP TABLE Album"})
    assert out.startswith("SQL 被安全闸门拒绝")


def test_readonly_physical_barrier():
    con = _connect()
    with pytest.raises(sqlite3.OperationalError):  # 只读 URI 下写操作物理失败
        con.execute("CREATE TABLE t_should_fail (id INTEGER)")
    con.close()
