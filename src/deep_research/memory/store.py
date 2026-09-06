"""长期记忆存储：SQLite 结构化条目库（data/memory.sqlite），替代早期 AGENTS.md 文件方案。

- 每条偏好/结论一行，支持单条增删查、容量上限（50 条，满员自动淘汰最旧结论，偏好永不自动清除）
- 智能体通过 memory/tools.py 的 save_memory/delete_memory 工具读写，注入与展示统一走 format_memory/list_entries
- 首次建库时从旧 AGENTS.md 一次性迁移既有条目（meta 表打标，保证只跑一次）
"""
import sqlite3
from datetime import datetime

from deep_research import configuration as cfg

DB_PATH = cfg.MEMORY_DB_PATH
LEGACY_MD_PATH = cfg.ROOT / "AGENTS.md"  # 旧文件方案的记忆文件，仅用于一次性迁移
CAPACITY = 50
CATEGORIES = ("偏好", "历史研究结论")
_EVICTABLE = "历史研究结论"  # 容量满时只淘汰结论类条目
_LEGACY_SECTIONS = {"用户偏好": "偏好", "历史研究结论": "历史研究结论"}  # 旧文件节名 → 条目类别

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_entry(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _migrate_from_agents_md(con: sqlite3.Connection) -> int:
    """解析旧 AGENTS.md 的「用户偏好」「历史研究结论」两节的条目行（- 开头）导入。"""
    if not LEGACY_MD_PATH.exists():
        return 0
    section = None
    now = datetime.now().isoformat(timespec="seconds")
    imported = 0
    for raw in LEGACY_MD_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            section = _LEGACY_SECTIONS.get(line[3:].strip())
            continue
        if section is None or not line.startswith("- "):
            continue
        content = line[2:].strip()
        if not content or content.startswith("（"):  # 跳过「（暂无…）」占位
            continue
        con.execute(
            "INSERT INTO memory_entry(category, content, created_at, updated_at) VALUES(?,?,?,?)",
            (section, content, now, now),
        )
        imported += 1
    return imported


def _ensure_db() -> None:
    """建表；首次建库时迁移旧文件记忆并打标（此后 reset 也不会重复导入）。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = _connect()
    try:
        con.executescript(_SCHEMA)
        migrated = con.execute("SELECT value FROM meta WHERE key='migrated'").fetchone()
        if migrated is None:
            _migrate_from_agents_md(con)
            con.execute("INSERT INTO meta(key, value) VALUES('migrated', '1')")
            con.commit()
    finally:
        con.close()


def add_entry(category: str, content: str) -> tuple[int, str]:
    """新增一条记忆，返回 (条目 id, 附加说明)。容量满时自动淘汰最旧的历史结论。"""
    if category not in CATEGORIES:
        raise ValueError(f"category 必须是 {CATEGORIES} 之一")
    content = content.strip()
    if not content:
        raise ValueError("内容不能为空")
    _ensure_db()
    now = datetime.now().isoformat(timespec="seconds")
    con = _connect()
    try:
        count = con.execute("SELECT COUNT(*) FROM memory_entry").fetchone()[0]
        note = ""
        if count >= CAPACITY:
            oldest = con.execute(
                "SELECT id FROM memory_entry WHERE category=? ORDER BY created_at, id LIMIT 1",
                (_EVICTABLE,),
            ).fetchone()
            if oldest is None:
                raise RuntimeError(f"记忆已满（{CAPACITY} 条）且没有可淘汰的历史结论，请先删除部分条目")
            con.execute("DELETE FROM memory_entry WHERE id=?", (oldest["id"],))
            note = f"（容量已满，自动淘汰了最旧的历史结论 #{oldest['id']}）"
        cur = con.execute(
            "INSERT INTO memory_entry(category, content, created_at, updated_at) VALUES(?,?,?,?)",
            (category, content, now, now),
        )
        con.commit()
        return cur.lastrowid, note
    finally:
        con.close()


def delete_entry(entry_id: int) -> bool:
    _ensure_db()
    con = _connect()
    try:
        cur = con.execute("DELETE FROM memory_entry WHERE id=?", (entry_id,))
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()


def list_entries() -> list[dict]:
    """全部条目，偏好在前、按写入顺序。"""
    _ensure_db()
    con = _connect()
    try:
        rows = con.execute(
            "SELECT id, category, content, created_at, updated_at FROM memory_entry "
            "ORDER BY CASE category WHEN '偏好' THEN 0 ELSE 1 END, id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def format_memory() -> str:
    """提示词注入用文本：条目带 [id] 标注，便于智能体精确调用 delete_memory。"""
    entries = list_entries()
    if not entries:
        return "（暂无长期记忆）"
    lines: list[str] = []
    for cat in CATEGORIES:
        items = [e for e in entries if e["category"] == cat]
        if items:
            lines.append(f"## {cat}")
            lines.extend(f"- [{e['id']}] {e['content']}" for e in items)
    return "\n".join(lines)


def reset_memory() -> int:
    """清空全部记忆，返回删除的条目数（迁移标记保留，不会从旧文件重复导入）。"""
    _ensure_db()
    con = _connect()
    try:
        n = con.execute("SELECT COUNT(*) FROM memory_entry").fetchone()[0]
        con.execute("DELETE FROM memory_entry")
        con.commit()
        return n
    finally:
        con.close()
