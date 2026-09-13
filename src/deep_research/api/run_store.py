"""SQLite-backed task/event store for durable runs and replayable SSE."""
import json
import sqlite3
import time
from dataclasses import dataclass

from deep_research import configuration as cfg

DB_PATH = cfg.DATA_DIR / "runs.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
    thread_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    done_at REAL,
    steps INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_thread_id ON events(thread_id, id);
CREATE TABLE IF NOT EXISTS idempotency(
    key TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class RunRecord:
    thread_id: str
    topic: str
    status: str
    started_at: float
    done_at: float | None
    steps: int
    input_tokens: int
    output_tokens: int
    error: str | None
    cancel_requested: bool


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    con = _connect()
    try:
        con.executescript(_SCHEMA)
        con.commit()
    finally:
        con.close()


def _record(row: sqlite3.Row | None) -> RunRecord | None:
    if row is None:
        return None
    return RunRecord(
        thread_id=row["thread_id"],
        topic=row["topic"],
        status=row["status"],
        started_at=row["started_at"],
        done_at=row["done_at"],
        steps=row["steps"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        error=row["error"],
        cancel_requested=bool(row["cancel_requested"]),
    )


def create_run(thread_id: str, topic: str, idempotency_key: str | None = None) -> tuple[str, bool]:
    """Create/reset a run. Returns (thread_id, created)."""
    init_db()
    now = time.time()
    con = _connect()
    try:
        if idempotency_key:
            existing = con.execute(
                "SELECT thread_id FROM idempotency WHERE key=?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                return existing["thread_id"], False

        con.execute("DELETE FROM events WHERE thread_id=?", (thread_id,))
        con.execute(
            """
            INSERT INTO runs(
                thread_id, topic, status, started_at, done_at, steps,
                input_tokens, output_tokens, error, cancel_requested
            ) VALUES(?,?, 'running', ?, NULL, 0, 0, 0, NULL, 0)
            ON CONFLICT(thread_id) DO UPDATE SET
                topic=excluded.topic,
                status='running',
                started_at=excluded.started_at,
                done_at=NULL,
                steps=0,
                input_tokens=0,
                output_tokens=0,
                error=NULL,
                cancel_requested=0
            """,
            (thread_id, topic, now),
        )
        if idempotency_key:
            con.execute(
                "INSERT INTO idempotency(key, thread_id) VALUES(?,?)",
                (idempotency_key, thread_id),
            )
        con.commit()
        return thread_id, True
    finally:
        con.close()


def get_run(thread_id: str) -> RunRecord | None:
    init_db()
    con = _connect()
    try:
        return _record(con.execute("SELECT * FROM runs WHERE thread_id=?", (thread_id,)).fetchone())
    finally:
        con.close()


def append_event(thread_id: str, event: dict) -> int:
    init_db()
    con = _connect()
    try:
        cur = con.execute(
            "INSERT INTO events(thread_id, payload, created_at) VALUES(?,?,?)",
            (thread_id, json.dumps(event, ensure_ascii=False), time.time()),
        )
        con.commit()
        return int(cur.lastrowid)
    finally:
        con.close()


def list_events(thread_id: str, after_id: int = 0, limit: int = 200) -> list[dict]:
    init_db()
    con = _connect()
    try:
        rows = con.execute(
            "SELECT id, payload FROM events WHERE thread_id=? AND id>? ORDER BY id LIMIT ?",
            (thread_id, after_id, limit),
        ).fetchall()
        return [{"id": row["id"], **json.loads(row["payload"])} for row in rows]
    finally:
        con.close()


def finish_run(
    thread_id: str,
    status: str,
    *,
    steps: int,
    input_tokens: int,
    output_tokens: int,
    error: str | None = None,
) -> None:
    init_db()
    con = _connect()
    try:
        con.execute(
            """
            UPDATE runs SET status=?, done_at=?, steps=?, input_tokens=?, output_tokens=?, error=?
            WHERE thread_id=?
            """,
            (status, time.time(), steps, input_tokens, output_tokens, error, thread_id),
        )
        con.commit()
    finally:
        con.close()


def request_cancel(thread_id: str) -> bool:
    init_db()
    con = _connect()
    try:
        cur = con.execute(
            "UPDATE runs SET cancel_requested=1 WHERE thread_id=? AND status='running'",
            (thread_id,),
        )
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()


def is_cancel_requested(thread_id: str) -> bool:
    run = get_run(thread_id)
    return bool(run and run.cancel_requested)


def recover_running_runs() -> int:
    """A process restart cannot resume in-memory agent threads, so close them explicitly."""
    init_db()
    con = _connect()
    try:
        rows = con.execute("SELECT thread_id FROM runs WHERE status='running'").fetchall()
        now = time.time()
        for row in rows:
            thread_id = row["thread_id"]
            payload = json.dumps({"type": "error", "content": "服务重启，任务已中断"}, ensure_ascii=False)
            con.execute(
                "INSERT INTO events(thread_id, payload, created_at) VALUES(?,?,?)",
                (thread_id, payload, now),
            )
            con.execute(
                "UPDATE runs SET status='error', done_at=?, error=? WHERE thread_id=?",
                (now, "服务重启，任务已中断", thread_id),
            )
        con.commit()
        return len(rows)
    finally:
        con.close()


def purge_stale_runs(stale_seconds: int) -> int:
    init_db()
    cutoff = time.time() - stale_seconds
    con = _connect()
    try:
        rows = con.execute(
            "SELECT thread_id FROM runs WHERE status!='running' AND done_at IS NOT NULL AND done_at<?",
            (cutoff,),
        ).fetchall()
        for row in rows:
            con.execute("DELETE FROM events WHERE thread_id=?", (row["thread_id"],))
            con.execute("DELETE FROM runs WHERE thread_id=?", (row["thread_id"],))
            con.execute(
                "DELETE FROM idempotency WHERE thread_id=?",
                (row["thread_id"],),
            )
        con.commit()
        return len(rows)
    finally:
        con.close()


def stats() -> dict:
    init_db()
    con = _connect()
    try:
        row = con.execute(
            """
            SELECT
                COUNT(*) AS runs,
                SUM(status='running') AS active_runs,
                SUM(status='error') AS errors,
                COALESCE(SUM(steps), 0) AS steps,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(CASE WHEN done_at IS NOT NULL THEN done_at-started_at ELSE 0 END), 0) AS seconds
            FROM runs
            """
        ).fetchone()
        tools: dict[str, int] = {}
        for event_row in con.execute("SELECT payload FROM events WHERE json_extract(payload, '$.type')='tool'"):
            label = json.loads(event_row["payload"]).get("label", "?")
            tools[label] = tools.get(label, 0) + 1
        return {
            "runs": row["runs"],
            "active_runs": row["active_runs"] or 0,
            "errors": row["errors"] or 0,
            "steps": row["steps"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "seconds": row["seconds"],
            "tools": tools,
        }
    finally:
        con.close()
