"""下载豆瓣电影数据并构建 SQLite 分析库（movies / movie_genres / movie_tags）。

用法: uv run python scripts/setup_db.py
"""
import csv
import io
import sqlite3
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_research import configuration as cfg  # noqa: E402


def parse_year(s: str | None) -> int | None:
    y = (s or "")[:4]
    return int(y) if y.isdigit() else None


def parse_runtime(s: str | None) -> int | None:
    digits = "".join(ch for ch in (s or "") if ch.isdigit())
    return int(digits) if digits else None


def main() -> None:
    print(f"下载豆瓣电影 CSV: {cfg.DOUBAN_CSV_URL}")
    r = httpx.get(cfg.DOUBAN_CSV_URL, timeout=60, follow_redirects=True)
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    print(f"原始 {len(rows)} 条")

    cfg.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(cfg.DB_PATH)
    cur = con.cursor()
    cur.executescript(
        """
        DROP TABLE IF EXISTS movies; DROP TABLE IF EXISTS movie_genres; DROP TABLE IF EXISTS movie_tags;
        CREATE TABLE movies(
            movie_id INTEGER PRIMARY KEY, title TEXT, rating REAL, total_ratings INTEGER,
            release_year INTEGER, runtime_min INTEGER, directors TEXT,
            countries TEXT, languages TEXT, link TEXT);
        CREATE TABLE movie_genres(movie_id INTEGER, genre TEXT);
        CREATE TABLE movie_tags(movie_id INTEGER, tag TEXT);
        CREATE INDEX idx_genres_genre ON movie_genres(genre);
        CREATE INDEX idx_tags_tag ON movie_tags(tag);
        CREATE INDEX idx_movies_rating ON movies(rating);
        """
    )

    def split_list(s: str | None) -> list[str]:
        return [x.strip() for x in (s or "").replace("，", ",").split(",") if x.strip()]

    for row in rows:
        mid = int(row["movie_id"])
        cur.execute(
            "INSERT OR REPLACE INTO movies VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                mid,
                row["title"],
                float(row["rating"]) if (row.get("rating") or "").strip() else None,
                int(row["total_ratings"]) if (row.get("total_ratings") or "").strip() else None,
                parse_year(row.get("release_date")),
                parse_runtime(row.get("runtime")),
                row.get("directors"),
                row.get("countries"),
                row.get("languages"),
                row.get("link"),
            ),
        )
        for g in split_list(row.get("genres")):
            cur.execute("INSERT INTO movie_genres VALUES(?,?)", (mid, g))
        for t in split_list(row.get("tags")):
            cur.execute("INSERT INTO movie_tags VALUES(?,?)", (mid, t))
    con.commit()

    for t in ("movies", "movie_genres", "movie_tags"):
        print(f"  {t}: {cur.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} 行")
    print("样例（评价人数 Top3）:")
    for row in cur.execute("SELECT title, rating, total_ratings FROM movies ORDER BY total_ratings DESC LIMIT 3"):
        print(f"  {row}")
    con.close()


if __name__ == "__main__":
    main()
