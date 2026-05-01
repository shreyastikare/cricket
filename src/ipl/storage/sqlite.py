"""Shared sqlite3 helpers for storage modules."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from .config import DB_PATH
from .utils import ensure_dir


def quote_ident(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    ensure_dir(path.parent)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def transaction(db_path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_ident(table_name)});").fetchall()
    return [row[1] for row in rows]


def build_upsert_sql(
    table_name: str,
    columns: list[str],
    conflict_columns: list[str],
    update_columns: list[str] | None = None,
) -> str:
    quoted_columns = ", ".join(quote_ident(col) for col in columns)
    placeholders = ", ".join(["?"] * len(columns))
    conflict_target = ", ".join(quote_ident(col) for col in conflict_columns)

    if update_columns is None:
        update_columns = [col for col in columns if col not in conflict_columns]

    if update_columns:
        update_clause = ", ".join(
            f"{quote_ident(col)}=excluded.{quote_ident(col)}" for col in update_columns
        )
        on_conflict = f"DO UPDATE SET {update_clause}"
    else:
        on_conflict = "DO NOTHING"

    return (
        f"INSERT INTO {quote_ident(table_name)} ({quoted_columns}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_target}) {on_conflict};"
    )


def dataframe_records(df: pd.DataFrame, columns: list[str]) -> Iterable[tuple]:
    normalized = df[columns].where(pd.notna(df[columns]), None)
    return normalized.itertuples(index=False, name=None)


def upsert_dataframe(
    conn: sqlite3.Connection,
    table_name: str,
    df: pd.DataFrame,
    conflict_columns: list[str],
    update_columns: list[str] | None = None,
    chunk_size: int = 5000,
) -> int:
    if df.empty:
        return 0

    columns = list(df.columns)
    sql = build_upsert_sql(
        table_name=table_name,
        columns=columns,
        conflict_columns=conflict_columns,
        update_columns=update_columns,
    )

    rows_written = 0
    records = list(dataframe_records(df, columns))
    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]
        conn.executemany(sql, chunk)
        rows_written += len(chunk)

    return rows_written
