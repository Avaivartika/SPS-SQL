"""Compile one read-only SQLite query without executing it or reading answers.

A successful plan checks database identifiers and SQL compilation, not whether
the query answers a natural-language question. It is never a correctness label.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
import time


@dataclass(frozen=True)
class PlanCheck:
    status: str
    error: str | None = None


def database_path(root: Path, db_id: str) -> Path:
    if not isinstance(db_id, str) or not db_id or db_id in {".", ".."} or any(c in db_id for c in ("/", "\\", "\x00")):
        raise ValueError("Invalid database ID")
    root = root.resolve(strict=True)
    path = (root / db_id / f"{db_id}.sqlite").resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Database is not a file within the supplied root")
    return path


def check_plan(database: Path, sql: str, *, timeout_seconds: float = 5.0) -> PlanCheck:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Timeout must be positive and finite")
    if not isinstance(sql, str) or not sql.strip():
        return PlanCheck("invalid", "empty_query")
    if not database.is_file():
        return PlanCheck("unavailable", "database_not_found")
    deadline = time.monotonic() + timeout_seconds
    connection = None
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

    def authorize(action, argument1, argument2, database_name, trigger):
        if action not in allowed:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and str(argument2).lower() in {"load_extension", "readfile", "writefile"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    try:
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout_seconds)
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(authorize)
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        # SQLite prepares the query, but never executes the candidate itself.
        connection.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        return PlanCheck("valid")
    except sqlite3.Error as error:
        message = str(error)
        unavailable = any(part in message.lower() for part in ("locked", "unable to open", "malformed", "not a database", "disk i/o"))
        status = "timeout" if time.monotonic() >= deadline else "unavailable" if unavailable else "invalid"
        return PlanCheck(status, message)
    finally:
        if connection is not None:
            connection.close()
