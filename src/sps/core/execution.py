"""Read-only candidate execution and the frozen search's result semantics."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sqlite3
import time
from typing import Any

from prettytable import PrettyTable


@dataclass(frozen=True)
class Execution:
    status: str
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    error: str | None = None

    @property
    def valid(self) -> bool:
        # Historical search excludes empty and all-NULL results from voting.
        # This is not a claim that such SQL is semantically incorrect.
        return self.status == "success" and any(any(v is not None for v in row) for row in self.rows)

    def as_record(self) -> dict:
        def scalar(value):
            if isinstance(value, bytes):
                return {"sqlite_blob_hex": value.hex()}
            if isinstance(value, float) and not math.isfinite(value):
                return {"sqlite_float": repr(value)}
            return value
        return {"status": self.status, "columns": list(self.columns),
                "rows": [[scalar(v) for v in row] for row in self.rows], "error": self.error,
                "valid_for_search": self.valid}


def execute(database: Path, sql: str, *, timeout_seconds: float = 60,
            max_rows: int | None = None) -> Execution:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Execution timeout must be finite and positive")
    if max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1):
        raise ValueError("A row bound must be positive")
    if not database.is_file():
        return Execution("unavailable", error="database_not_found")
    deadline = time.monotonic() + timeout_seconds
    connection = None
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    def authorize(action, arg1, arg2, db, trigger):
        if action not in allowed:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() in {"load_extension", "readfile", "writefile"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    try:
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout_seconds)
        connection.text_factory = lambda value: value.decode("utf-8", errors="replace")
        connection.execute("PRAGMA query_only=ON")
        connection.set_authorizer(authorize)
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        cursor = connection.execute(sql)
        if cursor.description is None:
            return Execution("error", error="query_returned_no_columns")
        columns = tuple(col[0] for col in cursor.description)
        rows = []
        while True:
            if time.monotonic() >= deadline:
                return Execution("timeout", error=f"SQL execution timed out after {timeout_seconds:g} seconds")
            batch = cursor.fetchmany(1024)
            if not batch:
                break
            rows.extend(batch)
            if max_rows is not None and len(rows) > max_rows:
                return Execution("row_limit", error=f"query_exceeded_{max_rows}_rows")
        return Execution("success", columns, tuple(tuple(row) for row in rows))
    except sqlite3.Error as exc:
        if time.monotonic() >= deadline:
            return Execution("timeout", error=f"SQL execution timed out after {timeout_seconds:g} seconds")
        return Execution("error", error=str(exc))
    finally:
        if connection is not None:
            connection.close()


def format_execution(result: Execution, row_limit: int = 3, value_limit: int = 100) -> str:
    if result.status != "success":
        return result.error or result.status
    table = PrettyTable()
    columns = result.columns
    if len(columns) != len(set(columns)):
        columns = tuple(f"{col}_{i}" for i, col in enumerate(columns))
    table.field_names = columns
    table.add_rows([[v[:value_limit] + "..." if isinstance(v, str) and len(v) > value_limit else v
                     for v in row] for row in result.rows[:row_limit]])
    return str(table)


def outcome_key(rows: tuple[tuple[Any, ...], ...]) -> str:
    """Canonical bag used by CEM feedback, distinct from local set voting."""
    def cell(value):
        if value is None or isinstance(value, (str, bytes, int)):
            return value
        if isinstance(value, float):
            if math.isnan(value):
                return ("float", "nan")
            if math.isinf(value):
                return ("float", "inf" if value > 0 else "-inf")
            return ("float", round(value, 9))
        return (type(value).__name__, str(value))
    counts = Counter(tuple(cell(v) for v in row) for row in rows)
    payload = ("unordered", tuple(sorted(counts.items(), key=lambda item: repr(item[0]))))
    return hashlib.sha256(repr(payload).encode()).hexdigest()
