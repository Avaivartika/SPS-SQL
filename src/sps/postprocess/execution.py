"""Original post-correction execution primitives."""
from __future__ import annotations
import hashlib
import math
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sqlglot
from sqlglot import exp

@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    rows: tuple[tuple[Any, ...], ...] = ()
    elapsed_seconds: float = 0.0
    error: str | None = None

def execute_sql(
    database_path: str | Path,
    sql: str,
    *,
    timeout_seconds: float = 30.0,
    max_rows: int = 1_000_000,
) -> ExecutionResult:
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        return ExecutionResult(ok=False, error=f"database_not_found:{path}")
    try:
        expression = sqlglot.parse_one(sql, read="sqlite")
    except Exception as error:
        return ExecutionResult(ok=False, error=f"parse_error:{type(error).__name__}:{error}")
    if not isinstance(expression, exp.Query):
        return ExecutionResult(ok=False, error=f"unsafe_statement:{type(expression).__name__}")

    start = time.perf_counter()
    deadline = start + timeout_seconds
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=timeout_seconds)
        # Some released Spider databases contain legacy non-UTF-8 TEXT bytes.
        # Returning raw bytes avoids a Python decoding failure while leaving
        # SQLite's query semantics unchanged; canonical comparison already
        # handles bytes deterministically.
        connection.text_factory = bytes
        connection.execute("PRAGMA query_only = ON")

        def _progress() -> int:
            return int(time.perf_counter() >= deadline)

        connection.set_progress_handler(_progress, 10_000)
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return ExecutionResult(
                ok=False,
                elapsed_seconds=time.perf_counter() - start,
                error=f"row_limit_exceeded:{max_rows}",
            )
        return ExecutionResult(
            ok=True,
            rows=tuple(tuple(row) for row in rows),
            elapsed_seconds=time.perf_counter() - start,
        )
    except sqlite3.Error as error:
        return ExecutionResult(
            ok=False,
            elapsed_seconds=time.perf_counter() - start,
            error=f"sqlite_error:{error}",
        )
    finally:
        if connection is not None:
            connection.close()

def execution_result_key(
    rows: tuple[tuple[Any, ...], ...],
    *,
    ordered: bool,
    set_semantics: bool = False,
) -> str:
    """Return a stable key with the same row semantics as execution accuracy.

    Candidate collection uses this key to collapse SQL strings that execute to
    the same outcome.  Reusing the comparison canonicalization is important:
    raw ``repr(rows)`` would split unordered results when SQLite happens to
    emit their rows in a different order, and it would disagree with the
    float normalization used by :func:`cached_results_equivalent`.
    """

    canonical_rows = tuple(_canonical_row(row) for row in rows)
    if set_semantics:
        payload: object = (
            "set",
            tuple(sorted(set(canonical_rows), key=repr)),
        )
    elif ordered:
        payload = ("ordered", canonical_rows)
    else:
        counts = Counter(canonical_rows)
        payload = (
            "unordered",
            tuple(sorted(counts.items(), key=lambda item: repr(item[0]))),
        )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

def _canonical_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(_canonical_cell(value) for value in row)

def _canonical_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
        return ("float", round(value, 9))
    return (type(value).__name__, str(value))
