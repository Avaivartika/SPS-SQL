"""Native SQL-producing action with explicit sampling, retries and local voting."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import random
import re
from typing import Awaitable, Callable

import sqlglot

from sps.core.execution import Execution


def extract_sql(response: str | None) -> str | None:
    # Preserve the actual frozen parser, including its fenced-SQL fallback.
    # The old configuration called this 'xml_only' despite that fallback.
    if not isinstance(response, str):
        return None
    match = re.search(r"<sql>(.*)</sql>", response, flags=re.DOTALL)
    if match is None:
        match = re.search(r"```(?:sqlite|sql)\b\s*(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    sql = match.group(1).strip()
    if sql.startswith("```sql") and sql.endswith("```"):
        sql = sql[6:-3]
    try:
        return sqlglot.parse_one(sql, dialect="sqlite").sql(dialect="sqlite", normalize=True, pretty=False, comments=False)
    except Exception:
        return sql


@dataclass(frozen=True)
class ModelBatch:
    responses: tuple[str | None, ...]
    input_tokens: int | None
    output_tokens: int | None
    finish_reasons: tuple[str | None, ...] = ()
    request_attempts: int = 1
    retry_errors: tuple[str, ...] = ()
    seed: int | None = None


@dataclass
class SQLGroup:
    status: str = "running"
    selected_sql: str | None = None
    consistency_score: float = 0.0
    selected_valid: bool = False
    attempts: list[dict] = field(default_factory=list)
    batches: list[dict] = field(default_factory=list)
    parse_failures: int = 0
    invalid_retries: int = 0
    error: str | None = None


async def sample_sql_group(
    sample: Callable[[int], Awaitable[ModelBatch]],
    execute: Callable[[str], Awaitable[Execution]],
    rng: random.Random,
    *, samples: int = 5, invalid_retry_limit: int = 15, max_batches: int = 32,
) -> SQLGroup:
    if samples < 1 or invalid_retry_limit < 0 or max_batches < 1:
        raise ValueError("Invalid action sampling budget")
    result = SQLGroup()
    admitted_sql = []
    groups = defaultdict(list)
    while len(admitted_sql) < samples:
        if result.invalid_retries >= invalid_retry_limit and admitted_sql:
            break
        if len(result.batches) >= max_batches:
            result.status, result.error = "budget_exhausted", "action_batch_limit"
            return result
        count = samples - len(admitted_sql)
        try:
            batch = await sample(count)
            result.batches.append({"requested_samples": count, "responses": list(batch.responses),
                                   "input_tokens": batch.input_tokens, "output_tokens": batch.output_tokens,
                                   "finish_reasons": list(batch.finish_reasons), "seed": batch.seed})
            if len(batch.responses) != count:
                raise ValueError("Model returned a different number of samples")
            for response in batch.responses:
                sql = extract_sql(response)
                if sql is None:
                    result.parse_failures += 1
                    continue
                outcome = await execute(sql)
                admitted = outcome.valid or result.invalid_retries >= invalid_retry_limit
                result.attempts.append({"sql": sql, "execution": outcome.as_record(), "admitted": admitted})
                if admitted:
                    admitted_sql.append(sql)
                    if outcome.valid:
                        # Local action consensus intentionally uses set equality.
                        groups[frozenset(outcome.rows)].append(sql)
                else:
                    result.invalid_retries += 1
        except Exception as exc:
            result.status, result.error = "error", f"{type(exc).__name__}: {exc}"
            return result
    if not groups:
        result.selected_sql = rng.choice(admitted_sql)
    else:
        max_size = 0
        for sqls in groups.values():
            # Preserve both first-group tie resolution and the historical RNG
            # draws for successively larger groups, not just the final winner.
            if len(sqls) > max_size:
                result.selected_sql = rng.choice(sqls)
                max_size = len(sqls)
        result.consistency_score = max_size / sum(len(sqls) for sqls in groups.values())
        result.selected_valid = True
    result.status = "ok"
    return result
