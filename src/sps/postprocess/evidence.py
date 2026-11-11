"""Read native archives directly; preserve original execution-support semantics."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from sps.core.sql_plan import database_path
from sps.postprocess.execution import execute_sql, execution_result_key


def archive_attempts(archive: dict) -> list[dict]:
    sqls = {c["candidate_id"]: c["sql"] for c in archive["candidates"]}
    if len(sqls) != len(archive["candidates"]):
        raise ValueError("Duplicate candidate identities")
    identities = [a["trajectory_id"] for a in archive["attempts"]]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate trajectory identities")
    result = []
    for item in archive["attempts"]:
        if item["status"] != "ok" or item.get("error") is not None:
            continue
        sql = sqls.get(item["candidate_id"])
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("Successful trajectory has no archived SQL")
        result.append({"sql": sql, "trajectory_id": item["trajectory_id"],
                       "round": item["round"], "route": "->".join(item["route"]),
                       "population_support": float(item.get("search_score") or 0),
                       "total_tokens": int(item.get("total_tokens") or 0)})
    return result


def candidate_evidence(archive: dict, database_root: Path) -> tuple[list[dict], dict]:
    attempts = archive_attempts(archive)
    database = database_path(database_root, archive["db_id"])
    outcomes = {}
    for item in attempts:
        sql = item["sql"]
        if sql not in outcomes:
            execution = execute_sql(database, sql, timeout_seconds=30, max_rows=1_000_000)
            # An empty result or a NULL aggregate is a successful execution.
            # Keep its evidence separate from syntax/runtime failures, just as
            # we do for any other result. Correctness is not known here.
            outcomes[sql] = {"fingerprint": execution_result_key(execution.rows, ordered=False) if execution.ok else None,
                             "error": execution.error}
    valid = [a for a in attempts if outcomes[a["sql"]]["fingerprint"]]
    route_counts = Counter((a["round"], a["route"]) for a in valid)
    mass = defaultdict(float)
    for item in valid:
        mass[outcomes[item["sql"]]["fingerprint"]] += 1.0 / route_counts[(item["round"], item["route"])]
    total = sum(mass.values())
    # Select the same representative BEFORE replacing population support with
    # cross-round, route-balanced support. SQL text equality is deliberately exact.
    candidates = {}
    for item in attempts:
        old = candidates.get(item["sql"])
        key = lambda x: (x["population_support"], -x["total_tokens"], -x["trajectory_id"])
        if old is None or key(item) > key(old):
            candidates[item["sql"]] = dict(item)
    for item in candidates.values():
        outcome = outcomes[item["sql"]]
        item["population_support"] = mass[outcome["fingerprint"]] / total if outcome["fingerprint"] and total else 0.0
    return list(candidates.values()), {"question_id": archive["question_id"],
        "db_id": archive["db_id"], "attempt_count": len(attempts), "unique_sql_count": len(outcomes),
        "execution_groups": len(mass), "outcomes": outcomes}
