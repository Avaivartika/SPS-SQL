"""Frozen final-population voting, separate from the cross-round SQL archive."""
from collections import defaultdict
import math
import statistics
import time

from sps.core.execution import execute


def ordered_final_sqls(population):
    if population["status"] != "ok":
        raise ValueError("Raw selection requires a complete population")
    attempts = {r["trajectory_id"]: r for r in population["attempted"]}
    ids = population["successful_ids"]
    if len(attempts) != len(population["attempted"]) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate final trajectory identity")
    sqls = []
    for identity in ids:
        attempt = attempts[identity]
        if attempt["status"] != "ok" or not isinstance(attempt["final_sql"], str) or not attempt["final_sql"].strip():
            raise ValueError("Invalid final path")
        sqls.append(attempt["final_sql"])
    if not sqls:
        raise ValueError("Empty final population")
    return sqls


def select_raw(sqls, *, execute_one, measure_time, repeats=20):
    if not sqls or type(repeats) is not int or repeats < 1:
        raise ValueError("Nonempty ordered paths and positive timing repeats required")
    valid_groups, all_groups = defaultdict(list), defaultdict(list)
    for index, sql in enumerate(sqls):
        result = execute_one(sql)
        if result.status != "success":
            continue
        key = frozenset(result.rows)
        all_groups[key].append(index)
        if result.valid:
            valid_groups[key].append(index)
    groups = valid_groups or all_groups
    if not groups:
        return {"selected_sql": "ERROR", "selected_index": None, "groups": [],
            "used_empty_result_fallback": False, "reason": "no_successful_execution"}
    total = sum(map(len, groups.values()))
    records = []
    for indices in groups.values():
        latency = measure_time(sqls[indices[0]], repeat=repeats)
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("Raw selection timing is unavailable")
        records.append({"representative_index": indices[0], "path_indices": indices,
            "support": len(indices) / total, "execution_time": latency})
    ranked = sorted(records, key=lambda row: (row["support"], -row["execution_time"]), reverse=True)
    winner = ranked[0]["representative_index"]
    return {"selected_sql": sqls[winner], "selected_index": winner, "groups": records,
        "used_empty_result_fallback": not bool(valid_groups), "reason": "execution_majority"}


def measure_execution_time(database, sql, *, repeat=20):
    durations = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = execute(database, sql, timeout_seconds=60)
        if result.status == "success":
            durations.append(time.perf_counter() - started)
    if not durations:
        raise ValueError("No successful timing execution")
    mean, std = statistics.mean(durations), statistics.pstdev(durations)
    trimmed = [d for d in durations if mean - 3 * std < d < mean + 3 * std]
    # The historical estimator returns NaN for an empty trimmed sample. Record
    # failure instead of allowing NaN into ranking or changing the tie rule.
    if not trimmed:
        raise ValueError("No finite trimmed timing estimate")
    return statistics.mean(trimmed)
