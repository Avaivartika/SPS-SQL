"""Record terminal SQL and execution evidence without selecting or editing it."""
from __future__ import annotations

from copy import deepcopy
import hashlib

from sps.core.route_execution import PublicQuestion


def collect_candidates(question: PublicQuestion, populations: list[dict]) -> dict:
    if not populations:
        raise ValueError("At least one population is required")
    candidates, attempts, rounds = {}, [], []
    previous_id = 0
    for round_index, population in enumerate(populations, start=1):
        if (population["question_id"], population["db_id"]) != (question.question_id, question.db_id):
            raise ValueError("Population identity differs from the public question")
        rounds.append({"round": round_index, "status": population["status"],
                       "error": population.get("error")})
        for attempt in population["attempted"]:
            trajectory_id = attempt["trajectory_id"]
            if type(trajectory_id) is not int or trajectory_id <= previous_id:
                raise ValueError("Trajectory IDs must increase across successive populations")
            previous_id = trajectory_id
            sql = attempt["final_sql"]
            if sql is not None and not isinstance(sql, str):
                raise ValueError("Terminal SQL must be text or null")
            # Preserve whitespace within quoted literals and distinct SQL texts,
            # even when their execution results coincide.
            sql = sql.strip() if sql else None
            identifier = hashlib.sha256(sql.encode()).hexdigest() if sql else None
            evidence = {"round": round_index, "trajectory_id": trajectory_id,
                "candidate_id": identifier, "status": attempt["status"],
                **{key: deepcopy(attempt.get(key)) for key in (
                    "route", "reward", "search_score", "outcome_key", "execution",
                    "input_tokens", "output_tokens", "total_tokens", "error")}}
            attempts.append(evidence)
            if identifier is None:
                continue
            if identifier not in candidates:
                candidates[identifier] = {"candidate_id": identifier, "sql": sql, "sources": []}
            elif candidates[identifier]["sql"] != sql:
                raise ValueError("Candidate hash collision")
            candidates[identifier]["sources"].append({"round": round_index,
                                                     "trajectory_id": trajectory_id})
    complete = all(row["status"] == "ok" for row in rounds)
    usage = {}
    for key in ("input_tokens", "output_tokens"):
        counts = [p.get(key) for p in populations]
        if any(value is not None and (type(value) is not int or value < 0) for value in counts):
            raise ValueError("Invalid population token accounting")
        usage[key] = sum(counts) if all(value is not None for value in counts) else None
    return {"artifact_kind": "native_sps_candidate_archive_v1",
        "question_id": question.question_id, "db_id": question.db_id,
        "status": "ok" if complete and candidates else "incomplete",
        "rounds": rounds, "candidates": list(candidates.values()), "attempts": attempts,
        "candidate_count": len(candidates), "attempt_count": len(attempts),
        "selection_performed": False, "token_usage": usage}
