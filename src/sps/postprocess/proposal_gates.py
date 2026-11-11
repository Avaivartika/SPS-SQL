"""Original post-correction proposal gates primitives."""
from __future__ import annotations
from collections import Counter
from typing import Any

def gate_support(rows: list[dict], records: dict[int, dict], minimum_support: float) -> dict:
    if not 0 <= minimum_support <= 1:
        raise ValueError("Minimum support must be in [0, 1]")
    if set(records) != {int(row["source_id"]) for row in rows}:
        raise ValueError("Selection records must cover every row")
    reasons: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []
    predictions: dict[str, str] = {}
    for source in rows:
        row = dict(source)
        source_id = int(row["source_id"])
        record = records[source_id]
        accepted = bool(row.get("repair_accepted"))
        support = float(record.get("selected_population_support") or 0.0)
        if not accepted:
            keep = False
            reason = "no_semantic_commit"
        elif support >= minimum_support:
            keep = True
            reason = "population_support_passed"
        else:
            keep = False
            reason = "population_support_below_threshold"
        final_sql = str(row["selected_sql"] if keep else row["original_sql"])
        row["repair_accepted"] = keep
        row["predicted_sql"] = final_sql
        row["population_support_gate"] = {
            "kept": keep,
            "reason": reason,
            "support": support,
            "threshold": minimum_support,
        }
        predictions[str(source_id)] = final_sql
        reasons[reason] += 1
        output_rows.append(row)

    return {"rows": output_rows, "predictions": predictions, "reasons": reasons}

def _proposal_kept(row: dict[str, Any]) -> bool:
    union = row.get("semantic_proposal_union")
    if isinstance(union, dict):
        return str(union.get("source")) != "baseline"
    gate = row.get("semantic_confidence_gate")
    if not isinstance(gate, dict):
        raise TypeError(
            "proposal row has neither a confidence gate nor a proposal union"
        )
    return bool(gate.get("kept"))

def merge_proposals(primary_rows: dict[int, dict], supplement_rows: dict[int, dict],
                    supplement_overrides_conflicts: bool = False) -> dict:
    if set(primary_rows) != set(supplement_rows):
        raise ValueError("Proposals must cover identical question IDs")
    output_rows: list[dict[str, Any]] = []
    predictions: dict[str, str] = {}
    decisions: Counter[str] = Counter()
    for source_id in sorted(primary_rows):
        primary = primary_rows[source_id]
        supplement = supplement_rows[source_id]
        if str(primary["db_id"]) != str(supplement["db_id"]):
            raise ValueError(f"database mismatch for question {source_id}")
        primary_kept = _proposal_kept(primary)
        supplement_kept = _proposal_kept(supplement)
        if (
            primary_kept
            and supplement_kept
            and supplement_overrides_conflicts
            and str(primary["predicted_sql"]) != str(supplement["predicted_sql"])
        ):
            selected_sql = str(supplement["predicted_sql"])
            source = "supplement_conflict"
        elif primary_kept:
            selected_sql = str(primary["predicted_sql"])
            source = "primary"
        elif supplement_kept:
            selected_sql = str(supplement["predicted_sql"])
            source = "supplement"
        else:
            selected_sql = str(primary["original_sql"])
            source = "baseline"
        decisions[source] += 1
        row = dict(primary)
        row["predicted_sql"] = selected_sql
        row["semantic_proposal_union"] = {
            "source": source,
            "primary_kept": primary_kept,
            "supplement_kept": supplement_kept,
        }
        output_rows.append(row)
        predictions[str(source_id)] = selected_sql

    return {"rows": output_rows, "predictions": predictions, "decisions": decisions}
