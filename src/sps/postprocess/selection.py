"""The original structural and semantic selection rules over native candidates."""
from __future__ import annotations

from sps.postprocess.comparison import score_contract, contract_field_changes, regresses_ordered_slot_subfield
from sps.postprocess.contracts import AnswerContract
from sps.postprocess.field_confidence import SEMANTIC_CONTRACT_FIELDS


def select(original_sql: str, context: dict, candidates: list[dict], schema: dict,
           *, semantic: bool, weight: float, gain: float) -> tuple[dict, dict]:
    row = {**context, "original_sql": original_sql, "selected_sql": original_sql,
           "predicted_sql": original_sql, "repair_accepted": False, "selected_edits": [],
           "static_score_margin": 0.0, "candidate_error": None, "selected_plan_error": None,
           "candidate_count": len(candidates)}
    record = {"question_id": int(context["source_id"]), "selected_population_support": 0.0}
    try:
        target = AnswerContract.model_validate(context["predicted_contract"])
    except (TypeError, ValueError):
        row.update(gate_reason="contract_parse_failed", candidate_error="invalid_predicted_contract")
        return row, record
    if not candidates:
        row.update(gate_reason="empty_candidate_pool_preserved", candidate_error="empty_candidate_pool")
        return row, record
    mode = "semantic" if semantic else "legacy"
    before_score, before, _ = score_contract(original_sql, target, schema, contract_mode=mode)
    ranked = []
    for candidate in candidates:
        item = dict(candidate)
        score, actual, error = score_contract(item["sql"], target, schema, contract_mode=mode)
        if before is not None and actual is not None:
            improved, regressed = contract_field_changes(target, before, actual)
            slot_regressed = regresses_ordered_slot_subfield(target, before, actual)
        else:
            improved, regressed, slot_regressed = [], list(SEMANTIC_CONTRACT_FIELDS), True
        item.update(contract_score=score, improved=improved, regressed=regressed,
                    slot_regressed=slot_regressed, contract_error=error,
                    evidence_score=item["population_support"] + weight * score,
                    arity=actual is not None and actual.output_arity == target.output_arity,
                    limit=actual is not None and (target.limit is None or actual.limit == target.limit))
        ranked.append(item)
    if semantic:
        eligible = [c for c in ranked if c["arity"] and c["limit"]]
        best = min(eligible or ranked, key=lambda c: (-c["evidence_score"], -c["population_support"],
                   -c["contract_score"], len(c["sql"]), c["sql"]))
        allowed = best["arity"] and best["limit"]
    else:
        eligible = [c for c in ranked if c["population_support"] >= .1 and not c["regressed"]
                    and not c["slot_regressed"] and len(c["improved"]) >= 1]
        if eligible:
            best = min(eligible, key=lambda c: (-len(c["improved"]), -c["contract_score"],
                       -c["population_support"], len(c["sql"]), c["sql"]))
        else:
            best = min(ranked, key=lambda c: (-c["evidence_score"], -c["population_support"],
                       -c["contract_score"], len(c["sql"]), c["sql"]))
        allowed = bool(eligible) and best["population_support"] >= .1
    delta = best["contract_score"] - before_score
    accepted = bool(allowed and best["sql"] != original_sql and delta >= gain)
    chosen = best["sql"] if accepted else original_sql
    row.update(selected_sql=chosen, predicted_sql=chosen, repair_accepted=accepted,
               gate_reason="contract_evidence_commit" if accepted else "consensus_preserved",
               static_score_margin=delta if accepted else 0.0,
               selected_edits=["cem_archive_contract_rerank"] if accepted else [])
    record.update(selected_population_support=best["population_support"],
                  improved_fields=best["improved"], regressed_fields=best["regressed"],
                  accepted=accepted, selected_sql=chosen)
    return row, record
