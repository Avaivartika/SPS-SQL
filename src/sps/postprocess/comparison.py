"""Original post-correction comparison primitives."""
from __future__ import annotations
from typing import Any
from sps.postprocess.contracts import AnswerContract, extract_answer_contract
from sps.postprocess.sql_edits import contract_similarity, semantic_contract_similarity
from sps.postprocess.field_confidence import SEMANTIC_CONTRACT_FIELDS, semantic_contract_field_exact

def score_contract(
    sql: str,
    target: AnswerContract,
    schema: dict[str, Any],
    *,
    contract_mode: str,
) -> tuple[float, AnswerContract | None, str | None]:
    try:
        actual = extract_answer_contract(sql, schema)
        scorer = (
            semantic_contract_similarity
            if contract_mode == "semantic"
            else contract_similarity
        )
        return scorer(target, actual), actual, None
    except Exception as error:  # noqa: BLE001 - malformed model SQL is evidence
        return float("-inf"), None, f"{type(error).__name__}:{error}"

def contract_field_changes(
    target: AnswerContract,
    original: AnswerContract,
    candidate: AnswerContract,
) -> tuple[list[str], list[str]]:
    """Return exact semantic fields improved and regressed by a candidate."""

    target_payload = target.model_dump(mode="json")
    original_payload = original.model_dump(mode="json")
    candidate_payload = candidate.model_dump(mode="json")
    improved: list[str] = []
    regressed: list[str] = []
    for field in SEMANTIC_CONTRACT_FIELDS:
        original_exact = semantic_contract_field_exact(
            original_payload,
            target_payload,
            field,
        )
        candidate_exact = semantic_contract_field_exact(
            candidate_payload,
            target_payload,
            field,
        )
        if candidate_exact and not original_exact:
            improved.append(field)
        elif original_exact and not candidate_exact:
            regressed.append(field)
    return improved, regressed

def regresses_ordered_slot_subfield(
    target: AnswerContract,
    original: AnswerContract,
    candidate: AnswerContract,
) -> bool:
    """Detect partial slot regressions hidden by whole-slot inexactness."""

    if not (
        len(target.ordered_slots)
        == len(original.ordered_slots)
        == len(candidate.ordered_slots)
    ):
        return False

    def normalized_set(values: list[str]) -> set[str]:
        return {value.casefold() for value in values}

    for expected, before, after in zip(
        target.ordered_slots,
        original.ordered_slots,
        candidate.ordered_slots,
        strict=True,
    ):
        components = (
            (expected.operation, before.operation, after.operation),
            (
                expected.aggregate_distinct,
                before.aggregate_distinct,
                after.aggregate_distinct,
            ),
            (
                normalized_set(expected.source_columns),
                normalized_set(before.source_columns),
                normalized_set(after.source_columns),
            ),
            (
                normalized_set(expected.operators),
                normalized_set(before.operators),
                normalized_set(after.operators),
            ),
        )
        if any(old == wanted and new != wanted for wanted, old, new in components):
            return True
    return False
