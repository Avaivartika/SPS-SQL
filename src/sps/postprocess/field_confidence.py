"""Original post-correction field confidence primitives."""
from __future__ import annotations
from collections import Counter
from typing import Any

STRUCTURAL_EDIT_FAMILIES = (
    "select_distinct",
    "limit",
    "order_removal",
    "projection_arity",
    "aggregate_distinct",
)

SEMANTIC_CONTRACT_FIELDS = (
    "output_arity",
    "ordered_slots",
    "row_mode",
    "select_distinct",
    "order_required",
    "limit",
    "tables",
    "predicate_columns",
    "predicate_operators",
    "predicate_values",
    "group_by_columns",
    "having_required",
    "order_by_columns",
    "order_directions",
    "set_operation",
)

_UNORDERED_CASEFOLD_FIELDS = {
    "tables",
    "predicate_columns",
    "predicate_operators",
    "group_by_columns",
    "order_by_columns",
}

def edit_family(edits: list[str] | tuple[str, ...]) -> str | None:
    """Map a single minimal AST edit to the field it changes."""

    if len(edits) != 1:
        return None
    edit = str(edits[0])
    if edit.startswith("slot_") and edit.endswith("_aggregate_distinct"):
        return "aggregate_distinct"
    if edit in STRUCTURAL_EDIT_FAMILIES:
        return edit
    return None

def contract_field_action(
    prediction: dict[str, Any] | None,
    family: str,
) -> str:
    """Return the target-state bucket for a prospective minimal edit.

    A field can be globally accurate while one edit direction is unreliable.
    For example, non-aggregate answer slots make ``aggregate_distinct`` look
    easy even though the actionable ``COUNT(DISTINCT x) -> COUNT(x)`` decision
    is harder.  These buckets let held-out calibration condition on the state
    the edit would actually impose.
    """

    if prediction is None:
        return "target_missing"
    if family == "select_distinct":
        value = prediction.get("select_distinct")
        return (
            f"target_{str(value).lower()}"
            if isinstance(value, bool)
            else "target_missing"
        )
    if family == "limit":
        return "target_none" if prediction.get("limit") is None else "target_present"
    if family == "order_removal":
        value = prediction.get("order_required")
        return (
            f"target_{str(value).lower()}"
            if isinstance(value, bool)
            else "target_missing"
        )
    if family == "projection_arity":
        value = prediction.get("output_arity")
        if not isinstance(value, int) or isinstance(value, bool):
            return "target_missing"
        return "target_one" if value == 1 else "target_multi"
    if family == "aggregate_distinct":
        values = [
            slot.get("aggregate_distinct")
            for slot in prediction.get("ordered_slots", [])
            if isinstance(slot, dict)
            and isinstance(slot.get("aggregate_distinct"), bool)
        ]
        if not values:
            return "target_missing"
        if all(value is False for value in values):
            return "target_false"
        if all(value is True for value in values):
            return "target_true"
        return "target_mixed"
    raise ValueError(f"unsupported edit family: {family}")

def semantic_contract_field_exact(
    prediction: dict[str, Any] | None,
    reference: dict[str, Any],
    field: str,
) -> bool:
    """Compare one predicted semantic-contract field to held-out supervision."""

    if field not in SEMANTIC_CONTRACT_FIELDS:
        raise ValueError(f"unsupported semantic contract field: {field}")
    if prediction is None:
        return False
    predicted = prediction.get(field)
    expected = reference.get(field)
    if field in _UNORDERED_CASEFOLD_FIELDS:
        return {str(value).casefold() for value in (predicted or [])} == {
            str(value).casefold() for value in (expected or [])
        }
    return predicted == expected

def apply_field_confidence_gate(
    rows: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    disabled_families: set[str] | None = None,
    action_conditioned_families: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply validation-frozen structural thresholds without dev labels."""

    score_by_id = {int(value["question_id"]): value for value in scores}
    expected = {int(value["source_id"]) for value in rows}
    if set(score_by_id) != expected:
        raise ValueError("confidence score IDs do not match evaluation IDs")
    fields = calibration["fields"]
    disabled = disabled_families or set()
    action_conditioned = action_conditioned_families or set()
    unknown_action_families = action_conditioned.difference(STRUCTURAL_EDIT_FAMILIES)
    if unknown_action_families:
        raise ValueError(
            f"unsupported action-conditioned families: {sorted(unknown_action_families)}"
        )
    output: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    kept_by_family: Counter[str] = Counter()
    for source in rows:
        row = dict(source)
        source_id = int(row["source_id"])
        family = edit_family(list(row.get("selected_edits", [])))
        score = float(score_by_id[source_id]["mean_logprob"])
        static_accepted = bool(row.get("repair_accepted"))
        field = fields.get(family) if family is not None else None
        action = (
            contract_field_action(row.get("predicted_contract"), family)
            if family is not None
            else None
        )
        calibration_scope = "field"
        calibration_cell = field
        if family in action_conditioned and field is not None:
            calibration_scope = "action"
            calibration_cell = (field.get("actions") or {}).get(action)
        recommended = (
            calibration_cell.get("recommended")
            if calibration_cell is not None
            else None
        )
        threshold = float(recommended["threshold"]) if recommended is not None else None
        accept = bool(
            static_accepted
            and family not in disabled
            and threshold is not None
            and score >= threshold
        )
        if accept:
            reason = "field_confidence_passed"
            assert family is not None
            kept_by_family[family] += 1
        elif not static_accepted:
            reason = "no_static_repair"
        elif family is None:
            reason = "unsupported_edit_family"
        elif family in disabled:
            reason = "edit_family_disabled"
        elif threshold is None:
            reason = "field_not_calibrated"
        else:
            reason = "field_confidence_below_threshold"
        reasons[reason] += 1
        row["field_confidence_gate"] = {
            "kept": accept,
            "reason": reason,
            "edit_family": family,
            "contract_mean_logprob": score,
            "threshold": threshold,
            "action": action,
            "calibration_scope": calibration_scope,
        }
        row["predicted_sql"] = row["selected_sql"] if accept else row["original_sql"]
        output.append(row)

    summary = {
        "artifact_kind": "minimal_contract_field_confidence_gate_v1",
        "uses_evaluation_labels_for_selection": False,
        "samples": len(rows),
        "static_repairs_entered": sum(
            bool(value.get("repair_accepted")) for value in rows
        ),
        "repairs_kept": sum(kept_by_family.values()),
        "kept_by_family": dict(sorted(kept_by_family.items())),
        "disabled_families": sorted(disabled),
        "action_conditioned_families": sorted(action_conditioned),
        "reasons": dict(sorted(reasons.items())),
    }
    return output, summary
