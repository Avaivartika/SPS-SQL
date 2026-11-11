"""Original post-correction confidence primitives."""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
import sqlglot
from sqlglot import exp
from sps.postprocess.contracts import AnswerContract, extract_answer_contract
from sps.postprocess.field_confidence import SEMANTIC_CONTRACT_FIELDS, semantic_contract_field_exact

@dataclass(frozen=True)
class ConfidenceOptions:
    mode: str
    block_confident_regressions: bool
    block_decision_field: tuple[str, ...]
    regression_only_field: tuple[str, ...]
    hard_regression_field: tuple[str, ...]
    require_passing_decision_field: tuple[str, ...]
    capability_boundary_gate: bool
    capability_evidence: str
    minimum_passing_decision_fields: int
    allow_explicit_literal_anchor: bool
    allow_cohort_average_rewrite: bool
    allow_percentage_who_scope: bool

def _decision_fields(
    target: AnswerContract,
    original: AnswerContract,
    selected: AnswerContract,
) -> list[str]:
    target_payload = target.model_dump(mode="json")
    original_payload = original.model_dump(mode="json")
    selected_payload = selected.model_dump(mode="json")
    return [
        field
        for field in SEMANTIC_CONTRACT_FIELDS
        if semantic_contract_field_exact(selected_payload, target_payload, field)
        and not semantic_contract_field_exact(original_payload, target_payload, field)
    ]

def _regressed_fields(
    target: AnswerContract,
    original: AnswerContract,
    selected: AnswerContract,
) -> list[str]:
    target_payload = target.model_dump(mode="json")
    original_payload = original.model_dump(mode="json")
    selected_payload = selected.model_dump(mode="json")
    return [
        field
        for field in SEMANTIC_CONTRACT_FIELDS
        if semantic_contract_field_exact(original_payload, target_payload, field)
        and not semantic_contract_field_exact(selected_payload, target_payload, field)
    ]

def _needs_unrepresented_offset(question: str) -> bool:
    return any(
        int(match.group(1)) > 1
        for match in re.finditer(r"\b(\d+)(?:st|nd|rd|th)\b", question, re.IGNORECASE)
    )

def _case_only_predicate_rewrite(
    original: AnswerContract,
    selected: AnswerContract,
) -> bool:
    original_values = sorted(original.predicate_values)
    selected_values = sorted(selected.predicate_values)
    return original_values != selected_values and sorted(
        value.casefold() for value in original_values
    ) == sorted(value.casefold() for value in selected_values)

def _is_arity_only_evidence(positive_decision_fields: list[str]) -> bool:
    """Reject commits justified only by a predicted output-width change."""

    return positive_decision_fields == ["output_arity"]

def _capability_evidence(fields: list[str], passes: dict[str, bool], mode: str) -> list[str]:
    """Uncalibrated changes cannot bypass an evidence-based safety boundary."""
    if mode == "legacy_uncalibrated":
        return list(fields)
    if mode != "calibrated":
        raise ValueError(f"unsupported capability evidence mode: {mode}")
    return [field for field in fields if passes.get(field, False)]

def _regresses_ordered_slot_subfield(
    target: AnswerContract,
    original: AnswerContract,
    selected: AnswerContract,
) -> bool:
    """Detect a hidden regression inside an otherwise still-inexact slot."""

    if not (
        len(target.ordered_slots)
        == len(original.ordered_slots)
        == len(selected.ordered_slots)
    ):
        return False

    def normalized_set(values: list[str]) -> set[str]:
        return {value.casefold() for value in values}

    for expected, before, after in zip(
        target.ordered_slots,
        original.ordered_slots,
        selected.ordered_slots,
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

def _explicit_literal_anchor(
    question: str,
    target: AnswerContract,
    selected_sql: str,
    decision_fields: list[str],
    regressed_fields: list[str],
) -> bool:
    if regressed_fields or not {
        "predicate_columns",
        "predicate_operators",
    }.issubset(decision_fields):
        return False
    quoted_values = {
        match.group(2).strip()
        for match in re.finditer(r"(['\"])(.{3,}?)\1", question)
        if match.group(2).strip()
    }
    selected_folded = selected_sql.casefold()
    target_values = {value.casefold() for value in target.predicate_values}
    return any(
        value.casefold() in selected_folded
        and f"string:{value}".casefold() in target_values
        for value in quoted_values
    )

def _cohort_average_rewrite(
    target: AnswerContract,
    original_sql: str,
    selected_sql: str,
) -> bool:
    target_payload = target.model_dump(mode="json")
    if target_payload["row_mode"] != "scalar" or target_payload["group_by_columns"]:
        return False
    try:
        original = sqlglot.parse_one(original_sql, read="sqlite")
        selected = sqlglot.parse_one(selected_sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return False
    if not isinstance(original, exp.Select) or not isinstance(selected, exp.Select):
        return False
    if original.args.get("group") is None or selected.args.get("group") is not None:
        return False
    if len(selected.expressions) != int(target_payload["output_arity"]):
        return False
    for projection in selected.expressions:
        aggregates = list(projection.find_all(exp.AggFunc))
        if len(aggregates) != 1 or not isinstance(aggregates[0], exp.Avg):
            return False
    return any(
        isinstance(subquery.this, exp.Select)
        and subquery.this.args.get("group") is not None
        and subquery.this.args.get("having") is not None
        for subquery in selected.find_all(exp.Subquery)
    )

def _percentage_who_scope_alignment(
    question: str,
    target: AnswerContract,
    original: AnswerContract,
    selected: AnswerContract,
) -> bool:
    if not re.search(
        r"\bpercent(?:age)?\s+of\b.+\bwho\b",
        question,
        re.IGNORECASE,
    ):
        return False
    if not any(slot.operation == "percentage" for slot in target.ordered_slots):
        return False
    target_payload = target.model_dump(mode="json")
    return (
        selected.model_dump(mode="json") == target_payload
        and original.model_dump(mode="json") != target_payload
    )

def _introduces_percentage_denominator_filter(
    target: AnswerContract,
    original_sql: str,
    selected_sql: str,
) -> bool:
    if not any(slot.operation == "percentage" for slot in target.ordered_slots):
        return False
    try:
        original = sqlglot.parse_one(original_sql, read="sqlite")
        selected = sqlglot.parse_one(selected_sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return False
    if not isinstance(original, exp.Select) or not isinstance(selected, exp.Select):
        return False
    return original.args.get("where") is None and selected.args.get("where") is not None

def _introduces_order_coercion(original_sql: str, selected_sql: str) -> bool:
    try:
        original = sqlglot.parse_one(original_sql, read="sqlite")
        selected = sqlglot.parse_one(selected_sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return False
    original_order = original.args.get("order")
    selected_order = selected.args.get("order")
    return bool(
        selected_order is not None
        and selected_order.find(exp.Cast) is not None
        and (original_order is None or original_order.find(exp.Cast) is None)
    )

def _unrepresented_nested_order_flip(
    target: AnswerContract,
    original_sql: str,
    selected_sql: str,
) -> bool:
    if target.order_required:
        return False
    try:
        original = sqlglot.parse_one(original_sql, read="sqlite")
        selected = sqlglot.parse_one(selected_sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return False

    def directions(root: exp.Expression) -> dict[str, set[bool]]:
        result: dict[str, set[bool]] = {}
        for order in root.find_all(exp.Order):
            for ordered in order.expressions:
                column = ordered.find(exp.Column)
                if column is None or not column.name:
                    continue
                result.setdefault(column.name.casefold(), set()).add(
                    bool(ordered.args.get("desc"))
                )
        return result

    original_directions = directions(original)
    selected_directions = directions(selected)
    return any(
        original_directions[column].isdisjoint(selected_directions[column])
        for column in original_directions.keys() & selected_directions.keys()
    )

def _birth_extremum_conflict(question: str, selected_sql: str) -> bool:
    question_folded = question.casefold()
    if "born" not in question_folded and "birthday" not in question_folded:
        return False
    selected_folded = selected_sql.casefold()
    return ("youngest" in question_folded and "min(" in selected_folded) or (
        "oldest" in question_folded and "max(" in selected_folded
    )

def _collapses_tie_preserving_extremum(
    original_sql: str,
    selected_sql: str,
) -> bool:
    tie_preserving = re.search(
        r"=\s*\(\s*select\s+(?:max|min)\s*\(",
        original_sql,
        re.IGNORECASE,
    )
    selected_folded = selected_sql.casefold()
    return bool(
        tie_preserving
        and "order by" in selected_folded
        and re.search(r"\blimit\s+1\b", selected_folded)
    )

def gate_proposals(rows: list[dict], score_rows: dict[int, dict],
                   calibration: dict, schemas: dict[str, dict],
                   options: ConfidenceOptions) -> dict:
    args = options
    expected = {int(row["source_id"]) for row in rows}
    if set(score_rows) != expected:
        raise ValueError("Confidence scores must cover all proposal question IDs")
    if args.minimum_passing_decision_fields < 1:
        raise ValueError("Minimum passing decision fields must be positive")
    score_mode = str(calibration.get("score_mode", "whole"))

    output_rows: list[dict[str, Any]] = []
    predictions: dict[str, str] = {}
    reasons: Counter[str] = Counter()
    kept_fields: Counter[str] = Counter()
    for source in rows:
        row = dict(source)
        source_id = int(row["source_id"])
        original_sql = str(row["original_sql"])
        selected_sql = str(row["selected_sql"])
        accepted = bool(row.get("repair_accepted"))
        decision_fields: list[str] = []
        positive_decision_fields: list[str] = []
        capability_evidence_fields: list[str] = []
        regressed_fields: list[str] = []
        capability_boundary: str | None = None
        evidence_override: str | None = None
        thresholds: dict[str, float | None] = {}
        decision_scores: dict[str, float] = {}
        passes: dict[str, bool] = {}
        error: str | None = None
        if accepted:
            try:
                target = AnswerContract.model_validate(row["predicted_contract"])
                schema = schemas[str(row["db_id"])]
                original = extract_answer_contract(original_sql, schema)
                selected = extract_answer_contract(selected_sql, schema)
                decision_fields = _decision_fields(target, original, selected)
                positive_decision_fields = [
                    field
                    for field in decision_fields
                    if field not in args.regression_only_field
                ]
                regressed_fields = _regressed_fields(target, original, selected)
                if args.capability_boundary_gate:
                    if _needs_unrepresented_offset(str(row["question"])):
                        capability_boundary = "unrepresented_ordinal_offset"
                    elif args.capability_evidence == "legacy_uncalibrated" and _is_arity_only_evidence(positive_decision_fields):
                        capability_boundary = "arity_only_evidence"
                    elif _regresses_ordered_slot_subfield(target, original, selected):
                        capability_boundary = "ordered_slot_subfield_regression"
                    elif _case_only_predicate_rewrite(original, selected):
                        capability_boundary = "case_only_predicate_rewrite"
                    elif _introduces_percentage_denominator_filter(
                        target,
                        original_sql,
                        selected_sql,
                    ):
                        capability_boundary = "percentage_denominator_scope"
                    elif _introduces_order_coercion(original_sql, selected_sql):
                        capability_boundary = "order_coercion_unrepresented"
                    elif _unrepresented_nested_order_flip(
                        target,
                        original_sql,
                        selected_sql,
                    ):
                        capability_boundary = "nested_order_direction_unrepresented"
                    elif _birth_extremum_conflict(
                        str(row["question"]),
                        selected_sql,
                    ):
                        capability_boundary = "birth_extremum_conflict"
                    elif _collapses_tie_preserving_extremum(
                        original_sql,
                        selected_sql,
                    ):
                        capability_boundary = "tie_policy_unrepresented"
                if args.allow_explicit_literal_anchor and _explicit_literal_anchor(
                    str(row["question"]),
                    target,
                    selected_sql,
                    decision_fields,
                    regressed_fields,
                ):
                    evidence_override = "explicit_literal_anchor"
                elif args.allow_cohort_average_rewrite and _cohort_average_rewrite(
                    target,
                    original_sql,
                    selected_sql,
                ):
                    evidence_override = "cohort_average_rewrite"
                elif (
                    args.allow_percentage_who_scope
                    and _percentage_who_scope_alignment(
                        str(row["question"]),
                        target,
                        original,
                        selected,
                    )
                ):
                    evidence_override = "percentage_who_scope_alignment"
                for field in {*decision_fields, *regressed_fields}:
                    recommended = (
                        calibration["fields"].get(field, {}).get("recommended")
                    )
                    threshold = (
                        float(recommended["threshold"])
                        if recommended is not None
                        else None
                    )
                    thresholds[field] = threshold
                    score_row = score_rows[source_id]
                    field_score = (score_row.get("field_mean_logprobs") or {}).get(
                        field
                    )
                    decision_score = (
                        float(field_score)
                        if score_mode == "field" and field_score is not None
                        else float(score_row["mean_logprob"])
                    )
                    passes[field] = bool(
                        threshold is not None and decision_score >= threshold
                    )
                    decision_scores[field] = decision_score
                capability_evidence_fields = _capability_evidence(
                    positive_decision_fields, passes, args.capability_evidence
                )
                if (args.capability_boundary_gate
                    and args.capability_evidence == "calibrated"
                    and capability_boundary is None
                    and _is_arity_only_evidence(capability_evidence_fields)):
                    capability_boundary = "arity_only_evidence"
            except (KeyError, TypeError, ValueError) as exception:
                error = f"{type(exception).__name__}:{exception}"

        if not accepted:
            keep = False
            reason = "no_semantic_commit"
        elif error is not None:
            keep = False
            reason = "confidence_analysis_failed"
        elif not positive_decision_fields:
            keep = False
            reason = "no_calibrated_decision_field"
        elif (
            set(args.block_decision_field) & set(decision_fields)
            and evidence_override != "cohort_average_rewrite"
        ):
            keep = False
            reason = "blocked_decision_field"
        elif capability_boundary is not None:
            keep = False
            reason = capability_boundary
        elif set(args.hard_regression_field) & set(regressed_fields):
            keep = False
            reason = "hard_structural_regression"
        elif args.block_confident_regressions and any(
            passes[field] for field in regressed_fields
        ):
            keep = False
            reason = "confident_field_regression"
        elif evidence_override is not None:
            keep = True
            reason = evidence_override
        elif any(
            field not in positive_decision_fields or not passes.get(field, False)
            for field in args.require_passing_decision_field
        ):
            keep = False
            reason = "required_decision_field_failed"
        elif (
            args.mode == "all"
            and len(positive_decision_fields) >= args.minimum_passing_decision_fields
            and all(passes[field] for field in positive_decision_fields)
        ):
            keep = True
            reason = "all_field_confidence_passed"
        elif (
            args.mode == "any"
            and sum(passes[field] for field in positive_decision_fields)
            >= args.minimum_passing_decision_fields
        ):
            keep = True
            reason = "any_field_confidence_passed"
        else:
            keep = False
            reason = "field_confidence_below_threshold"
        if keep:
            for field in decision_fields:
                kept_fields[field] += 1
        final_sql = selected_sql if keep else original_sql
        predictions[str(source_id)] = final_sql
        reasons[reason] += 1
        row["predicted_sql"] = final_sql
        row["semantic_confidence_gate"] = {
            "kept": keep,
            "reason": reason,
            "mode": args.mode,
            "minimum_passing_decision_fields": args.minimum_passing_decision_fields,
            "mean_logprob": float(score_rows[source_id]["mean_logprob"]),
            "score_mode": score_mode,
            "decision_fields": decision_fields,
            "positive_decision_fields": positive_decision_fields,
            "capability_evidence": args.capability_evidence,
            "capability_evidence_fields": capability_evidence_fields,
            "regressed_fields": regressed_fields,
            "capability_boundary": capability_boundary,
            "evidence_override": evidence_override,
            "thresholds": thresholds,
            "decision_scores": decision_scores,
            "passes": passes,
            "error": error,
        }
        output_rows.append(row)

    return {"rows": output_rows, "predictions": predictions,
            "reasons": reasons, "kept_fields": kept_fields}
