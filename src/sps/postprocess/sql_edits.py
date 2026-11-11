"""Original post-correction sql edits primitives."""
from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import Any
import sqlglot
from sqlglot import exp
from sps.postprocess.contracts import AnswerContract, extract_answer_contract
from sps.postprocess.sqlite_rendering import NativeSQLite

@dataclass(frozen=True)
class ContractCandidate:
    sql: str
    edits: tuple[str, ...]
    contract: AnswerContract
    static_score: float
    is_original: bool = False

@dataclass(frozen=True)
class ContractSelection:
    selected: ContractCandidate
    original: ContractCandidate
    accepted_repair: bool
    reason: str
    score_margin: float

def generate_source_preserving_candidates(
    predicted_sql: str,
    target: AnswerContract,
    schema: dict[str, Any],
) -> list[ContractCandidate]:
    """Replace root-projection columns without rebuilding its expression.

    This path is intentionally narrow.  It applies only when arity and slot
    operations already agree, preserving MAX, COUNT(DISTINCT), CAST, CASE,
    arithmetic topology, ordering, limits, grouping, and filters verbatim.
    """

    parsed = sqlglot.parse_one(predicted_sql, read=NativeSQLite)
    if not isinstance(parsed, exp.Select):
        raise TypeError("source-preserving candidates require a SELECT root")
    original_contract = extract_answer_contract(predicted_sql, schema)
    if original_contract.output_arity != target.output_arity or [
        value.operation for value in original_contract.ordered_slots
    ] != [value.operation for value in target.ordered_slots]:
        return [_contract_candidate(predicted_sql, (), schema, is_original=True)]

    mismatched_slots = [
        index
        for index, (observed, expected) in enumerate(
            zip(
                original_contract.ordered_slots,
                target.ordered_slots,
                strict=True,
            )
        )
        if [value.casefold() for value in observed.source_columns]
        != [value.casefold() for value in expected.source_columns]
    ]
    if not mismatched_slots:
        return [_contract_candidate(predicted_sql, (), schema, is_original=True)]

    raw: list[tuple[exp.Select, tuple[str, ...]]] = [(parsed.copy(), ())]
    all_slots = _source_substitution_variant(
        parsed,
        target,
        slot_indices=mismatched_slots,
    )
    if all_slots is not None:
        raw.append((all_slots, ("source_substitution",)))
    for slot_index in mismatched_slots:
        partial = _source_substitution_variant(
            parsed,
            target,
            slot_indices=(slot_index,),
        )
        if partial is not None:
            raw.append((partial, (f"slot_{slot_index}_source_substitution",)))

    candidates: list[ContractCandidate] = []
    seen: set[str] = set()
    for tree, edits in raw:
        sql = tree.sql(dialect=NativeSQLite, pretty=False) if edits else predicted_sql
        canonical = sqlglot.parse_one(sql, read=NativeSQLite).sql(
            dialect=NativeSQLite, normalize=True, pretty=False
        )
        if canonical in seen:
            continue
        seen.add(canonical)
        try:
            contract = extract_answer_contract(sql, schema)
        except Exception:  # noqa: BLE001, S112 - discard malformed candidate SQL
            continue
        candidates.append(
            ContractCandidate(
                sql=sql,
                edits=edits,
                contract=contract,
                static_score=contract_similarity(target, contract),
                is_original=not edits,
            )
        )
    candidates.sort(
        key=lambda value: (
            -value.static_score,
            len(value.edits),
            not value.is_original,
            value.sql,
        )
    )
    return candidates

def generate_minimal_contract_candidates(
    predicted_sql: str,
    target: AnswerContract,
    schema: dict[str, Any],
) -> list[ContractCandidate]:
    """Generate one-family AST edits without rebuilding projections."""

    parsed = sqlglot.parse_one(predicted_sql, read=NativeSQLite)
    if not isinstance(parsed, exp.Select):
        raise TypeError("minimal contract candidates require a SELECT root")
    raw: list[tuple[exp.Select, tuple[str, ...]]] = [(parsed.copy(), ())]

    current_distinct = parsed.args.get("distinct") is not None
    if current_distinct != target.select_distinct:
        variant = parsed.copy()
        variant.set("distinct", exp.Distinct() if target.select_distinct else None)
        raw.append((variant, ("select_distinct",)))

    current_limit = _literal_select_limit(parsed)
    if current_limit != target.limit:
        variant = parsed.copy()
        variant.set(
            "limit",
            exp.Limit(expression=exp.Literal.number(target.limit))
            if target.limit is not None
            else None,
        )
        if target.limit is None:
            variant.set("offset", None)
        raw.append((variant, ("limit",)))

    # Keep this diagnostic candidate in the pool so an order mismatch is not
    # misattributed to another edit family.  The commit policy may abstain from
    # this family because removing ORDER BY is either result-invariant (no
    # LIMIT) or unsafe (with LIMIT).
    if not target.order_required and parsed.args.get("order") is not None:
        variant = parsed.copy()
        variant.set("order", None)
        raw.append((variant, ("order_removal",)))

    raw.extend(
        (value, ("projection_arity",))
        for value in _arity_variants(parsed, target.output_arity)
    )
    for slot_index, slot in enumerate(target.ordered_slots):
        variant = _aggregate_distinct_variant(parsed, slot_index, slot.aggregate_distinct)
        if variant is not None:
            raw.append((variant, (f"slot_{slot_index}_aggregate_distinct",)))

    for candidate in generate_source_preserving_candidates(predicted_sql, target, schema):
        if candidate.is_original:
            continue
        tree = sqlglot.parse_one(candidate.sql, read=NativeSQLite)
        if isinstance(tree, exp.Select):
            raw.append((tree, candidate.edits))

    candidates: list[ContractCandidate] = []
    seen: set[str] = set()
    for tree, edits in raw:
        sql = tree.sql(dialect=NativeSQLite, pretty=False) if edits else predicted_sql
        canonical = sqlglot.parse_one(sql, read=NativeSQLite).sql(
            dialect=NativeSQLite, normalize=True, pretty=False
        )
        if canonical in seen:
            continue
        seen.add(canonical)
        try:
            contract = extract_answer_contract(sql, schema)
        except Exception:  # noqa: BLE001, S112 - discard malformed candidate SQL
            continue
        candidates.append(
            ContractCandidate(
                sql=sql,
                edits=edits,
                contract=contract,
                static_score=contract_similarity(target, contract),
                is_original=not edits,
            )
        )
    candidates.sort(
        key=lambda value: (
            -value.static_score,
            len(value.edits),
            not value.is_original,
            value.sql,
        )
    )
    return candidates

def select_conservative_candidate(
    candidates: list[ContractCandidate],
    target: AnswerContract,
    *,
    minimum_margin: float = 2.0,
) -> ContractSelection:
    """Repair only a localized mismatch with an exact contract-shaped option."""

    originals = [value for value in candidates if value.is_original]
    if len(originals) != 1:
        raise ValueError("candidate pool must contain exactly one original")
    original = originals[0]
    best = candidates[0]
    margin = best.static_score - original.static_score
    if best.is_original:
        return ContractSelection(best, original, False, "original_best", margin)
    if margin < minimum_margin:
        return ContractSelection(original, original, False, "insufficient_margin", margin)
    if len(best.edits) > 2:
        return ContractSelection(original, original, False, "too_many_edit_families", margin)
    if best.contract.output_arity != target.output_arity:
        return ContractSelection(original, original, False, "arity_not_satisfied", margin)
    best_operations = [value.operation for value in best.contract.ordered_slots]
    target_operations = [value.operation for value in target.ordered_slots]
    if best_operations != target_operations:
        return ContractSelection(original, original, False, "operations_not_satisfied", margin)
    return ContractSelection(best, original, True, "localized_contract_repair", margin)

def contract_similarity(target: AnswerContract, actual: AnswerContract) -> float:
    """Interpretable static score used before counterfactual validation."""

    score = 0.0
    score += 4.0 if target.output_arity == actual.output_arity else 0.0
    score += 2.0 if target.row_mode == actual.row_mode else 0.0
    score += 0.5 if target.select_distinct == actual.select_distinct else 0.0
    score += 0.5 if target.order_required == actual.order_required else 0.0
    score += 0.5 if target.limit == actual.limit else 0.0
    width = max(len(target.ordered_slots), len(actual.ordered_slots))
    for index in range(width):
        if index >= len(target.ordered_slots) or index >= len(actual.ordered_slots):
            continue
        expected = target.ordered_slots[index]
        observed = actual.ordered_slots[index]
        if expected.operation == observed.operation:
            score += 2.0
        expected_sources = {value.casefold() for value in expected.source_columns}
        observed_sources = {value.casefold() for value in observed.source_columns}
        if expected_sources or observed_sources:
            score += len(expected_sources & observed_sources) / len(
                expected_sources | observed_sources
            )
        else:
            score += 1.0
        if expected.aggregate_distinct == observed.aggregate_distinct:
            score += 0.5
    return score

def semantic_contract_similarity(
    target: AnswerContract,
    actual: AnswerContract,
) -> float:
    """Score output shape plus the query semantics used to produce it.

    The original answer-contract score intentionally looked only at result
    shape.  This extension remains interpretable, but also distinguishes SQL
    candidates that return the same shaped answer through different tables,
    predicates, grouping, ordering, or set operations.
    """

    score = contract_similarity(target, actual)
    score += 2.0 * _set_similarity(target.tables, actual.tables)
    score += 2.0 * _set_similarity(
        target.predicate_columns,
        actual.predicate_columns,
    )
    score += _set_similarity(
        target.predicate_operators,
        actual.predicate_operators,
    )
    score += 0.5 * _set_similarity(
        target.predicate_values,
        actual.predicate_values,
        case_sensitive=True,
    )
    score += 1.5 * _set_similarity(
        target.group_by_columns,
        actual.group_by_columns,
    )
    score += 0.5 if target.having_required == actual.having_required else 0.0
    score += _set_similarity(
        target.order_by_columns,
        actual.order_by_columns,
    )
    score += 0.5 * _sequence_similarity(
        target.order_directions,
        actual.order_directions,
    )
    score += 1.0 if target.set_operation == actual.set_operation else 0.0
    return score

def _set_similarity(
    expected: list[str], observed: list[str], *, case_sensitive: bool = False,
) -> float:
    # SQL identifiers can ignore case, but literal text must retain its content.
    # This is evidence similarity, not a claim of SQL equivalence under every collation.
    normalize = str if case_sensitive else str.casefold
    expected_keys = {normalize(value) for value in expected}
    observed_keys = {normalize(value) for value in observed}
    if not expected_keys and not observed_keys:
        return 1.0
    return len(expected_keys & observed_keys) / len(expected_keys | observed_keys)

def _sequence_similarity(expected: list[str], observed: list[str]) -> float:
    if not expected and not observed:
        return 1.0
    width = max(len(expected), len(observed))
    if width == 0:
        return 1.0
    matches = sum(
        left.casefold() == right.casefold()
        for left, right in zip(expected, observed, strict=False)
    )
    return matches / width

def _source_substitution_variant(
    query: exp.Select,
    target: AnswerContract,
    *,
    slot_indices: Any,
) -> exp.Select | None:
    aliases = _table_aliases(query)
    selected = {int(value) for value in slot_indices}
    result = query.copy()
    expressions = [value.copy() for value in query.expressions]
    changed = False
    for slot_index in selected:
        if slot_index >= len(expressions) or slot_index >= len(target.ordered_slots):
            return None
        projection = query.expressions[slot_index]
        # Use the same left-to-right recursive order as grounding labels.
        # ``find_all`` uses SQLGlot's DFS stack order, which can visit a
        # division denominator before a CASE expression in its numerator.
        root_columns = _ordered_projection_columns(projection)
        unique_keys: list[str] = []
        for column in root_columns:
            key = _column_key(column)
            if key not in unique_keys:
                unique_keys.append(key)
        target_columns = [
            _compile_column(value, aliases)
            for value in target.ordered_slots[slot_index].source_columns
        ]
        if any(value is None for value in target_columns):
            return None
        replacements = [value for value in target_columns if value is not None]
        if not unique_keys and not replacements:
            continue
        if len(unique_keys) != len(replacements) or not unique_keys:
            return None
        mapping = dict(zip(unique_keys, replacements, strict=True))

        def replace(
            node: exp.Expression,
            replacements_by_column: dict[str, exp.Column] = mapping,
        ) -> exp.Expression:
            # Scalar subqueries have their own projection/filter semantics.
            if isinstance(node, exp.Subquery):
                return node.copy()
            if isinstance(node, exp.Column):
                replacement = replacements_by_column.get(_column_key(node))
                if replacement is not None:
                    return replacement.copy()
            return node

        rewritten = projection.transform(replace, copy=True)
        if rewritten.sql(dialect=NativeSQLite) != projection.sql(dialect=NativeSQLite):
            changed = True
        expressions[slot_index] = rewritten
    if not changed:
        return None
    result.set("expressions", expressions)
    return result

def _column_key(column: exp.Column) -> str:
    return column.sql(dialect=NativeSQLite, normalize=True, pretty=False).casefold()

def _ordered_projection_columns(expression: exp.Expression) -> list[exp.Column]:
    if isinstance(expression, exp.Column):
        return [expression]
    if isinstance(expression, exp.Subquery):
        return []
    values: list[exp.Column] = []
    for child in expression.iter_expressions():
        values.extend(_ordered_projection_columns(child))
    return values

def _contract_candidate(
    sql: str,
    edits: tuple[str, ...],
    schema: dict[str, Any],
    *,
    is_original: bool,
) -> ContractCandidate:
    return ContractCandidate(
        sql=sql,
        edits=edits,
        contract=extract_answer_contract(sql, schema),
        static_score=0.0,
        is_original=is_original,
    )

def _literal_select_limit(query: exp.Select) -> int | None:
    limit = query.args.get("limit")
    if limit is None:
        return None
    value = limit.args.get("expression")
    if isinstance(value, exp.Literal) and not value.is_string:
        try:
            return int(str(value.this))
        except ValueError:
            return None
    return None

def _aggregate_distinct_variant(
    query: exp.Select,
    slot_index: int,
    desired: bool | None,
) -> exp.Select | None:
    if desired is None or slot_index >= len(query.expressions):
        return None
    projection = query.expressions[slot_index]
    aggregates = [
        value
        for value in projection.find_all(exp.AggFunc)
        if value.find_ancestor(exp.Select) is query
    ]
    if len(aggregates) != 1:
        return None
    aggregate = aggregates[0]
    current = isinstance(aggregate.this, exp.Distinct)
    if current == desired:
        return None
    result = query.copy()
    result_projection = result.expressions[slot_index]
    result_aggregates = [
        value
        for value in result_projection.find_all(exp.AggFunc)
        if value.find_ancestor(exp.Select) is result
    ]
    if len(result_aggregates) != 1:
        return None
    result_aggregate = result_aggregates[0]
    if desired:
        operand = result_aggregate.this or exp.Star()
        # SQLite rejects COUNT(DISTINCT *), and there is no semantics-preserving
        # way to infer which row identity the question intends here.
        if isinstance(operand, exp.Star):
            return None
        result_aggregate.set("this", exp.Distinct(expressions=[operand.copy()]))
    else:
        distinct = result_aggregate.this
        assert isinstance(distinct, exp.Distinct)
        expressions = list(distinct.expressions)
        if len(expressions) != 1:
            return None
        result_aggregate.set("this", expressions[0].copy())
    return result

def _compile_column(value: str, aliases: dict[str, str]) -> exp.Column | None:
    value = value.strip()
    if not value:
        return None
    if "." in value:
        table, column = value.rsplit(".", 1)
        alias = aliases.get(table.casefold())
        if alias is None:
            return None
        return exp.column(column, table=alias)
    return exp.column(value)

def _table_aliases(query: exp.Select) -> dict[str, str]:
    result: dict[str, str] = {}
    for table in query.find_all(exp.Table):
        # A table visible only inside a nested SELECT cannot be referenced by
        # a new root projection.  Treat it as unavailable rather than emitting
        # a candidate that SQLite will reject for an out-of-scope alias.
        if table.find_ancestor(exp.Select) is not query:
            continue
        if table.name:
            result[table.name.casefold()] = table.alias_or_name
            result[table.alias_or_name.casefold()] = table.alias_or_name
    return result

def _arity_variants(query: exp.Select, arity: int) -> list[exp.Select]:
    if arity < 1 or arity >= len(query.expressions):
        return []
    # A predicted column count is not evidence that a reported group's label
    # is redundant. Preserve row-identifying expressions when shortening a
    # grouped answer. Aggregate measures may still be trimmed, and ordinary
    # (ungrouped) helper-column correction is unchanged.
    required = {i for i, expression in enumerate(query.expressions)
                if query.args.get("group") is not None and _has_group_label(expression)}
    if len(required) > arity:
        return []
    optional = [i for i in range(len(query.expressions)) if i not in required]
    values: list[exp.Select] = []
    for selected in itertools.islice(itertools.combinations(optional, arity - len(required)), 8):
        indices = sorted(required.union(selected))
        variant = query.copy()
        variant.set(
            "expressions",
            [query.expressions[index].copy() for index in indices],
        )
        values.append(variant)
    return values


def _has_group_label(expression: exp.Expression) -> bool:
    """Find row-level columns without descending into aggregates/subqueries."""
    if isinstance(expression, (exp.AggFunc, exp.Window, exp.Query, exp.Filter, exp.WithinGroup)):
        return False
    if isinstance(expression, (exp.Column, exp.Star)):
        return True
    return any(_has_group_label(child) for child in expression.iter_expressions())
