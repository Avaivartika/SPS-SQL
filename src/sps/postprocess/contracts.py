"""Original post-correction contracts primitives."""
from __future__ import annotations
from typing import Any, Literal
import sqlglot
from pydantic import BaseModel, ConfigDict, Field
from sqlglot import exp
from sps.postprocess.bindings import SQLBindings

STRUCTURAL_CONTRACT_SYSTEM_PROMPT = """You infer the observable answer contract of a database question.
Use only the question, public evidence, and schema. Do not write SQL and do not
add helper columns used only for filtering, grouping, or ordering. Return one
JSON object matching the requested shape and no markdown."""

CONTRACT_SYSTEM_PROMPT = """You infer a soft semantic contract for a database question.
Use only the question, public evidence, and schema. Do not write SQL. Describe
the requested output plus the relations, filters, grouping, and ordering needed
to produce it. Return one JSON object matching the requested shape and no markdown."""

SlotOperation = Literal[
    "attribute",
    "count",
    "sum",
    "average",
    "minimum",
    "maximum",
    "ratio",
    "percentage",
    "difference",
    "rank",
    "conditional",
    "formula",
]

RowMode = Literal[
    "scalar",
    "rows",
    "distinct_rows",
    "grouped_rows",
    "top_k",
    "ranked_rows",
]

class AnswerSlot(BaseModel):
    """One ordered output column in the expected answer."""

    model_config = ConfigDict(extra="forbid")

    operation: SlotOperation
    source_columns: list[str] = Field(default_factory=list)
    aggregate_distinct: bool | None = None
    operators: list[str] = Field(default_factory=list)

class AnswerContract(BaseModel):
    """A compact contract that can be compared to candidate SQL behavior."""

    model_config = ConfigDict(extra="forbid")

    output_arity: int = Field(ge=1)
    ordered_slots: list[AnswerSlot]
    row_mode: RowMode
    select_distinct: bool
    order_required: bool
    limit: int | None = Field(default=None, ge=0)
    tables: list[str] = Field(default_factory=list)
    predicate_columns: list[str] = Field(default_factory=list)
    predicate_operators: list[str] = Field(default_factory=list)
    predicate_values: list[str] = Field(default_factory=list)
    group_by_columns: list[str] = Field(default_factory=list)
    having_required: bool = False
    order_by_columns: list[str] = Field(default_factory=list)
    order_directions: list[str] = Field(default_factory=list)
    set_operation: Literal["none", "union", "intersect", "except"] = "none"

def extract_answer_contract(
    sql: str,
    schema: dict[str, Any] | None = None,
) -> AnswerContract:
    """Derive a deterministic answer contract from one reference query."""

    statements = [
        item for item in sqlglot.parse(sql, read="sqlite") if item is not None
    ]
    if len(statements) != 1:
        raise ValueError(f"expected one statement, received {len(statements)}")
    root = statements[0]
    select = root if isinstance(root, exp.Select) else root.find(exp.Select)
    if select is None:
        raise ValueError("query has no SELECT")
    bindings = SQLBindings(root, schema)
    projections = [_unwrap_alias(item) for item in select.expressions]
    slots = [
        _extract_slot(item, source_columns=bindings.columns(item))
        for item in projections
    ]
    if not slots:
        raise ValueError("query has no output projections")

    has_window = any(item.find(exp.Window) is not None for item in projections)
    has_group = select.args.get("group") is not None
    all_aggregate = all(item.find(exp.AggFunc) is not None for item in projections)
    has_order = select.args.get("order") is not None
    limit = _literal_limit(select.args.get("limit"))
    is_distinct = select.args.get("distinct") is not None
    filters = list(root.find_all(exp.Where)) + list(root.find_all(exp.Having))
    filter_expression = (
        exp.and_(*[value.this for value in filters]) if filters else None
    )
    group = select.args.get("group")
    order = select.args.get("order")
    if has_window:
        row_mode: RowMode = "ranked_rows"
    elif has_group:
        row_mode = "grouped_rows"
    elif all_aggregate:
        row_mode = "scalar"
    elif has_order and select.args.get("limit") is not None:
        row_mode = "top_k"
    elif is_distinct:
        row_mode = "distinct_rows"
    else:
        row_mode = "rows"
    return AnswerContract(
        output_arity=len(slots),
        ordered_slots=slots,
        row_mode=row_mode,
        select_distinct=is_distinct,
        order_required=has_order,
        limit=limit,
        tables=bindings.tables(),
        # Resolve original filter nodes before combining them. exp.and_
        # copies expressions, which would detach columns from their scope.
        predicate_columns=_unique(column for value in filters
                                  for column in bindings.columns(value)),
        predicate_operators=(
            _predicate_operators(filter_expression)
            if filter_expression is not None
            else []
        ),
        predicate_values=(
            _predicate_values(filter_expression)
            if filter_expression is not None
            else []
        ),
        group_by_columns=(
            bindings.columns(group)
            if group is not None
            else []
        ),
        having_required=bool(list(root.find_all(exp.Having))),
        order_by_columns=(
            bindings.columns(order)
            if order is not None
            else []
        ),
        order_directions=(
            [
                "desc" if bool(value.args.get("desc")) else "asc"
                for value in order.expressions
            ]
            if order is not None
            else []
        ),
        set_operation=_set_operation(root),
    )

def render_schema(schema: dict[str, Any]) -> str:
    """Render every relation and column, including public semantic names."""

    tables = list(schema.get("table_names_original") or schema["table_names"])
    semantic_tables = list(schema.get("table_names") or tables)
    raw_columns = list(schema.get("column_names_original") or schema["column_names"])
    semantic_columns = list(schema.get("column_names") or raw_columns)
    types = list(schema.get("column_types") or [""] * len(raw_columns))
    primary_keys = set(_flatten_integer_indices(schema.get("primary_keys") or []))
    grouped: dict[int, list[str]] = {index: [] for index in range(len(tables))}
    for index, ((table_index, original), semantic_pair) in enumerate(
        zip(raw_columns, semantic_columns, strict=True)
    ):
        table_index = int(table_index)
        if table_index < 0 or str(original) == "*":
            continue
        semantic = str(semantic_pair[1])
        annotation = ""
        if semantic.casefold() != str(original).casefold():
            annotation = f" ({semantic})"
        type_name = str(types[index]) if index < len(types) else ""
        key = " PK" if index in primary_keys else ""
        grouped[table_index].append(f"{original}:{type_name}{key}{annotation}")
    lines: list[str] = []
    for index, table in enumerate(tables):
        semantic = str(semantic_tables[index])
        label = str(table)
        if semantic.casefold() != label.casefold():
            label += f" ({semantic})"
        lines.append(f"- {label}: " + ", ".join(grouped[index]))
    if schema.get("foreign_keys"):
        rendered_keys = []
        for source, target in schema["foreign_keys"]:
            source_table, source_name = raw_columns[int(source)]
            target_table, target_name = raw_columns[int(target)]
            rendered_keys.append(
                f"{tables[int(source_table)]}.{source_name}="
                f"{tables[int(target_table)]}.{target_name}"
            )
        lines.append("- foreign keys: " + "; ".join(rendered_keys))
    return "\n".join(lines)

def render_contract_prompt(
    *,
    question: str,
    evidence: str,
    schema: dict[str, Any],
) -> str:
    """Build the candidate-independent user prompt."""

    return f"""Question: {question.strip()}
Evidence: {evidence.strip() or "(none)"}
Schema:
{render_schema(schema)}

Return exactly this JSON shape:
{{"output_arity":<integer>,"ordered_slots":[{{"aggregate_distinct":<true,false,or null>,"operation":"attribute|count|sum|average|minimum|maximum|ratio|percentage|difference|rank|conditional|formula","operators":["normalized operator names"],"source_columns":["table.column"]}}],"row_mode":"scalar|rows|distinct_rows|grouped_rows|top_k|ranked_rows","select_distinct":<boolean>,"order_required":<boolean>,"limit":<integer or null>,"tables":["table"],"predicate_columns":["table.column"],"predicate_operators":["eq|gt|gte|lt|lte|between|like|in|is|not|and|or"],"predicate_values":["string:value|number:value|null"],"group_by_columns":["table.column"],"having_required":<boolean>,"order_by_columns":["table.column"],"order_directions":["asc|desc"],"set_operation":"none|union|intersect|except"}}"""

def _extract_slot(
    expression: exp.Expression,
    *,
    source_columns: list[str],
) -> AnswerSlot:
    aggregates = list(expression.find_all(exp.AggFunc))
    aggregate_distinct: bool | None = None
    if aggregates:
        aggregate_distinct = any(
            isinstance(item.this, exp.Distinct) for item in aggregates
        )
    operators = _operators(expression)
    operation = _operation(expression, aggregates=aggregates)
    return AnswerSlot(
        operation=operation,
        source_columns=source_columns,
        aggregate_distinct=aggregate_distinct,
        operators=operators,
    )

def _operation(
    expression: exp.Expression,
    *,
    aggregates: list[exp.AggFunc],
) -> SlotOperation:
    names = {type(node).__name__.casefold() for node in expression.walk()}
    if expression.find(exp.Window) is not None and names & {
        "rank",
        "denserank",
        "rownumber",
        "ntile",
    }:
        return "rank"
    if "div" in names:
        literals = {
            str(node.this).strip()
            for node in expression.find_all(exp.Literal)
            if not node.is_string
        }
        if any(_is_hundred(value) for value in literals) and "mul" in names:
            return "percentage"
        return "ratio"
    if "sub" in names:
        return "difference"
    if expression.find(exp.Case) is not None or expression.find(exp.If) is not None:
        return "conditional"
    aggregate_names = {type(item).__name__.casefold() for item in aggregates}
    for node_name, label in (
        ("count", "count"),
        ("sum", "sum"),
        ("avg", "average"),
        ("min", "minimum"),
        ("max", "maximum"),
    ):
        if node_name in aggregate_names:
            return label  # type: ignore[return-value]
    if isinstance(expression, (exp.Column, exp.Star)):
        return "attribute"
    if not _operators(expression) and list(expression.find_all(exp.Column)):
        return "attribute"
    return "formula"

def _operators(expression: exp.Expression) -> list[str]:
    interesting = (
        exp.AggFunc,
        exp.Add,
        exp.Sub,
        exp.Mul,
        exp.Div,
        exp.Mod,
        exp.Case,
        exp.If,
        exp.Window,
    )
    values: list[str] = []
    for node in expression.walk():
        if isinstance(node, interesting):
            name = type(node).__name__.casefold()
            if name not in values:
                values.append(name)
    return values

def _predicate_operators(expression: exp.Expression) -> list[str]:
    interesting = (
        exp.EQ,
        exp.GT,
        exp.GTE,
        exp.LT,
        exp.LTE,
        exp.Between,
        exp.Like,
        exp.In,
        exp.Is,
        exp.Not,
        exp.And,
        exp.Or,
    )
    return _unique(
        type(value).__name__.casefold()
        for value in expression.walk()
        if isinstance(value, interesting)
    )

def _predicate_values(expression: exp.Expression) -> list[str]:
    values: list[str] = []
    for literal in expression.find_all(exp.Literal):
        kind = "string" if literal.is_string else "number"
        rendered = f"{kind}:{literal.this}"
        if rendered not in values:
            values.append(rendered)
    if expression.find(exp.Null) is not None:
        values.append("null")
    return values

def _set_operation(
    root: exp.Expression,
) -> Literal["none", "union", "intersect", "except"]:
    if isinstance(root, exp.Union):
        return "union"
    if isinstance(root, exp.Intersect):
        return "intersect"
    if isinstance(root, exp.Except):
        return "except"
    return "none"

def _unique(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw)
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result

def _literal_limit(limit: exp.Expression | None) -> int | None:
    if limit is None:
        return None
    value = limit.args.get("expression")
    if isinstance(value, exp.Literal) and not value.is_string:
        try:
            return int(str(value.this))
        except ValueError:
            return None
    return None

def _is_hundred(value: str) -> bool:
    try:
        return float(value) == 100.0
    except ValueError:
        return False

def _flatten_integer_indices(values: list[Any]) -> list[int]:
    result: list[int] = []
    for value in values:
        if isinstance(value, list):
            result.extend(_flatten_integer_indices(value))
        else:
            result.append(int(value))
    return result

def _unwrap_alias(expression: exp.Expression) -> exp.Expression:
    return expression.this if isinstance(expression, exp.Alias) else expression

def render_structural_contract_prompt(
    *, question: str, evidence: str, schema: dict[str, Any],
) -> str:
    """Structural adapter's six-field training format; no semantic fields.

    Recovered from answer_contract_v0/data/train.jsonl, whose 8,285 stored
    prompts all use this exact output shape. Keep semantic inference separate.
    """
    return f"""Question: {question.strip()}
Evidence: {evidence.strip() or "(none)"}
Schema:
{render_schema(schema)}

Return exactly this JSON shape:
{{"output_arity":<integer>,"ordered_slots":[{{"aggregate_distinct":<true,false,or null>,"operation":"attribute|count|sum|average|minimum|maximum|ratio|percentage|difference|rank|conditional|formula","operators":["normalized operator names"],"source_columns":["table.column"]}}],"row_mode":"scalar|rows|distinct_rows|grouped_rows|top_k|ranked_rows","select_distinct":<boolean>,"order_required":<boolean>,"limit":<integer or null>}}"""
