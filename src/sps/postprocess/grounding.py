"""Original post-correction grounding primitives."""
from __future__ import annotations
import json
from typing import Any
from pydantic import BaseModel, ConfigDict
from sps.postprocess.contracts import AnswerContract

GROUNDING_SYSTEM_PROMPT = """You ground each ordered answer slot to database columns.
Use only the question, public evidence, source-free answer structure, and the
indexed schema. Return schema IDs exactly as listed. Preserve operand order for
ratios, percentages, differences, and formulas. Do not write SQL, column names,
or explanations. Return one JSON object and no markdown."""

class SchemaColumn(BaseModel):
    """One selectable physical schema column."""

    model_config = ConfigDict(extra="forbid")

    column_id: str
    table: str
    column: str
    semantic_table: str
    semantic_column: str
    data_type: str = ""
    primary_key: bool = False

    @property
    def canonical_name(self) -> str:
        return f"{self.table}.{self.column}"

class SchemaGroundingTarget(BaseModel):
    """Ordered physical columns used by every answer slot."""

    model_config = ConfigDict(extra="forbid")

    slot_source_ids: list[list[str]]

def schema_column_catalog(schema: dict[str, Any]) -> list[SchemaColumn]:
    """Create stable IDs from native BIRD schema-column indices."""

    tables = list(schema.get("table_names_original") or schema["table_names"])
    semantic_tables = list(schema.get("table_names") or tables)
    raw_columns = list(schema.get("column_names_original") or schema["column_names"])
    semantic_columns = list(schema.get("column_names") or raw_columns)
    types = list(schema.get("column_types") or [""] * len(raw_columns))
    primary_keys = set(_flatten_integer_indices(schema.get("primary_keys") or []))
    result: list[SchemaColumn] = []
    for schema_index, ((table_index, original), semantic_pair) in enumerate(
        zip(raw_columns, semantic_columns, strict=True)
    ):
        table_index = int(table_index)
        if table_index < 0 or str(original) == "*":
            continue
        result.append(
            SchemaColumn(
                column_id=f"C{schema_index}",
                table=str(tables[table_index]),
                column=str(original),
                semantic_table=str(semantic_tables[table_index]),
                semantic_column=str(semantic_pair[1]),
                data_type=str(types[schema_index]) if schema_index < len(types) else "",
                primary_key=schema_index in primary_keys,
            )
        )
    if not result:
        raise ValueError("schema contains no selectable columns")
    return result

def source_free_contract(contract: AnswerContract) -> dict[str, Any]:
    """Return structural slot information without any source-column labels."""

    return {
        "output_arity": contract.output_arity,
        "ordered_slots": [
            {
                "slot": index,
                "operation": slot.operation,
                "aggregate_distinct": slot.aggregate_distinct,
                "operators": slot.operators,
            }
            for index, slot in enumerate(contract.ordered_slots)
        ],
        "row_mode": contract.row_mode,
        "select_distinct": contract.select_distinct,
        "order_required": contract.order_required,
        "limit": contract.limit,
    }

def render_indexed_schema(schema: dict[str, Any]) -> str:
    """Render selectable columns and relation constraints compactly."""

    catalog = schema_column_catalog(schema)
    lines = []
    for item in catalog:
        aliases = []
        if item.semantic_table.casefold() != item.table.casefold():
            aliases.append(f"table meaning={item.semantic_table}")
        if item.semantic_column.casefold() != item.column.casefold():
            aliases.append(f"column meaning={item.semantic_column}")
        metadata = [item.data_type]
        if item.primary_key:
            metadata.append("primary key")
        metadata.extend(aliases)
        suffix = "; ".join(value for value in metadata if value)
        lines.append(
            f"{item.column_id} = {item.table}.{item.column}"
            + (f" [{suffix}]" if suffix else "")
        )

    raw_columns = list(schema.get("column_names_original") or schema["column_names"])
    by_index = {int(item.column_id[1:]): item for item in catalog}
    foreign_keys = []
    for source, target in schema.get("foreign_keys") or []:
        source_index, target_index = int(source), int(target)
        if source_index in by_index and target_index in by_index:
            foreign_keys.append(
                f"{by_index[source_index].column_id}={by_index[target_index].column_id}"
            )
        elif source_index < len(raw_columns) and target_index < len(raw_columns):
            continue
    if foreign_keys:
        lines.append("Foreign keys: " + ", ".join(foreign_keys))
    return "\n".join(lines)

def render_grounding_prompt(
    *,
    question: str,
    evidence: str,
    schema: dict[str, Any],
    contract: AnswerContract,
) -> str:
    structure = json.dumps(
        source_free_contract(contract),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"""Question: {question.strip()}
Evidence: {evidence.strip() or "(none)"}
Answer structure (slot order is fixed):
{structure}
Indexed schema:
{render_indexed_schema(schema)}

For every slot, select the physical column IDs needed to compute it. Preserve
operand order. COUNT(*) has an empty list. Return exactly:
{{"slot_source_ids":[["C<number>"]]}}"""

def _flatten_integer_indices(values: list[Any]) -> list[int]:
    result: list[int] = []
    for value in values:
        if isinstance(value, list):
            result.extend(_flatten_integer_indices(value))
        else:
            result.append(int(value))
    return result
