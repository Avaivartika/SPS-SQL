"""Build public database context directly, without a legacy task or executor.

The text format preserves the frozen full-schema prompt, including its empty
PRIMARY KEY clause. This is model context, not a database creation script.
Formatting semantics originate in Alpha-SQL (license packaged with prompts).
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import sqlite3
import time

import chardet
import pandas as pd
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from sps.core.route_execution import PublicQuestion, schema_blocks
from sps.core.sql_plan import database_path


class UnpreparedQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question_id: StrictInt
    db_id: StrictStr
    question: StrictStr
    evidence: StrictStr = ""


def quote(name):
    return "`" + name.replace("`", "``") + "`"


def normalize(value):
    value = value.replace("\r", "").replace("\n", "").replace("commonsense evidence:", "").strip()
    while "  " in value:
        value = value.replace("  ", " ")
    return value


def descriptions(directory: Path):
    result, sources = {}, []
    if not directory.exists():
        return result, sources
    for path in sorted(directory.glob("*.csv")):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(directory.parent.resolve()):
            raise ValueError("Description file escapes database directory")
        payload = path.read_bytes()
        encoding = chardet.detect(payload)["encoding"]
        if encoding is None:
            raise ValueError(f"Unknown description encoding: {path.name}")
        frame = pd.read_csv(io.BytesIO(payload), encoding=encoding, index_col=False)
        required = {"original_column_name", "column_name", "column_description", "value_description", "data_format"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Incomplete description fields: {path.name}")
        table = path.stem.lower().strip()
        if table in result:
            raise ValueError("Ambiguous table description")
        columns = {}
        for _, row in frame.iterrows():
            if pd.isna(row["original_column_name"]):
                continue
            name = row["original_column_name"].strip().lower()
            if name in columns:
                raise ValueError("Duplicate column description")
            fields = {}
            for source, target in (("column_name", "expanded"), ("column_description", "description"),
                                   ("value_description", "values")):
                value = "" if pd.isna(row[source]) else row[source]
                if not isinstance(value, str):
                    raise ValueError("Description fields must contain text")
                fields[target] = value.strip() if source == "column_name" else normalize(value)
            if fields["values"].lower().startswith("not useful"):
                fields["values"] = fields["values"][len("not useful"):].strip()
            columns[name] = fields
        result[table] = columns
        sources.append({"file": path.name, "sha256": hashlib.sha256(payload).hexdigest(), "encoding": encoding})
    return result, sources


def build_context(root: Path, db_id: str, *, timeout_seconds=60.0):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Positive finite schema-query timeout required")
    database = database_path(root, db_id)
    desc, sources = descriptions(database.parent / "database_description")
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=timeout_seconds)
    try:
        connection.execute("PRAGMA query_only=ON")

        def query(sql):
            deadline = time.monotonic() + timeout_seconds
            connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            try:
                return connection.execute(sql).fetchall()
            finally:
                connection.set_progress_handler(None, 0)

        tables = [r[0] for r in query("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")]
        # The route parser currently assumes simple backtick-delimited names.
        # Reject unsupported identifiers instead of silently dropping tables.
        if not tables or any(t != t.strip() or '`' in t or '\n' in t or '\r' in t for t in tables):
            raise ValueError("Unsupported or empty table names for the frozen schema format")
        info = {t: query(f"PRAGMA table_info({quote(t)});") for t in tables}
        blocks = []
        for table in tables:
            foreign = query(f"PRAGMA foreign_key_list({quote(table)});")
            lines, primary, keys = [], [], []
            for column in info[table]:
                name, kind = column[1], column[2].strip()
                if name != name.strip():
                    raise ValueError("Whitespace-padded column names are unsupported")
                fields = desc.get(table.lower(), {}).get(name.lower(), {})
                expanded = fields.get("expanded", "")
                description = fields.get("description", "")
                values = fields.get("values", "")
                comments = []
                if expanded and expanded.lower() != name.lower():
                    comments.append(f"Column Meaning: {expanded}")
                if description and description.lower() not in {name.lower(), expanded.lower()}:
                    comments.append(f"Column Description: {description}")
                if values:
                    comments.append(f"Value Description: {values}")
                # Only TEXT examples appear in the frozen rendered prompt.
                if kind.upper() == "TEXT":
                    q = quote(name)
                    examples = query(f"SELECT DISTINCT {q} FROM {quote(table)} WHERE {q} IS NOT NULL "
                        f"AND {q} != '' AND length(cast({q} as text)) <= 100 LIMIT 3;")
                    if examples:
                        comments.append("Value Examples: " + ", ".join(f"`{r[0]}`" for r in examples))
                line = f"\t{quote(name)} {kind},"
                if comments:
                    line += " -- " + " | ".join(comments)
                lines.append(line)
                if column[5]:
                    primary.append(name)
                for key in foreign:
                    if key[3].strip().lower() != name.lower():
                        continue
                    target_table, target_column = key[2].strip(), key[4]
                    if target_column is None:
                        target_info = next((v for t, v in info.items() if t.lower() == target_table.lower()), [])
                        target_keys = [c[1].strip() for c in target_info if c[5]]
                        if not target_keys:
                            raise ValueError("Implicit foreign key has no resolvable primary key")
                        target_column = target_keys[0]
                    keys.append(f"\tFOREIGN KEY ({quote(name)}) REFERENCES {quote(target_table)}({quote(target_column.strip())}),")
            lines.append("\tPRIMARY KEY (" + ", ".join(map(quote, primary)) + "),")
            lines.extend(keys)
            lines[-1] = lines[-1][:-1]
            blocks.append(f"CREATE TABLE {quote(table)} (\n" + "\n".join(lines) + "\n);")
        context = "\n".join(blocks)
        schema_blocks(context)
        return context, {"db_id": db_id, "tables": len(tables), "columns": sum(map(len, info.values())),
            "description_sources": sources, "schema_sha256": hashlib.sha256(context.encode()).hexdigest(),
            "read_only": True, "format": "frozen_full_schema_v6", "reference_labels_loaded": False}
    finally:
        connection.close()


def prepare_question(raw: UnpreparedQuestion, context: str):
    if not raw.question.strip():
        raise ValueError("Question must be nonempty")
    return PublicQuestion(**raw.model_dump(), schema_context=context)
