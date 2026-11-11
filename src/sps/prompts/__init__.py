"""Packaged, checksummed prompt assets; no legacy workspace lookup."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib.resources import files
from typing import Mapping

FINAL_GUARD = """

Final task-consistency check:
- The original user question below is authoritative.  Earlier rephrasing and
  analysis are fallible working notes.
- Return exactly the requested expressions and preserve their requested order.
- Use the evidence as an exact mapping when it names a column, value, formula,
  or operator, and copy schema identifiers exactly.
- Preserve the requested answer grain.  Do not add or remove DISTINCT,
  grouping, aggregation, ordering, ties, or LIMIT unless the question,
  evidence, or join multiplicity requires it.

Original user question:
{ORIGINAL_QUESTION}

Original task evidence:
{ORIGINAL_EVIDENCE}
"""


@lru_cache(maxsize=1)
def templates() -> Mapping[str, str]:
    root = files(__package__)
    manifest = json.loads(root.joinpath("manifest.json").read_text(encoding="utf-8"))
    result = {}
    for name, record in manifest["templates"].items():
        text = root.joinpath("templates", name + ".txt").read_text(encoding="utf-8")
        text = text.rstrip("\n") + "\n" * record["trailing_newlines"]
        if hashlib.sha256(text.encode()).hexdigest() != record["sha256"]:
            raise ValueError(f"Prompt asset checksum mismatch: {name}")
        result[name] = text
    from types import MappingProxyType
    return MappingProxyType(result)


def render(name: str, arguments: Mapping[str, str], *, question: str, evidence: str) -> str:
    template = templates()[name]
    if name not in {"sql_generation", "sql_revision"}:
        return template.format(**arguments)
    before, marker, after = template.rpartition("Only output xml format")
    if not marker:
        raise ValueError(f"Missing final output marker in {name}")
    guard = FINAL_GUARD.format(ORIGINAL_QUESTION=question,
                               ORIGINAL_EVIDENCE=evidence or "None provided.")
    # Format template fields once. Braces in user text are literal, never a
    # second formatting pass (the old renderer could fail on such questions).
    return before.format(**arguments) + guard + "\n" + marker + after.format(**arguments)


def prompt_digest() -> str:
    payload = {"base": dict(templates()), "final_guard": FINAL_GUARD,
               "scope": "sql_generation_and_sql_revision_only"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
