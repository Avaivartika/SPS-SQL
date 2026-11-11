"""Per-route reasoning context, with no tree nodes or module-level state."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
import math
import random
import re
from typing import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from sps.core.execution import Execution, format_execution
from sps.core.routes import Route
from sps.core.sql_sampling import ModelBatch, sample_sql_group
from sps.prompts import render

TEMPLATES = {"rephrase": "raphrase_question", "schema": "schema_selection",
             "values": "identify_column_values", "functions": "identify_column_functions",
             "generate": "sql_generation", "revise": "sql_revision"}


class PublicQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question_id: StrictInt
    db_id: StrictStr
    question: StrictStr
    evidence: StrictStr
    schema_context: StrictStr


def _checkpoint_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _rng_tuples(value):
    return tuple(_rng_tuples(v) for v in value) if isinstance(value, (list, tuple)) else value


def schema_blocks(context: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?m)^CREATE TABLE\s+`([^`]+)`\s*\(", context))
    if not matches:
        raise ValueError("Expected prepared backtick-quoted table DDL blocks")
    blocks = {}
    for i, match in enumerate(matches):
        name = match.group(1)
        if name.lower() in {n.lower() for n in blocks}:
            raise ValueError("Ambiguous case-insensitive schema identifiers")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        blocks[name] = context[match.start():end].strip()
    return blocks


def select_schema(response: str | None, blocks: dict[str, str], rng: random.Random) -> str | None:
    try:
        match = re.search(r"```json\n(.*)```", response, flags=re.DOTALL)
        selection = json.loads(match.group(1))
    except (TypeError, AttributeError, json.JSONDecodeError):
        return None
    if not isinstance(selection, dict) or not all(isinstance(t, str) and isinstance(cols, list)
            and all(isinstance(col, str) for col in cols) for t, cols in selection.items()):
        return None
    # Prepared contexts preserve complete DDL blocks for selected tables in the
    # frozen executor, even when the reply lists only a subset of columns.
    chosen = {}
    for name in selection:
        for original, block in blocks.items():
            if original.lower() == name.lower():
                chosen[original] = block
    if not chosen:
        return None
    values = list(chosen.values())
    rng.shuffle(values)
    return "\n".join(values)


async def execute_route(question: PublicQuestion, route: Route,
    request: Callable[[str, str, int], Awaitable[ModelBatch]],
    execute: Callable[[str], Awaitable[Execution]],
    route_rng: random.Random, action_rng: random.Random,
    *, samples: int = 5, invalid_retry_limit: int = 15, max_batches: int = 32,
    pause_after: int | None = None, resume_from: dict | None = None) -> dict:
    """Execute the released single-sample auxiliary-action configuration.

    Model routing belongs to the caller's DataDesigner facade. This function
    carries only one question's working text, SQL, action history and RNGs.
    Optional action-boundary checkpoints support controlled continuation studies;
    the normal pipeline passes neither option and follows the unchanged route.
    """
    working_question = question.question
    selected_context = question.schema_context
    notes = []
    events = []
    sql = None
    score, valid = 0.0, False
    start_index = 0
    question_digest = _checkpoint_digest(question.model_dump())
    if resume_from is not None:
        state = deepcopy(resume_from["state"])
        if resume_from.get("sha256") != _checkpoint_digest(state):
            raise ValueError("Continuation checkpoint checksum mismatch")
        version = state.get("version")
        if version not in (1, 2) or state.get("question_sha256") != question_digest:
            raise ValueError("Continuation checkpoint belongs to a different question")
        prefix = tuple(state["prefix"])
        last_boundary = route.steps.index("generate") + (version == 2)
        if (prefix != route.steps[:len(prefix)] or len(prefix) > last_boundary
                or [event["action"] for event in state["events"]] != list(prefix)):
            raise ValueError("Continuation route does not extend the saved prefix")
        if version == 2:
            if (not prefix or prefix[-1] != "generate"
                    or not isinstance(state.get("sql"), str) or not state["sql"].strip()
                    or type(state.get("valid")) is not bool
                    or isinstance(state.get("score"), bool)
                    or not isinstance(state.get("score"), (float, int))
                    or not math.isfinite(state["score"])):
                raise ValueError("Post-generation checkpoint requires SQL and its selection state")
            sql, score, valid = state["sql"], state["score"], state["valid"]
        start_index = len(prefix)
        working_question, selected_context = state["working_question"], state["selected_context"]
        notes, events = state["notes"], state["events"]
        route_rng.setstate(_rng_tuples(state["route_rng"]))
        action_rng.setstate(_rng_tuples(state["action_rng"]))
    if pause_after is not None and (isinstance(pause_after, bool)
            or not isinstance(pause_after, int)
            or not start_index <= pause_after <= route.steps.index("generate") + 1):
        raise ValueError("Pause must be at an operation boundary before accept/revise")
    try:
        blocks = schema_blocks(question.schema_context) if "schema" in route.steps else {}
        for index, action in enumerate(route.steps):
            if index < start_index:
                continue
            if index == pause_after:
                state = {"version": 1, "question_sha256": question_digest,
                    "prefix": list(route.steps[:index]), "working_question": working_question,
                    "selected_context": selected_context, "notes": deepcopy(notes),
                    "events": deepcopy(events), "route_rng": route_rng.getstate(),
                    "action_rng": action_rng.getstate()}
                if sql is not None:
                    state.update(version=2, sql=sql, score=score, valid=valid)
                return {"status": "paused", "route": list(route.steps), "final_sql": None,
                    "reward": None, "selected_valid": False, "events": events, "error": None,
                    "checkpoint": {"state": state, "sha256": _checkpoint_digest(state)}}
            if action == "accept":
                break
            thoughts = ""
            for previous, content in notes:
                if previous == "values" and action in {"schema", "functions", "generate", "revise"}:
                    thoughts += f"Identify column values: {content}\n"
                elif previous == "functions" and action in {"schema", "values", "generate", "revise"}:
                    thoughts += f"Identify column functions: {content}\n"
            if action == "revise":
                outcome = await execute(sql)
                thoughts += f"SQL generation: {sql}\nSQL execution result:\n{format_execution(outcome)}\n"
            hint = question.evidence + (f"\n\nHere are my previous thoughts:\n{thoughts}" if thoughts else "")
            arguments = {"QUESTION": question.question if action == "rephrase" else working_question,
                         "HINT": hint, "SCHEMA_CONTEXT": question.schema_context if action == "schema" else selected_context}
            template = TEMPLATES[action]
            prompt = render(template, arguments, question=question.question, evidence=question.evidence)
            if action in {"generate", "revise"}:
                async def sample(count):
                    return await request(template, prompt, count)
                group = await sample_sql_group(sample, execute, action_rng, samples=samples,
                    invalid_retry_limit=invalid_retry_limit, max_batches=max_batches)
                events.append({"action": action, "prompt": prompt, "sql_group": asdict(group)})
                if group.status != "ok" or not group.selected_sql:
                    raise RuntimeError(group.error or "SQL action produced no nonempty selection")
                sql, score, valid = group.selected_sql, group.consistency_score, group.selected_valid
            else:
                event = {"action": action, "prompt": prompt, "batches": []}
                events.append(event)
                success = False
                for _ in range(4 if action == "schema" else 1):
                    batch = await request(template, prompt, 1)
                    event["batches"].append(asdict(batch))
                    if len(batch.responses) != 1:
                        raise ValueError("Expected one auxiliary action sample")
                    response = batch.responses[0]
                    if action == "schema":
                        context = select_schema(response, blocks, action_rng)
                        if context is None:
                            continue
                        selected_context = context
                    elif action == "rephrase":
                        working_question = response or question.question
                    else:
                        notes.append((action, response))
                    # Preserve the outer route sampler's draw even with one
                    # auxiliary child. Omitting it changes later sampled routes.
                    route_rng.choice((0,))
                    success = True
                    break
                if not success:
                    raise RuntimeError("Schema selection produced no valid child after four batches")
        return {"status": "ok", "route": list(route.steps), "final_sql": sql,
                "reward": score, "selected_valid": valid, "events": events, "error": None}
    except Exception as exc:
        return {"status": "error", "route": list(route.steps), "final_sql": None,
                "reward": None, "selected_valid": False, "events": events,
                "error": f"{type(exc).__name__}: {exc}"}
