"""A native DataDesigner population column, with per-question state."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import random

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModel
from data_designer.engine.models.utils import prompt_to_messages

from sps.core.execution import execute
from sps.core.population import generate_population
from sps.core.route_execution import PublicQuestion
from sps.core.sql_plan import database_path
from sps.core.sql_sampling import ModelBatch
from sps.core.update import restore_policy
from sps.integration.config import PopulationConfig


MODEL_REQUEST_ATTEMPTS = 5
MODEL_SEED_MODULUS = 2_147_483_647


def _nonretryable_client_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    text = str(error).casefold()
    unavailable_free_tier = status == 429 and ("limit: 0" in text or "free_tier" in text)
    return (isinstance(status, int) and 400 <= status < 500
            and (status != 429 or unavailable_free_tier))


def deterministic_request_seed(*, random_seed: int, question_id: int,
                               population_name: str, request_index: int,
                               template: str, prompt: str) -> int:
    """Derive a scheduling-independent seed for one logical model request."""
    payload = "\0".join((str(random_seed), str(question_id), population_name,
                         str(request_index), template,
                         hashlib.sha256(prompt.encode()).hexdigest()))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % MODEL_SEED_MODULUS


async def request_model_batch(model, messages, *, count: int, max_tokens: int | None = None,
                              seed: int | None = None) -> ModelBatch:
    """Make one logical Alpha-style request with bounded transport retries.

    DataDesigner's request executor does not retry provider timeouts.  The
    released Alpha client retried transient request failures five times, so the
    native bridge must preserve that boundary instead of turning one timeout
    into a failed question.  A response without usage is a malformed provider
    response because token cost participates in CEM tie-breaking.
    """
    errors = []
    for attempt in range(1, MODEL_REQUEST_ATTEMPTS + 1):
        try:
            overrides = {"max_tokens": max_tokens} if max_tokens else {}
            if seed is not None:
                overrides["seed"] = seed
            response = await model.acompletion(
                messages, n=count, temperature=.8, allow_multiple_choices=True,
                skip_usage_tracking=False, **overrides)
            choices = tuple(response.choices)
            if len(choices) != count:
                raise ValueError(f"Model returned {len(choices)} of {count} requested samples")
            usage = response.usage
            input_tokens = usage.input_tokens if usage else None
            output_tokens = usage.output_tokens if usage else None
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in (input_tokens, output_tokens)):
                raise ValueError("Model response omitted valid token usage")
            return ModelBatch(
                tuple(choice.message.content for choice in choices),
                input_tokens,
                output_tokens,
                tuple(choice.finish_reason for choice in choices),
                request_attempts=attempt,
                retry_errors=tuple(errors),
                seed=seed,
            )
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}"[:1000])
            if _nonretryable_client_error(error) or attempt == MODEL_REQUEST_ATTEMPTS:
                raise RuntimeError(
                    f"Model request failed after {attempt} attempt(s): {errors[-1]}"
                ) from error
            await asyncio.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError("unreachable model request retry state")


def restore_rng(state, seed: int) -> random.Random:
    rng = random.Random(seed)
    if state is not None:
        if not isinstance(state, str):
            raise TypeError("RNG state must be a JSON string")

        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value

        rng.setstate(tuples(json.loads(state)))
    return rng


class PopulationGenerator(ColumnGeneratorWithModel[PopulationConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    async def agenerate(self, data: dict) -> dict:
        context = data[self.config.question_column]
        if not isinstance(context, str):
            raise TypeError("Public question must be a JSON string")
        question = PublicQuestion.model_validate_json(context)
        snapshot = data[self.config.policy_column]
        if not isinstance(snapshot, str):
            raise TypeError("Policy must be a JSON string")
        snapshot = json.loads(snapshot)
        if "probabilities" in snapshot:
            if set(snapshot) != {"probabilities", "elite_ids", "population_size"}:
                raise ValueError("Unexpected fields in policy update")
            snapshot = snapshot["probabilities"]
        policy = restore_policy(snapshot)
        previous = None
        if self.config.previous_population_column:
            previous = json.loads(data[self.config.previous_population_column])
            if (previous["question_id"], previous["db_id"]) != (question.question_id, question.db_id):
                raise ValueError("Previous population belongs to a different question")
        status = "blocked" if previous and previous["status"] != "ok" else None
        try:
            database = database_path(Path(self.config.database_root), question.db_id)
        except FileNotFoundError:
            status = "unavailable"
        if status is None:
            seed = self.config.random_seed + 1_000_003 * question.question_id
            route_rng = restore_rng(json.dumps(previous["route_rng_after"]) if previous else None, seed)
            action_rng = restore_rng(json.dumps(previous["action_rng_after"]) if previous else None, seed)
        if status:
            record = {"status": status, "question_id": question.question_id, "db_id": question.db_id,
                      "error": "previous_population_failed" if status == "blocked" else "database_not_found",
                      "attempted": [], "successful_ids": [], "scored_population": [],
                      "policy_used": policy.snapshot()}
        else:
            cache = {}
            request_index = 0

            async def request(template, prompt, count):
                nonlocal request_index
                seed = deterministic_request_seed(
                    random_seed=self.config.random_seed,
                    question_id=question.question_id,
                    population_name=self.config.name,
                    request_index=request_index,
                    template=template,
                    prompt=prompt,
                )
                request_index += 1
                max_tokens = (self.config.sql_max_tokens
                              if template in {"sql_generation", "sql_revision"} else None)
                return await request_model_batch(
                    self.model, prompt_to_messages(user_prompt=prompt),
                    count=count, max_tokens=max_tokens, seed=seed)

            async def execute_one(sql):
                if sql not in cache:
                    cache[sql] = await asyncio.to_thread(execute, database, sql,
                        timeout_seconds=self.config.timeout_seconds, max_rows=self.config.max_rows)
                return cache[sql]

            record = await generate_population(question, policy, request, execute_one, route_rng, action_rng,
                population_size=self.config.population_size, max_attempt_multiplier=self.config.max_attempt_multiplier,
                trajectory_id=previous["last_trajectory_id"] if previous else 0,
                samples=self.config.samples, invalid_retry_limit=self.config.invalid_retry_limit,
                max_batches=self.config.max_batches)
        dump = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False)
        return {**data, self.config.name: dump(record),
                self.config.policy_output: dump(record["policy_used"]),
                self.config.scored_output: dump(record["scored_population"])}
