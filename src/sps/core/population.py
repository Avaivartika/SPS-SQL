"""One question's bounded population, independent of model transport."""
from __future__ import annotations

import hashlib
import time

from sps.core.execution import outcome_key
from sps.core.route_execution import execute_route
from sps.core.routes import population_support_scores


async def generate_population(question, policy, request, execute, route_rng, action_rng, *,
    population_size=8, max_attempt_multiplier=3, trajectory_id=0,
    samples=5, invalid_retry_limit=15, max_batches=32):
    if population_size < 1 or max_attempt_multiplier < 1:
        raise ValueError("Population and attempt multiplier must be positive")
    attempted, successful, keys = [], [], []
    started = time.perf_counter()
    for _ in range(population_size * max_attempt_multiplier):
        route = policy.sample(route_rng)
        trajectory_id += 1
        calls = []

        async def tracked_request(template, prompt, count):
            call = {"template": template, "requested_samples": count,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "input_tokens": None, "output_tokens": None, "error": None}
            calls.append(call)
            start = time.perf_counter()
            try:
                batch = await request(template, prompt, count)
                call.update(input_tokens=batch.input_tokens, output_tokens=batch.output_tokens,
                            request_attempts=batch.request_attempts,
                            retry_errors=list(batch.retry_errors), seed=batch.seed)
                return batch
            except Exception as exc:
                call["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                call["wall_seconds"] = time.perf_counter() - start

        result = await execute_route(question, route, tracked_request, execute, route_rng, action_rng,
            samples=samples, invalid_retry_limit=invalid_retry_limit, max_batches=max_batches)
        known = all(isinstance(c[k], int) and not isinstance(c[k], bool) and c[k] >= 0
                    for c in calls for k in ("input_tokens", "output_tokens"))
        result.update(trajectory_id=trajectory_id, calls=calls,
            input_tokens=sum(c["input_tokens"] for c in calls) if known else None,
            output_tokens=sum(c["output_tokens"] for c in calls) if known else None,
            total_tokens=sum(c["input_tokens"] + c["output_tokens"] for c in calls) if known else None)
        attempted.append(result)
        if result["status"] == "ok":
            outcome = await execute(result["final_sql"])
            key = outcome_key(outcome.rows) if outcome.valid else None
            result.update(outcome_key=key, execution=outcome.as_record())
            successful.append(result)
            keys.append(key)
        # Cost is part of elite ranking. Missing usage cannot be silently
        # treated as zero or repaired by generating another route.
        if not known or len(successful) == population_size:
            break
    status = "ok"
    if any(r["total_tokens"] is None for r in attempted):
        status = "usage_unavailable"
    elif len(successful) != population_size:
        status = "incomplete"
    for result, support in zip(successful, population_support_scores(keys), strict=True):
        result["search_score"] = support
    scored = [{k: r[k] for k in ("trajectory_id", "route", "reward", "search_score", "total_tokens")}
              for r in successful] if status == "ok" else []
    known = all(r["total_tokens"] is not None for r in attempted)
    return {"status": status, "question_id": question.question_id, "db_id": question.db_id,
            "target_population_size": population_size, "attempted": attempted,
            "successful_ids": [r["trajectory_id"] for r in successful], "scored_population": scored,
            "policy_used": policy.snapshot(), "route_rng_after": route_rng.getstate(),
            "action_rng_after": action_rng.getstate(), "last_trajectory_id": trajectory_id,
            "input_tokens": sum(r["input_tokens"] for r in attempted) if known else None,
            "output_tokens": sum(r["output_tokens"] for r in attempted) if known else None,
            "wall_seconds": time.perf_counter() - started}
