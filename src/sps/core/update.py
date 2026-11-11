"""Serializable CEM update, separate from route execution and SQL selection."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from sps.core.routes import GENERATE, Route, RoutePolicy


def restore_policy(snapshot: Mapping[str, Mapping[str, float]]) -> RoutePolicy:
    policy = RoutePolicy()
    for name, distribution in snapshot.items():
        state = () if name == "root" else (GENERATE,) if name == "post_generate" else tuple(name.split("|"))
        choices = policy.choices_for_state(state)
        if set(distribution) != set(choices):
            raise ValueError(f"Invalid policy choices for {name}")
        if any(not math.isfinite(p) or p < 0 for p in distribution.values()):
            raise ValueError("Policy probabilities must be finite and nonnegative")
        if not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-12):
            raise ValueError("Policy probabilities must sum to one")
        # JSON snapshots sort keys. Sampling must retain the original action order.
        policy._probabilities[state] = {action: distribution[action] for action in choices}
    return policy


def rank_population(population: Sequence[Mapping[str, Any]]) -> list[tuple]:
    """Validate scored trajectories and retain the frozen deterministic order."""
    if not population:
        raise ValueError("A nonempty successful population is required")
    prepared = []
    seen = set()
    allowed = {"trajectory_id", "route", "reward", "search_score", "total_tokens"}
    for row in population:
        if set(row) - allowed:
            raise ValueError("Unexpected fields in CEM input")
        identifier = row["trajectory_id"]
        if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier in seen:
            raise ValueError("Trajectory IDs must be unique integers")
        seen.add(identifier)
        route = Route(tuple(row["route"]))
        reward = float(row["reward"])
        score = reward if row.get("search_score") is None else float(row["search_score"])
        tokens = row["total_tokens"]
        if not math.isfinite(reward) or not math.isfinite(score):
            raise ValueError("Scores must be finite")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("Token cost must be a nonnegative integer")
        key = (-score, -reward, tokens, len(route.steps), route.as_text(), identifier)
        prepared.append((key, identifier, route))
    return sorted(prepared, key=lambda item: item[0])


def update_policy(
    snapshot: Mapping[str, Mapping[str, float]],
    population: Sequence[Mapping[str, Any]],
    *,
    elite_fraction: float = 0.25,
    old_weight: float = 0.25,
    exploration_mix: float = 0.1,
) -> dict[str, Any]:
    """Consume successful scored paths with the frozen SPS ranking order.

    Reward assignment and failed-attempt handling remain the executor's job.
    No reference SQL, labels, contracts, or model calls enter this component.
    """
    if not 0 < elite_fraction <= 1:
        raise ValueError("A valid elite fraction is required")
    ranked = rank_population(population)
    count = max(1, math.ceil(len(ranked) * elite_fraction))
    elites = ranked[:count]
    policy = restore_policy(snapshot)
    policy.update_from_elites((item[2] for item in elites), old_weight=old_weight,
                             exploration_mix=exploration_mix)
    return {"probabilities": policy.snapshot(), "elite_ids": [item[1] for item in elites],
            "population_size": len(population)}
