"""Cross-entropy optimization over Alpha-SQL's legal action routes.

The expensive Alpha-SQL actions remain unchanged.  This module only models
which action is executed next.  A route always generates SQL exactly once and
then either accepts it or performs one revision before terminating.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

REPHRASE = "rephrase"
SCHEMA = "schema"
VALUES = "values"
FUNCTIONS = "functions"
GENERATE = "generate"
ACCEPT = "accept"
REVISE = "revise"

AUXILIARY_ACTIONS = (REPHRASE, SCHEMA, VALUES, FUNCTIONS)
PERMUTABLE_ACTIONS = (SCHEMA, VALUES, FUNCTIONS)
POST_GENERATION_ACTIONS = (ACCEPT, REVISE)

State = tuple[str, ...]


def _ordered_float_sum(values: Iterable[float]) -> float:
    """Add floats left-to-right, independent of Python's ``sum`` version.

    Python 3.12 changed the implementation of ``sum`` for floats.  CEM
    probabilities are serialized and later drive seeded categorical draws, so
    even a one-ulp cross-version difference must not enter the saved policy.
    This loop preserves the original Python 3.11 update arithmetic on every
    supported interpreter.
    """
    total = 0.0
    for value in values:
        total += value
    return total


@dataclass(frozen=True, order=True)
class Route:
    """One legal high-level Alpha-SQL trajectory.

    ``steps`` contains zero or more auxiliary actions, mandatory ``generate``,
    and one terminal decision: ``accept`` or ``revise``.  The final EndAction
    is deterministic and therefore omitted from the optimized route.
    """

    steps: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_route(self)

    @property
    def auxiliary_steps(self) -> tuple[str, ...]:
        return self.steps[: self.steps.index(GENERATE)]

    @property
    def post_generation(self) -> str:
        return self.steps[-1]

    def decisions(self) -> tuple[tuple[State, str], ...]:
        prefix: list[str] = []
        decisions: list[tuple[State, str]] = []
        for action in self.auxiliary_steps:
            state = tuple(prefix)
            decisions.append((state, action))
            prefix.append(action)
        decisions.append((tuple(prefix), GENERATE))
        decisions.append(((GENERATE,), self.post_generation))
        return tuple(decisions)

    def as_text(self) -> str:
        return "->".join(self.steps)


def valid_next_actions(prefix: Sequence[str]) -> tuple[str, ...]:
    """Return legal pre-generation choices for an Alpha-SQL prefix."""

    prefix = tuple(prefix)
    if GENERATE in prefix or ACCEPT in prefix or REVISE in prefix:
        raise ValueError(f"prefix already passed generation: {prefix}")
    if len(prefix) != len(set(prefix)):
        raise ValueError(f"prefix repeats an action: {prefix}")
    if any(action not in AUXILIARY_ACTIONS for action in prefix):
        raise ValueError(f"unknown auxiliary action in prefix: {prefix}")
    if REPHRASE in prefix and prefix[0] != REPHRASE:
        raise ValueError("rephrase is legal only at the root")

    choices: list[str] = []
    if not prefix:
        choices.append(REPHRASE)
    choices.extend(action for action in PERMUTABLE_ACTIONS if action not in prefix)
    choices.append(GENERATE)
    return tuple(choices)


def validate_route(route: Route | Sequence[str]) -> None:
    steps = route.steps if isinstance(route, Route) else tuple(route)
    if len(steps) < 2:
        raise ValueError("route must contain generate and a terminal decision")
    if steps.count(GENERATE) != 1:
        raise ValueError("route must contain exactly one generate action")
    generate_index = steps.index(GENERATE)
    if generate_index != len(steps) - 2:
        raise ValueError("generate must immediately precede the terminal decision")
    if steps[-1] not in POST_GENERATION_ACTIONS:
        raise ValueError("route must end in accept or revise")
    prefix = steps[:generate_index]
    if len(prefix) != len(set(prefix)):
        raise ValueError("auxiliary actions may not repeat")
    if any(action not in AUXILIARY_ACTIONS for action in prefix):
        raise ValueError("route contains an unknown auxiliary action")
    if REPHRASE in prefix and prefix[0] != REPHRASE:
        raise ValueError("rephrase is legal only as the first action")
    # Exercise the same transition constraints used during sampling.
    seen: list[str] = []
    for action in prefix:
        if action not in valid_next_actions(seen):
            raise ValueError(f"illegal transition to {action} from {tuple(seen)}")
        seen.append(action)
    if GENERATE not in valid_next_actions(seen):
        raise ValueError(f"cannot generate from prefix {tuple(seen)}")


def enumerate_routes() -> tuple[Route, ...]:
    routes: list[Route] = []
    for length in range(len(PERMUTABLE_ACTIONS) + 1):
        for suffix in itertools.permutations(PERMUTABLE_ACTIONS, length):
            for include_rephrase in (False, True):
                prefix = ((REPHRASE,) if include_rephrase else ()) + suffix
                for terminal in POST_GENERATION_ACTIONS:
                    routes.append(Route(prefix + (GENERATE, terminal)))
    return tuple(sorted(routes))


def population_support_scores(
    outcome_keys: Sequence[Hashable | None],
) -> list[float]:
    """Compute empirical support among valid cross-route outcomes.

    Invalid executions are represented by ``None`` and receive zero support.
    The denominator contains valid outcomes only, matching Alpha-SQL's final
    execution-result voting rule.
    """

    counts = Counter(key for key in outcome_keys if key is not None)
    valid_count = sum(counts.values())
    if valid_count == 0:
        return [0.0] * len(outcome_keys)
    return [
        counts[key] / valid_count if key is not None else 0.0
        for key in outcome_keys
    ]


class RoutePolicy:
    """Tabular autoregressive distribution over legal action routes."""

    def __init__(self) -> None:
        self._probabilities: dict[State, dict[str, float]] = {}

    @staticmethod
    def choices_for_state(state: State) -> tuple[str, ...]:
        if state == (GENERATE,):
            return POST_GENERATION_ACTIONS
        return valid_next_actions(state)

    def probabilities(self, state: State) -> Mapping[str, float]:
        if state not in self._probabilities:
            choices = self.choices_for_state(state)
            probability = 1.0 / len(choices)
            self._probabilities[state] = {
                action: probability for action in choices
            }
        return dict(self._probabilities[state])

    def sample(self, rng: random.Random) -> Route:
        prefix: list[str] = []
        while True:
            state = tuple(prefix)
            distribution = self.probabilities(state)
            choice = rng.choices(
                tuple(distribution),
                weights=tuple(distribution.values()),
                k=1,
            )[0]
            if choice == GENERATE:
                break
            prefix.append(choice)
        post_distribution = self.probabilities((GENERATE,))
        terminal = rng.choices(
            tuple(post_distribution),
            weights=tuple(post_distribution.values()),
            k=1,
        )[0]
        return Route(tuple(prefix) + (GENERATE, terminal))

    def update_from_elites(
        self,
        elite_routes: Iterable[Route],
        *,
        old_weight: float,
        exploration_mix: float,
        max_total_variation: float | None = None,
    ) -> None:
        self.update_from_weighted_routes(
            ((route, 1.0) for route in elite_routes),
            old_weight=old_weight,
            exploration_mix=exploration_mix,
            max_total_variation=max_total_variation,
        )

    def update_from_weighted_routes(
        self,
        route_weights: Iterable[tuple[Route, float]],
        *,
        old_weight: float,
        exploration_mix: float,
        max_total_variation: float | None = None,
    ) -> None:
        """Fit the tabular policy from non-negative route evidence weights.

        ``max_total_variation`` is an optional trust region for each categorical
        route decision. ``None`` preserves the historical update exactly.
        """

        if not 0.0 <= old_weight < 1.0:
            raise ValueError("old_weight must be in [0, 1)")
        if not 0.0 <= exploration_mix < 1.0:
            raise ValueError("exploration_mix must be in [0, 1)")
        if max_total_variation is not None and not 0.0 < max_total_variation <= 1.0:
            raise ValueError("max_total_variation must be in (0, 1]")

        counts: dict[State, Counter[str]] = defaultdict(Counter)
        evidence = list(route_weights)
        if not evidence or any(weight <= 0.0 for _, weight in evidence):
            raise ValueError("route weights must be non-empty and positive")
        for route, weight in evidence:
            for state, action in route.decisions():
                counts[state][action] += weight

        for state, state_counts in counts.items():
            choices = self.choices_for_state(state)
            total = _ordered_float_sum(state_counts.values())
            old = self.probabilities(state)
            updated: dict[str, float] = {}
            uniform = 1.0 / len(choices)
            for action in choices:
                maximum_likelihood = state_counts[action] / total
                smoothed = old_weight * old[action] + (1.0 - old_weight) * maximum_likelihood
                updated[action] = (
                    (1.0 - exploration_mix) * smoothed
                    + exploration_mix * uniform
                )
            normalizer = _ordered_float_sum(updated.values())
            updated = {
                action: probability / normalizer
                for action, probability in updated.items()
            }
            total_variation = 0.5 * _ordered_float_sum(
                abs(updated[action] - old[action]) for action in choices
            )
            if (
                max_total_variation is not None
                and total_variation > max_total_variation
            ):
                scale = max_total_variation / total_variation
                updated = {
                    action: old[action] + scale * (updated[action] - old[action])
                    for action in choices
                }
            self._probabilities[state] = updated

    def snapshot(self) -> dict[str, dict[str, float]]:
        def state_name(state: State) -> str:
            return "post_generate" if state == (GENERATE,) else "root" if not state else "|".join(state)

        return {
            state_name(state): dict(sorted(distribution.items()))
            for state, distribution in sorted(self._probabilities.items())
        }

    def mean_entropy(self) -> float:
        if not self._probabilities:
            return 0.0
        entropies = []
        for distribution in self._probabilities.values():
            entropy = -_ordered_float_sum(
                probability * math.log(probability)
                for probability in distribution.values()
                if probability > 0.0
            )
            entropies.append(entropy)
        return _ordered_float_sum(entropies) / len(entropies)
