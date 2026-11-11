"""DataDesigner-native implementation of the SPS policy-update column."""
from __future__ import annotations

import json
from typing import Any

from data_designer.engine.column_generators.generators.base import ColumnGeneratorCellByCell
from sps.core.update import update_policy
from sps.integration.config import CEMUpdateConfig, CandidateArchiveConfig
from sps.core.archive import collect_candidates
from sps.core.route_execution import PublicQuestion


class CandidateArchiveGenerator(ColumnGeneratorCellByCell[CandidateArchiveConfig]):
    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        if any(not isinstance(data[column], str) for column in self.config.required_columns):
            raise TypeError("Candidate archive requires JSON-string inputs")
        question = PublicQuestion.model_validate_json(data[self.config.question_column])
        populations = [json.loads(data[column]) for column in self.config.population_columns]
        archive = collect_candidates(question, populations)
        return {**data, self.config.name: json.dumps(archive, ensure_ascii=False, allow_nan=False)}


class CEMUpdateGenerator(ColumnGeneratorCellByCell[CEMUpdateConfig]):
    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)
        # Keep variable-key policies out of Arrow struct unification. Otherwise
        # absent states can become null-valued states when rows are combined.
        policy = data[self.config.policy_column]
        population = data[self.config.population_column]
        if not isinstance(policy, str) or not isinstance(population, str):
            raise TypeError("CEM columns require JSON strings, not nested Arrow structs")
        if self.config.population_status_column:
            status = json.loads(data[self.config.population_status_column])["status"]
            if status != "ok":
                # Keep the failed question in the dataset and stop its later
                # populations. The status column explains why no update ran.
                output[self.config.name] = json.dumps({"probabilities": json.loads(policy),
                    "elite_ids": [], "population_size": 0}, allow_nan=False)
                return output
        scored = json.loads(population)
        result = update_policy(
            json.loads(policy), scored,
            elite_fraction=self.config.elite_fraction,
            old_weight=self.config.old_weight,
            exploration_mix=self.config.exploration_mix,
        )
        output[self.config.name] = json.dumps(result, sort_keys=True, allow_nan=False)
        return output
