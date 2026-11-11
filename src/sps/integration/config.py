"""Data declarations for the SPS CEM column plugin."""
from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator




class AuxiliaryPredictionConfig(SingleColumnConfig):
    column_type: Literal["sps-answer-priors"] = "sps-answer-priors"
    question_column: str = "public_question"
    model_sha256: str

    @property
    def required_columns(self) -> list[str]:
        return [self.question_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return []


class PostCorrectionConfig(SingleColumnConfig):
    column_type: Literal["sps-post-correction"] = "sps-post-correction"
    question_column: str = "public_question"
    priors_column: str = "answer_priors"
    selection_column: str = "raw_selection"
    archive_column: str = "candidate_archive"
    database_root: str
    schema_catalog: str

    @property
    def required_columns(self) -> list[str]:
        return [self.question_column, self.priors_column, self.selection_column, self.archive_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return []


class SchemaContextConfig(SingleColumnConfig):
    column_type: Literal["sps-schema-context"] = "sps-schema-context"
    database_root: str
    context_column: str = "public_question"
    timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)

    @property
    def provenance_column(self) -> str:
        return self.name + "__schema_provenance"

    @property
    def required_columns(self) -> list[str]:
        return [self.context_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return [self.provenance_column]


class RawSelectionConfig(SingleColumnConfig):
    column_type: Literal["sps-raw-selection"] = "sps-raw-selection"
    database_root: str
    question_column: str = "question_input"
    population_column: str
    population_columns: list[str] | None = None

    @property
    def required_columns(self) -> list[str]:
        return [self.question_column, *(self.population_columns or [self.population_column])]

    @property
    def side_effect_columns(self) -> list[str]:
        return []

class CandidateArchiveConfig(SingleColumnConfig):
    column_type: Literal["sps-candidate-archive"] = "sps-candidate-archive"
    question_column: str = "question_input"
    population_columns: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_populations(self):
        if len(set(self.population_columns)) != len(self.population_columns):
            raise ValueError("Population columns must be unique")
        return self

    @property
    def required_columns(self) -> list[str]:
        return [self.question_column, *self.population_columns]

    @property
    def side_effect_columns(self) -> list[str]:
        return []




class PopulationConfig(SingleColumnConfig):
    column_type: Literal["sps-population"] = "sps-population"
    model_alias: str
    database_root: str
    question_column: str = "question_input"
    policy_column: str = "initial_policy"
    previous_population_column: str | None = None
    random_seed: int = 42
    population_size: int = Field(default=8, ge=1)
    max_attempt_multiplier: int = Field(default=3, ge=1)
    samples: int = Field(default=3, ge=1)
    invalid_retry_limit: int = Field(default=15, ge=0)
    max_batches: int = Field(default=32, ge=1)
    sql_max_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    max_rows: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_continuation(self):
        if self.previous_population_column == self.name:
            raise ValueError("A population cannot continue itself")
        return self

    @property
    def policy_output(self) -> str:
        return f"{self.name}_policy"

    @property
    def scored_output(self) -> str:
        return f"{self.name}_scored_population"

    @property
    def required_columns(self) -> list[str]:
        return [self.question_column, self.policy_column] + (
            [self.previous_population_column] if self.previous_population_column else [])

    @property
    def side_effect_columns(self) -> list[str]:
        return [self.policy_output, self.scored_output]


class CEMUpdateConfig(SingleColumnConfig):
    column_type: Literal["sps-cem-update"] = "sps-cem-update"
    policy_column: str = "policy_before"
    population_column: str = "scored_population"
    population_status_column: str | None = None
    elite_fraction: float = Field(default=0.25, gt=0, le=1)
    old_weight: float = Field(default=0.25, ge=0, lt=1)
    exploration_mix: float = Field(default=0.1, ge=0, lt=1)


    @property
    def required_columns(self) -> list[str]:
        return [self.policy_column, self.population_column] + (
            [self.population_status_column] if self.population_status_column else [])

    @property
    def side_effect_columns(self) -> list[str]:
        return []
