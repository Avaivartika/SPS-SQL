"""Strict, label-free boundaries between native inference and SQL correction."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from sps.core.schema_context import UnpreparedQuestion
from sps.postprocess.contracts import AnswerContract


def input_fingerprint(question: UnpreparedQuestion, schema: dict) -> str:
    payload = {"question": question.model_dump(mode="json"), "schema": schema}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class ConfidenceScore(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mean_logprob: float = Field(allow_inf_nan=False, le=0)
    field_mean_logprobs: dict[str, float | None]

    @model_validator(mode="after")
    def finite_fields(self):
        import math
        if any(v is not None and (not math.isfinite(v) or v > 0)
               for v in self.field_mean_logprobs.values()):
            raise ValueError("Field confidence must contain finite nonpositive log probabilities")
        return self


class PriorPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    contract: AnswerContract | None
    confidence: ConfidenceScore
    raw_prediction: StrictStr
    error: StrictStr | None = None


class AnswerPriors(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    artifact_kind: Literal["sps_answer_priors_v1"] = "sps_answer_priors_v1"
    question_id: StrictInt
    db_id: StrictStr
    input_sha256: StrictStr
    model_sha256: StrictStr
    structural: PriorPrediction
    semantic: PriorPrediction
    grounding_raw: StrictStr = ""
    timings: dict[str, float] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    source: Literal["inference", "cache"] = "inference"

    def validate_input(self, question: UnpreparedQuestion, schema: dict) -> None:
        if (self.question_id, self.db_id) != (question.question_id, question.db_id):
            raise ValueError("Auxiliary prediction belongs to a different question/database")
        if self.input_sha256 != input_fingerprint(question, schema):
            raise ValueError("Auxiliary prediction uses a different question, evidence or schema")
