"""Native DataDesigner columns for batched priors and CPU SQL correction."""
from __future__ import annotations

import asyncio
from pathlib import Path

from data_designer.config.scheduling import SchedulingMetadata
from data_designer.engine.column_generators.generators.base import ColumnGeneratorCellByCell, ColumnGeneratorFullColumn

from sps.core.schema_context import UnpreparedQuestion
from sps.integration.config import AuxiliaryPredictionConfig, PostCorrectionConfig
from sps.postprocess.config import load_schemas
from sps.postprocess.processor import process_row


class AuxiliaryPredictionGenerator(ColumnGeneratorFullColumn[AuxiliaryPredictionConfig]):
    def get_scheduling_metadata(self):
        return SchedulingMetadata.custom_model("sps", "answer-priors", self.config.model_sha256)

    def generate(self, data):
        runtime = self.resource_provider.auxiliary_runtime
        if runtime is None or runtime.model_sha256 != self.config.model_sha256:
            raise ValueError("Native auxiliary resource is absent or has the wrong model identity")
        questions = [UnpreparedQuestion.model_validate_json(value)
                     for value in data[self.config.question_column]]
        priors = runtime.predict(questions)
        if len(priors) != len(questions):
            raise ValueError("Auxiliary inference changed the cohort size")
        for question, prior in zip(questions, priors, strict=True):
            prior.validate_input(question, runtime.schemas[question.db_id])
            if prior.model_sha256 != self.config.model_sha256:
                raise ValueError("Auxiliary model identity changed")
        output = data.copy()
        output[self.config.name] = [p.model_dump_json() for p in priors]
        return output


class PostCorrectionGenerator(ColumnGeneratorCellByCell[PostCorrectionConfig]):
    def _initialize(self):
        self.schemas = load_schemas(Path(self.config.schema_catalog))

    def _arguments(self, data):
        question = UnpreparedQuestion.model_validate_json(data[self.config.question_column])
        return (data[self.config.question_column], data[self.config.priors_column],
                data[self.config.selection_column], data[self.config.archive_column],
                self.schemas[question.db_id], self.config.database_root)

    def generate(self, data):
        return {**data, self.config.name: process_row(*self._arguments(data))}

    async def agenerate(self, data):
        executor = self.resource_provider.post_executor
        if executor is None:
            raise ValueError("Native post-correction CPU resource is absent")
        result = await asyncio.get_running_loop().run_in_executor(executor, process_row, *self._arguments(data))
        return {**data, self.config.name: result}
