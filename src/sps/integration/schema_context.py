"""Native public-input preparation before generation dependencies run."""
import json
from pathlib import Path

from data_designer.engine.column_generators.generators.base import ColumnGeneratorCellByCell

from sps.core.schema_context import UnpreparedQuestion, build_context, prepare_question
from sps.integration.config import SchemaContextConfig


class SchemaContextGenerator(ColumnGeneratorCellByCell[SchemaContextConfig]):
    def generate(self, data):
        raw = UnpreparedQuestion.model_validate_json(data[self.config.context_column])
        context, provenance = build_context(Path(self.config.database_root), raw.db_id,
                                            timeout_seconds=self.config.timeout_seconds)
        return {**data, self.config.name: prepare_question(raw, context).model_dump_json(),
            self.config.provenance_column: json.dumps(provenance, ensure_ascii=False, allow_nan=False)}
