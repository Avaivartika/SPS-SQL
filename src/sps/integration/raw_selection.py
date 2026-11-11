"""Native SQL voting over final paths; no learned selector or repair fallback."""
from functools import lru_cache
import json
from pathlib import Path
import time

from data_designer.engine.column_generators.generators.base import ColumnGeneratorCellByCell
from sps.core.execution import execute
from sps.core.raw_selection import ordered_final_sqls, select_raw, measure_execution_time
from sps.core.route_execution import PublicQuestion
from sps.core.sql_plan import database_path
from sps.integration.config import RawSelectionConfig


class RawSelectionGenerator(ColumnGeneratorCellByCell[RawSelectionConfig]):
    def generate(self, data):
        question = PublicQuestion.model_validate_json(data[self.config.question_column])
        populations = [json.loads(data[c]) for c in
                       (self.config.population_columns or [self.config.population_column])]
        from sps.core.modes import all_round_sqls
        sqls = all_round_sqls(populations, question.question_id, question.db_id)
        record = {"question_id": question.question_id, "db_id": question.db_id,
            "status": "blocked", "execution_time_repeats": 20}
        if not sqls:
            record["error"] = "previous_population_failed"
        else:
            started = time.perf_counter()
            try:
                database = database_path(Path(self.config.database_root), question.db_id)
                @lru_cache(maxsize=None)
                def execute_one(sql):
                    return execute(database, sql, timeout_seconds=60)
                selected = select_raw(sqls, execute_one=execute_one,
                    measure_time=lambda sql, repeat: measure_execution_time(database, sql, repeat=repeat))
                record.update(selected, status="ok" if selected["selected_index"] is not None else "failed")
            except (ValueError, OSError) as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
            record["wall_seconds"] = time.perf_counter() - started
        return {**data, self.config.name: json.dumps(record, ensure_ascii=False, allow_nan=False)}
