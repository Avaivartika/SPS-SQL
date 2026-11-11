"""Compose search, fresh auxiliary inference and correction in one native run."""
from __future__ import annotations

import json
from pathlib import Path

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.engine.storage.artifact_storage import ResumeMode

from sps.integration.config import AuxiliaryPredictionConfig, PostCorrectionConfig
from sps.integration.designer import SPSDataDesigner
from sps.postprocess.config import PostConfig, file_digest, load_schemas, object_digest


def prepare_post(path: Path, questions):
    config = PostConfig.read(path)
    schemas = load_schemas(config.schema_catalog)
    for question in questions:
        if question.db_id not in schemas:
            raise ValueError(f"No public schema for {question.db_id}")
    manifest = config.manifest()
    if config.mode == "cache":
        # Fail a stale or incomplete cache before spending any generator calls.
        from sps.postprocess.inference import AuxiliaryRuntime
        runtime = AuxiliaryRuntime(config, manifest["model_sha256"], schemas)
        try:
            runtime.predict(questions)
        finally:
            runtime.close()
    source = Path(__file__).resolve().parents[1]
    manifest["implementation_sha256"] = object_digest({
        str(p.relative_to(source)): file_digest(p)
        for p in sorted(source.rglob("*")) if p.is_file() and p.suffix in {".py", ".json", ".txt"}})
    return config, manifest, schemas


def run_complete(*, builder, post, post_manifest, schemas, output, database_root,
                 providers, run_config, num_records, resume, save, search_service=None):
    phases = json.loads((output / "phases.json").read_text()) if resume and (output / "phases.json").exists() else {}
    def record(phase, value):
        history = list(phases.get(phase, {}).get("attempts", []))
        if value["status"] == "running":
            history.append(dict(value))
        elif history:
            history[-1] = dict(value)
        else:
            history.append(dict(value))
        phases[phase] = {**value, "attempts": history,
            "recorded_attempt_seconds": sum(item.get("elapsed_seconds", 0) for item in history)}
        save("phases.json", phases)
    designer = SPSDataDesigner(post_config=post, post_manifest=post_manifest, schemas=schemas,
        search_run_config=run_config, phase_callback=record, artifact_path=output / "framework",
        auto_configure_logging=False, model_providers=providers, search_service=search_service)
    priors = DataDesignerConfigBuilder(model_configs=[])
    priors.add_column(AuxiliaryPredictionConfig(name="answer_priors", model_sha256=post_manifest["model_sha256"]))
    correction = DataDesignerConfigBuilder(model_configs=[])
    correction.add_column(PostCorrectionConfig(name="post_correction",
        database_root=str(database_root.resolve()), schema_catalog=str(post.schema_catalog)))
    priors.write_config(output / "auxiliary_workflow.yaml")
    correction.write_config(output / "post_workflow.yaml")
    workflow = designer.compose_workflow(name="sps")
    # Keep detailed model transcripts in search's native artifact directory.
    # Downstream stages carry only the public input, SQL archive and accounting.
    # Data transfer is a native column projection, not a rendered prompt.
    # Large execution results can exceed the template sandbox's text limit.
    # DataDesigner preserves the removed traces in dropped-columns-parquet-files.
    trace_columns = ["initial_policy", "search_round_1", "search_round_1_policy",
                     "search_round_1_scored_population"]
    search_columns = builder.build().columns
    if any("search_round_1_update" in column.required_columns for column in search_columns):
        trace_columns.append("search_round_1_update")
    for column in search_columns:
        if column.name.startswith("search_round_"):
            trace_columns.extend([column.name, *column.side_effect_columns])
    trace_columns = list(dict.fromkeys(trace_columns))
    compact = DropColumnsProcessorConfig(name="post_inputs", column_names=trace_columns)
    builder.add_processor(compact)
    builder.write_config(output / "workflow.yaml")
    workflow.add_stage("search", builder, num_records=num_records)
    workflow.add_stage("auxiliary", priors)
    workflow.add_stage("post_correction", correction)
    result = workflow.run(resume=ResumeMode.ALWAYS if resume else ResumeMode.NEVER)
    return result.load_dataset(), phases
