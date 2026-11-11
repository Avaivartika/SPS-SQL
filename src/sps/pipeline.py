"""Run native SPS search and its integrated original post-correction workflow."""
from __future__ import annotations

import argparse
import fcntl
from functools import wraps
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import pandas as pd
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.interface.data_designer import DataDesigner

from sps.core.route_execution import PublicQuestion
from sps.core.schema_context import UnpreparedQuestion
from sps.core.sql_plan import database_path
from sps.integration.config import (
    CandidateArchiveConfig,
    CEMUpdateConfig,
    PopulationConfig,
    RawSelectionConfig,
    SchemaContextConfig,
)
from sps.model_manifest import validate as validate_model_manifest
from sps.prompts import prompt_digest


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_inputs(path: Path) -> list[UnpreparedQuestion]:
    rows = [UnpreparedQuestion.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row.question_id for row in rows}) != len(rows):
        raise ValueError("A nonempty fixed cohort with unique question IDs is required")
    return rows


def _compact_population(population: dict) -> dict:
    """Retain exactly the first-round state needed by a paired replay."""
    required = (
        "status", "question_id", "db_id", "attempted", "scored_population",
        "policy_used", "route_rng_after", "action_rng_after", "last_trajectory_id",
        "input_tokens", "output_tokens",
    )
    missing = set(required) - population.keys()
    if missing:
        raise ValueError(f"Cached first population is missing fields: {sorted(missing)}")
    attempts = []
    for attempt in population["attempted"]:
        keep = {
            key: attempt.get(key) for key in (
                "trajectory_id", "route", "status", "final_sql", "reward",
                "search_score", "outcome_key", "execution", "input_tokens",
                "output_tokens", "total_tokens", "error",
            )
        }
        attempts.append(keep)
    compact = {key: attempts if key == "attempted" else population[key] for key in required}
    # All-round voting and the candidate archive need the successful-path IDs,
    # including when round one is loaded from an immutable cache.
    successful_ids = [a["trajectory_id"] for a in attempts if a["status"] == "ok"]
    if "successful_ids" in population and population["successful_ids"] != successful_ids:
        raise ValueError("Cached successful-path IDs disagree with attempted paths")
    if set(successful_ids) != {a["trajectory_id"] for a in population["scored_population"]}:
        raise ValueError("Cached successful paths disagree with scored paths")
    compact["successful_ids"] = successful_ids
    return compact


def _load_first_round_cache(path: Path, rows: list[UnpreparedQuestion], *, model: str,
                            model_artifact: dict | None, max_tokens: int,
                            population_size: int, sql_samples: int, random_seed: int,
                            max_attempt_multiplier: int, invalid_retry_limit: int,
                            max_batches: int, inputs_sha256: str) -> tuple[dict[int, dict], dict]:
    """Validate and compact a completed first round for a paired ablation.

    The replay deliberately reuses the exact first-round model outputs and RNG
    states. Subsequent rounds continue from that exact first population.
    The two-round CEM and frozen-policy comparison differs solely in the policy
    supplied to round two.
    """
    protocol_path = path.parent / "protocol.json"
    if not path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("First-round replay requires search_artifacts.jsonl and its protocol.json")
    protocol = json.loads(protocol_path.read_text())
    expected_ids = [row.question_id for row in rows]
    expected_by_id = {row.question_id: row for row in rows}
    expected_search = {
        "population_size": population_size,
        "sql_samples": sql_samples,
        "random_seed": random_seed,
        "max_attempt_multiplier": max_attempt_multiplier,
        "invalid_retry_limit": invalid_retry_limit,
        "max_batches": max_batches,
    }
    observed_search = protocol.get("search_config", {})
    checks = {
        "question IDs": protocol.get("question_ids") == expected_ids,
        "public inputs": protocol.get("inputs_sha256") == inputs_sha256,
        "prompts": protocol.get("prompt_sha256") == prompt_digest(),
        "generator name": protocol.get("model_config", {}).get("model") == model,
        "maximum output tokens": protocol.get("model_config", {}).get(
            "inference_parameters", {}).get("max_tokens") == max_tokens,
        "generator weights": protocol.get("model_artifact") == model_artifact,
        "first-round search settings": all(
            observed_search.get(key) == value for key, value in expected_search.items()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError("First-round cache differs from this run: " + ", ".join(failed))
    cached = {}
    expected_id_set = set(expected_ids)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            artifact = json.loads(line)
            qid = artifact.get("question_id")
            if qid in cached or qid not in expected_id_set:
                raise ValueError("Duplicate or unexpected question in first-round cache")
            first = _compact_population(artifact["search_round_1"])
            if first["status"] != "ok" or len(first["scored_population"]) != population_size:
                raise ValueError(f"First-round cache is incomplete for question {qid}")
            expected = expected_by_id[qid]
            if ((first["question_id"], first["db_id"]) != (qid, artifact.get("db_id"))
                    or first["db_id"] != expected.db_id):
                raise ValueError("First-round cache identity mismatch")
            cached[qid] = first
    if list(sorted(cached)) != list(sorted(expected_ids)):
        raise ValueError("First-round cache does not cover the complete cohort")
    source = {"path": str(path.resolve()), "sha256": _digest(path),
              "protocol_sha256": _digest(protocol_path)}
    return cached, source


def _endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Use an HTTP(S) endpoint without embedded credentials")


def _exclusive_output(function):
    @wraps(function)
    def locked(*args, **kwargs):
        output = Path(kwargs["output"]).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        # Keep the inode after release, so racing resume processes always lock
        # the same file. No other run directory or model process is affected.
        with (output.parent / ("." + output.name + ".sps.lock")).open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another SPS process owns this output directory") from error
            return function(*args, **kwargs)
    return locked


@_exclusive_output
def run(*, inputs: Path, output: Path, database_root: Path,
        endpoint: str, model: str, parallel: int = 16, row_parallel: int = 16,
        max_tokens: int = 4096, request_timeout_seconds: int = 1800,
        api_key_env: str | None = None, population_size: int = 8, sql_samples: int = 3,
        search_rounds: int = 2,
        elite_fraction: float = .25, old_weight: float = .25, exploration_mix: float = .1,
        random_seed: int = 42, max_attempt_multiplier: int = 3,
        invalid_retry_limit: int = 15, max_batches: int = 32,
        model_manifest: Path | None = None, post_config: Path | None = None,
        resume: bool = False, managed_server_config: Path | None = None,
        first_round_cache: Path | None = None, freeze_cem_update: bool = False,
        direct_generation: bool = False) -> dict:
    if direct_generation and (population_size != 1 or search_rounds != 1 or max_attempt_multiplier != 1
                              or first_round_cache is not None or freeze_cem_update):
        raise ValueError("Direct generation requires one operation, one attempt, no learning or cache replay")
    invocation_started = time.monotonic()
    rows = _load_inputs(inputs)
    expected_by_id = {row.question_id: row for row in rows}
    for row in rows:
        database_path(database_root, row.db_id)
    _endpoint(endpoint)
    if parallel < 1 or row_parallel < 1 or max_tokens < 1 or request_timeout_seconds < 1:
        raise ValueError("Invalid model concurrency or output budget")
    if type(search_rounds) is not int or not 1 <= search_rounds <= 4:
        raise ValueError("Search rounds must be an integer between 1 and 4")
    key = os.environ[api_key_env] if api_key_env else None
    model_artifact = validate_model_manifest(model_manifest, model, max_tokens) if model_manifest else None
    if freeze_cem_update and first_round_cache is None:
        raise ValueError("A frozen CEM ablation requires an exact first-round cache")
    if freeze_cem_update and search_rounds != 2:
        raise ValueError("The frozen CEM ablation requires exactly two search rounds")
    post = post_manifest = schemas = None
    if post_config is not None:
        from sps.integration.workflow import prepare_post
        from sps.postprocess.config import file_digest
        post, post_manifest, schemas = prepare_post(post_config, rows)
        database_artifacts = {}
        for db_id in sorted({row.db_id for row in rows}):
            database = database_path(database_root, db_id)
            database_artifacts[db_id] = {"sqlite_sha256": file_digest(database),
                "descriptions": {p.name: file_digest(p)
                    for p in sorted((database.parent / "database_description").glob("*.csv"))}}
        post_manifest["database_artifacts"] = database_artifacts
    service = None
    if managed_server_config is not None:
        from sps.integration.serving import ManagedServerConfig, OwnedModelServer
        if post is None:
            raise ValueError("Managed GPU handover requires the integrated --post-config workflow")
        serving = ManagedServerConfig.model_validate_json(managed_server_config.read_text())
        serving.model_path = (managed_server_config.resolve().parent / serving.model_path).resolve()
        if not serving.model_path.is_dir():
            raise FileNotFoundError(serving.model_path)
        if model_artifact is not None and Path(model_artifact["base_model"]).resolve() != serving.model_path:
            raise ValueError("Managed server model differs from the validated generator manifest")
        service = OwnedModelServer(serving, endpoint, model, output / "generation_server.log", random_seed)
    if resume and post is None:
        raise ValueError("Native composite resume requires --post-config")
    if resume and not (output / "protocol.json").is_file():
        raise ValueError("Resume requires an existing matching SPS run")
    output.mkdir(parents=True, exist_ok=resume)

    def save(name: str, value) -> None:
        destination = output / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)

    population_columns = [f"search_round_{index}" for index in range(1, search_rounds + 1)]
    first = population_columns[0]
    updated = first + "_update"
    cached_first = cache_source = None
    if first_round_cache is not None:
        cached_first, cache_source = _load_first_round_cache(first_round_cache, rows,
            model=model, model_artifact=model_artifact, max_tokens=max_tokens,
            population_size=population_size, sql_samples=sql_samples, random_seed=random_seed,
            max_attempt_multiplier=max_attempt_multiplier, invalid_retry_limit=invalid_retry_limit,
            max_batches=max_batches, inputs_sha256=_digest(inputs))
    from sps.core.modes import direct_policy
    initial_policy = direct_policy() if direct_generation else {}
    seed_rows = []
    for row in rows:
        value = {"public_question": row.model_dump_json(), "initial_policy": json.dumps(initial_policy)}
        if cached_first is not None:
            population = cached_first[row.question_id]
            value[first] = json.dumps(population, ensure_ascii=False, allow_nan=False)
            value[first + "_policy"] = json.dumps(population["policy_used"], sort_keys=True,
                                                  allow_nan=False)
            value[first + "_scored_population"] = json.dumps(
                population["scored_population"], ensure_ascii=False, allow_nan=False)
            if freeze_cem_update:
                value[updated] = json.dumps({"probabilities": population["policy_used"],
                    "elite_ids": [], "population_size": len(population["scored_population"])},
                    sort_keys=True, allow_nan=False)
        seed_rows.append(value)
    if cached_first is None:
        seed = output / "public_seed.jsonl"
        seed_text = "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in seed_rows)
        if resume:
            if not seed.is_file() or seed.read_text() != seed_text:
                raise ValueError("Resume public questions differ from the original run")
        else:
            seed.write_text(seed_text)
    else:
        # Completed populations may exceed DuckDB's JSON object limit. Parquet
        # is DataDesigner's native large-row seed format and preserves every
        # replay field without relaxing process-wide parser safeguards.
        seed = output / "public_seed.parquet"
        if resume:
            if not seed.is_file() or pd.read_parquet(seed).to_dict(orient="records") != seed_rows:
                raise ValueError("Resume public questions differ from the original run")
        else:
            pd.DataFrame.from_records(seed_rows).to_parquet(seed, index=False)
    model_config = ModelConfig(alias="generator", model=model, provider="search-endpoint", skip_health_check=True,
        inference_parameters=ChatCompletionInferenceParams(temperature=.8, top_p=.8, max_tokens=max_tokens,
            timeout=request_timeout_seconds, max_parallel_requests=parallel))
    builder = DataDesignerConfigBuilder(model_configs=[model_config])
    builder.with_seed_dataset(LocalFileSeedSource(path=str(seed.resolve())))
    builder.add_column(SchemaContextConfig(name="question_input", database_root=str(database_root.resolve()),
                                           context_column="public_question"))
    population_options = dict(model_alias="generator", database_root=str(database_root.resolve()),
        question_column="question_input", population_size=population_size, samples=sql_samples,
        random_seed=random_seed, max_attempt_multiplier=max_attempt_multiplier,
        invalid_retry_limit=invalid_retry_limit, max_batches=max_batches)

    if cached_first is None:
        builder.add_column(PopulationConfig(name=first, policy_column="initial_policy", **population_options))
    search_trace_columns = [first]
    search_stages = ["schema_context", "round_1_replay" if cached_first is not None else "round_1"]
    for index, previous in enumerate(population_columns[:-1], start=2):
        updated = previous + "_update"
        if not freeze_cem_update:
            builder.add_column(CEMUpdateConfig(name=updated, policy_column=previous + "_policy",
                population_column=previous + "_scored_population",
                population_status_column=previous,
                elite_fraction=elite_fraction, old_weight=old_weight, exploration_mix=exploration_mix))
        current = population_columns[index - 1]
        builder.add_column(PopulationConfig(name=current, policy_column=updated,
            previous_population_column=previous, **population_options))
        search_trace_columns.extend([updated, current])
        search_stages.extend(["frozen_update" if freeze_cem_update else "cem_update", f"round_{index}"])
    search_stages.append("execution_vote")
    builder.add_column(RawSelectionConfig(name="raw_selection", population_column=population_columns[-1], population_columns=population_columns,
        database_root=str(database_root.resolve())))
    # Archive and vote over completed attempts from all populations.
    builder.add_column(CandidateArchiveConfig(name="candidate_archive",
        population_columns=population_columns))
    protocol = {"artifact_kind": "native_datadesigner_sps_lightweight_run_v1",
        "approved_search_source": "8e10868:tools/run_native_bird14b.py",
        "reference_labels_loaded": False, "questions": len(rows),
        "question_ids": [row.question_id for row in rows],
        "stages": search_stages,
        "archive_role": "trace_only", "final_candidate_scope": f"round_{search_rounds}",
        "inputs_sha256": _digest(inputs), "prompt_sha256": prompt_digest(),
        "model_config": model_config.model_dump(mode="json"), "model_artifact": model_artifact,
        "serving_config": {"max_parallel_requests": parallel,
                           "max_concurrent_row_groups": min(row_parallel, len(rows)),
                           "buffer_size": 1,
                           "request_timeout_seconds": request_timeout_seconds,
                           "request_seed_scheme": "sha256(global_seed,question_id,population,request_index,template,prompt)"},
        "search_config": {"rounds": search_rounds, "population_size": population_size,
            "sql_samples": sql_samples, "auxiliary_samples": 1,
            "elite_fraction": elite_fraction, "old_weight": old_weight,
            "exploration_mix": exploration_mix, "random_seed": random_seed,
            "max_attempt_multiplier": max_attempt_multiplier,
            "invalid_retry_limit": invalid_retry_limit, "max_batches": max_batches,
            "update_mode": "frozen" if freeze_cem_update else "cem",
            "first_round_reused": cached_first is not None,
            "first_round_source": cache_source},
        "data_designer_version": importlib.metadata.version("data-designer")}
    if post is not None:
        protocol.update(artifact_kind="native_sps_complete_run_v1", postprocessing=post_manifest,
            stages=[*protocol["stages"], "auxiliary_predictions_and_confidence", "original_post_correction"],
            archive_role="post_correction_candidates_and_evidence",
            final_candidate_scope=("original_post_over_both_rounds" if search_rounds == 2
                                   else "original_post_over_all_rounds"))
        protocol["managed_server"] = service.config.model_dump(mode="json") if service is not None else None
    protocol.update(module_ablation={"search_enabled": not direct_generation,
        "post_order": ["repair", "structural_selection", "semantic_selection"],
        "direct_generation_samples": sql_samples if direct_generation else None,
        "priors": (post.mode if post is not None else "disabled")},
        raw_candidate_scope="all_rounds", final_candidate_scope="all_rounds")
    if direct_generation:
        protocol["search_config"]["update_mode"] = "none"
    if resume and json.loads((output / "protocol.json").read_text()) != protocol:
        raise ValueError("Resume configuration, inputs, models or implementation changed; use a new output directory")
    builder.write_config(output / "workflow.yaml")
    save("protocol.json", protocol)
    # One question per native row group makes completed questions durable
    # immediately; concurrency is controlled independently.
    run_config = RunConfig(buffer_size=1,
        max_concurrent_row_groups=min(row_parallel, len(rows)),
        max_conversation_restarts=0, max_conversation_correction_steps=0,
        write_scheduler_events=True, otel_metrics_port=None)
    started = time.monotonic()
    preflight_seconds = started - invocation_started
    try:
        phases = {}
        if post is None:
            designer = DataDesigner(artifact_path=output / "framework", auto_configure_logging=False,
                model_providers=[ModelProvider(name="search-endpoint", endpoint=endpoint, api_key=key)])
            designer.set_run_config(run_config)
            created = designer.create(builder, num_records=len(rows), dataset_name="sps-lightweight")
            dataset_rows = created.load_dataset().to_dict(orient="records")
        else:
            from sps.integration.workflow import run_complete
            dataset, phases = run_complete(builder=builder, post=post, post_manifest=post_manifest,
                schemas=schemas, output=output, database_root=database_root,
                providers=[ModelProvider(name="search-endpoint", endpoint=endpoint, api_key=key)],
                run_config=run_config, num_records=len(rows), resume=resume, save=save, search_service=service)
            dataset_rows = dataset.to_dict(orient="records")
        indexed = {}
        for row in dataset_rows:
            question = PublicQuestion.model_validate_json(row["question_input"])
            if question.question_id in indexed:
                raise ValueError("Duplicate output question")
            expected = expected_by_id.get(question.question_id)
            observed = question.model_dump()
            observed.pop("schema_context")
            if expected is None or observed != expected.model_dump():
                raise ValueError("Public question changed during schema preparation")
            final = json.loads(row["post_correction"] if post is not None else row["raw_selection"])
            if post is not None:
                final["selected_sql"] = final["predicted_sql"]
            if (final.get("question_id"), final.get("db_id")) != (question.question_id, question.db_id):
                raise ValueError("Final artifact identity mismatch")
            indexed[question.question_id] = (row, final)
        if set(indexed) != set(expected_by_id):
            raise ValueError("Incomplete or substituted output cohort")
        artifacts = output / "search_artifacts.jsonl"
        trace_columns = ([*search_trace_columns, "raw_selection", "candidate_archive"] if post is None else
                         ["raw_selection", "candidate_archive"])
        artifacts.write_text("".join(json.dumps({"question_id": source.question_id,
            "db_id": source.db_id,
            **{name: json.loads(indexed[source.question_id][0][name]) for name in trace_columns}},
            ensure_ascii=False, allow_nan=False) + "\n" for source in rows))
        if post is not None:
            for column, filename in (("answer_priors", "answer_priors.jsonl"),
                                     ("post_correction", "post_artifacts.jsonl")):
                (output / filename).write_text("".join(indexed[source.question_id][0][column] + "\n" for source in rows))
            for stage in ("baseline", "structural", "minimal", "low", "high", "final"):
                save(stage + "_predictions.json", {str(qid): final["predictions"][stage]
                                                    for qid, (_, final) in indexed.items()})
        failures = [qid for qid, (_, final) in indexed.items()
                    if final.get("status") != "ok" or not isinstance(final.get("selected_sql"), str)]
        if failures and post is None:
            save("partial_predictions.json", {
                str(source.question_id): indexed[source.question_id][1]["selected_sql"]
                for source in rows if source.question_id not in failures})
            save("failed_questions.json", {"question_ids": failures,
                "failures": [{"question_id": qid, "selection": indexed[qid][1],
                    "round_status": {name: json.loads(indexed[qid][0][name])["status"]
                                     for name in population_columns}} for qid in failures]})
            search_label = "Two-round" if search_rounds == 2 else f"{search_rounds}-round"
            raise RuntimeError(f"{search_label} search failed on {len(failures)} questions")
        predictions = {str(source.question_id): indexed[source.question_id][1]["selected_sql"] for source in rows}
        totals = {}
        for token_type in ("input_tokens", "output_tokens"):
            counts = ([json.loads(row[name])[token_type]
                       for row, _ in indexed.values() for name in population_columns] if post is None else
                      [json.loads(row["candidate_archive"])["token_usage"][token_type] for row, _ in indexed.values()])
            if any(type(count) is not int or count < 0 for count in counts):
                if post is None:
                    raise ValueError("Search token usage is unavailable")
                totals[token_type] = None
            else:
                totals[token_type] = sum(counts)
        save("final_predictions.json", predictions)
        report = {"artifact_kind": "native_datadesigner_sps_lightweight_complete_v1",
            "questions": len(rows), "rounds": search_rounds, **totals,
            "candidate_count": sum(json.loads(row["candidate_archive"])["candidate_count"]
                                   for row, _ in indexed.values()),
            "elapsed_seconds": time.monotonic() - started,
            "artifacts_sha256": _digest(artifacts),
            "predictions_sha256": _digest(output / "final_predictions.json")}
        if post is not None:
            report.update(artifact_kind="native_sps_complete_v1", failed_question_ids=failures,
                status="complete_with_failures" if failures else "complete", phases=phases,
                preflight_seconds=preflight_seconds,
                invocation_wall_seconds=time.monotonic() - invocation_started,
                search_token_usage_complete=all(value is not None for value in totals.values()),
                post_mode=post.mode, auxiliary_inference_included=post.mode == "fresh",
                auxiliary_token_usage={name: sum(json.loads(row["answer_priors"])["token_usage"].get(name, 0)
                    for row, _ in indexed.values() if post.mode == "fresh") for name in
                    sorted({key for row, _ in indexed.values()
                            for key in json.loads(row["answer_priors"])["token_usage"]})})
            save("failed_questions.json", {"question_ids": failures,
                 "evaluation_denominator": len(rows), "failures_count_as_incorrect": True})
        save("completed.json", report)
        return report
    except BaseException as exc:
        save("interrupted.json", {"error_type": type(exc).__name__, "error": str(exc)[:2000],
                                  "elapsed_seconds": time.monotonic() - started})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "output", "database-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--row-parallel", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout-seconds", type=int, default=1800)
    parser.add_argument("--api-key-env")
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--search-rounds", type=int, choices=(1, 2, 3, 4), default=2,
                        help="Sampling populations; one CEM update precedes every population after the first")
    parser.add_argument("--sql-samples", type=int, default=3)
    parser.add_argument("--elite-fraction", type=float, default=.25)
    parser.add_argument("--old-weight", type=float, default=.25)
    parser.add_argument("--exploration-mix", type=float, default=.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-attempt-multiplier", type=int, default=3)
    parser.add_argument("--invalid-retry-limit", type=int, default=15)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--post-config", type=Path, help="Integrated original auxiliary and post-correction configuration")
    parser.add_argument("--resume", action="store_true", help="Resume matching native stages without redoing completed search")
    parser.add_argument("--managed-server-config", type=Path,
        help="Own a local generation server and release it before auxiliary inference; external endpoints are never stopped")
    parser.add_argument("--first-round-cache", type=Path,
        help="Reuse a validated search_artifacts.jsonl first round for a paired replay")
    parser.add_argument("--freeze-cem-update", action="store_true",
        help="Keep the cached first-round policy unchanged for round two")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
