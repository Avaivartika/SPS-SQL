"""Original postprocessing as an explicit, model-free native workflow operation."""
from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import sqlite3
import time

from sps.core.schema_context import UnpreparedQuestion
from sps.core.sql_plan import database_path
from sps.postprocess.confidence import ConfidenceOptions, gate_proposals
from sps.postprocess.contracts import AnswerContract
from sps.postprocess.evidence import candidate_evidence
from sps.postprocess.field_confidence import apply_field_confidence_gate
from sps.postprocess.proposal_gates import gate_support, merge_proposals
from sps.postprocess.selection import select
from sps.postprocess.sql_edits import generate_minimal_contract_candidates, select_conservative_candidate
from sps.postprocess.types import AnswerPriors


def process_row(question_json: str, prior_json: str, selection_json: str,
                archive_json: str, schema: dict, database_root: str) -> str:
    """Pickle-safe CPU task dispatched by the native per-row generator."""
    started = time.monotonic()
    question = UnpreparedQuestion.model_validate_json(question_json)
    prior = AnswerPriors.model_validate_json(prior_json)
    selection = json.loads(selection_json)
    if selection.get("selected_sql") is None and selection.get("status") != "ok":
        selection["selected_sql"] = "ERROR"
    output = correct(question, schema, selection, json.loads(archive_json), prior, Path(database_root))
    output["elapsed_seconds"] = time.monotonic() - started
    return json.dumps(output, ensure_ascii=False, allow_nan=False)


def calibration(kind: str) -> dict:
    if kind not in {"structural", "semantic"}:
        raise ValueError("Unknown original calibration")
    return json.loads(files("sps.postprocess").joinpath("calibration", kind + ".json").read_text())


def minimal_edit(source: dict, schema: dict, database_root: Path) -> dict:
    row = dict(source)
    original = str(row["predicted_sql"])
    row.update(original_sql=original, selected_sql=original, repair_accepted=False,
               selected_edits=[], candidate_error=None)
    if row.get("predicted_contract") is None:
        return row
    try:
        target = AnswerContract.model_validate(row["predicted_contract"])
        candidates = generate_minimal_contract_candidates(original, target, schema)
        selection = select_conservative_candidate(candidates, target, minimum_margin=.5)
        chosen, accepted = selection.selected.sql, selection.accepted_repair
        if accepted:
            connection = None
            try:
                database = database_path(database_root, row["db_id"])
                connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
                connection.execute("PRAGMA query_only=ON")
                connection.execute(f"EXPLAIN QUERY PLAN {chosen}").fetchall()
                row["selected_plan_error"] = None
            except sqlite3.Error as error:
                accepted = False
                row["selected_plan_error"] = f"{type(error).__name__}:{error}"
            finally:
                if connection is not None:
                    connection.close()
        row.update(selected_sql=chosen if accepted else original, repair_accepted=accepted,
                   selected_edits=list(selection.selected.edits), static_score_margin=selection.score_margin,
                   candidate_count=len(candidates))
    except Exception as error:
        row["candidate_error"] = f"{type(error).__name__}: {error}"
    return row


def correct(question: UnpreparedQuestion, schema: dict, baseline: dict, archive: dict,
            priors: AnswerPriors, database_root: Path) -> dict:
    priors.validate_input(question, schema)
    for record in (baseline, archive):
        if (record["question_id"], record["db_id"]) != (question.question_id, question.db_id):
            raise ValueError("Postprocessing input identity mismatch")
    qid, db = question.question_id, question.db_id
    original = baseline["selected_sql"]
    if not isinstance(original, str):
        raise ValueError("Raw selection must include SQL text or explicit ERROR")
    candidates, evidence = candidate_evidence(archive, database_root)
    def context(prediction):
        return {"source_id": str(qid), "db_id": db, "question": question.question,
                "evidence": question.evidence,
                "predicted_contract": prediction.contract.model_dump(mode="json") if prediction.contract else None}
    def score(prediction):
        return {"question_id": qid, "db_id": db, **prediction.confidence.model_dump()}
    minimal = minimal_edit({**context(priors.structural), "predicted_sql": original}, schema, database_root)
    minimal_rows, _ = apply_field_confidence_gate([minimal], [score(priors.structural)], calibration("structural"),
                                                  disabled_families={"order_removal"})
    minimal = minimal_rows[0]
    structural, structural_record = select(minimal["predicted_sql"], context(priors.structural), candidates, schema,
                                            semantic=False, weight=.2, gain=3.)
    outputs = {"baseline": original, "structural": structural["predicted_sql"], "minimal": minimal["predicted_sql"]}
    proposals, decisions = [], {}
    for name, weight, gain in (("low", .1, 2.), ("high", .4, 3.)):
        selected, record = select(outputs["structural"], context(priors.semantic), candidates, schema,
                                  semantic=True, weight=weight, gain=gain)
        supported = gate_support([selected], {qid: record}, .25)
        options = ConfidenceOptions("any", True, ("group_by_columns",), ("ordered_slots",),
            ("group_by_columns",), (), True, "legacy_uncalibrated", 1, True, True, False)
        gated = gate_proposals(supported["rows"], {qid: score(priors.semantic)},
                               calibration("semantic"), {db: schema}, options)
        proposals.append({qid: gated["rows"][0]})
        outputs[name] = gated["predictions"][str(qid)]
        decisions[name] = gated["rows"][0]["semantic_confidence_gate"]
    merged = merge_proposals(*proposals)
    outputs["final"] = merged["predictions"][str(qid)]
    return {"artifact_kind": "sps_repair_first_post_v1", "question_id": qid, "db_id": db,
            "status": "failed" if outputs["final"] == "ERROR" else "ok",
            "predicted_sql": outputs["final"], "predictions": outputs,
            "input_sha256": priors.input_sha256, "prior_model_sha256": priors.model_sha256,
            "evidence": evidence, "decisions": {"structural": structural_record,
                "minimal": minimal["field_confidence_gate"], **decisions,
                "final": merged["rows"][0]["semantic_proposal_union"]}}
