"""Controls for a frozen search/postprocessing module ablation."""
from sps.core.routes import RoutePolicy, GENERATE, ACCEPT
from sps.core.raw_selection import ordered_final_sqls


def direct_policy():
    return {"root": {a: float(a == GENERATE) for a in RoutePolicy.choices_for_state(())},
            "post_generate": {a: float(a == ACCEPT) for a in RoutePolicy.choices_for_state((GENERATE,))}}


def all_round_sqls(populations, qid, db):
    sqls = []
    for population in populations:
        if (population["question_id"], population["db_id"]) != (qid, db):
            raise ValueError("Population identity mismatch")
        # Preserve every archived successful path, including paths in an incomplete round.
        # This matches CandidateArchiveGenerator's successful_ids scope exactly.
        if population.get("successful_ids"):
            sqls.extend(ordered_final_sqls({**population, "status": "ok"}))
    return sqls
