"""Original post-correction model math primitives."""
from __future__ import annotations
import json
import math
import re
from typing import Any
import torch
from sps.postprocess.contracts import AnswerContract
from sps.postprocess.grounding import SchemaGroundingTarget, schema_column_catalog
from sps.postprocess.field_confidence import SEMANTIC_CONTRACT_FIELDS

def ground_contract(
    contract: AnswerContract,
    grounding: SchemaGroundingTarget | None,
    schema: dict[str, Any],
) -> tuple[AnswerContract | None, str | None]:
    if grounding is None:
        return None, "parse_failed"
    if len(grounding.slot_source_ids) != len(contract.ordered_slots):
        return None, "slot_count_mismatch"
    by_id = {value.column_id: value for value in schema_column_catalog(schema)}
    result = contract.model_copy(deep=True)
    for slot, identifiers in zip(
        result.ordered_slots, grounding.slot_source_ids, strict=True
    ):
        invalid = [value for value in identifiers if value not in by_id]
        if invalid:
            return None, "invalid_schema_id"
        slot.source_columns = [by_id[value].canonical_name for value in identifiers]
    return result, None

def score_contract_completions(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    *,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Score a left-padded batch while retaining per-field token spans."""

    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have equal length")
    if not prompts:
        return []

    sequences: list[list[int]] = []
    target_rows: list[list[int]] = []
    offset_rows: list[list[tuple[int, int]]] = []
    for prompt, completion in zip(prompts, completions, strict=True):
        prompt_encoding = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Transformers 5 returns a BatchEncoding here for Qwen3 tokenizers,
        # whereas Transformers 4 returned the token-id list directly.
        prompt_ids = (
            prompt_encoding["input_ids"]
            if hasattr(prompt_encoding, "keys")
            and "input_ids" in prompt_encoding
            else prompt_encoding
        )
        if prompt_ids and isinstance(prompt_ids[0], list):
            if len(prompt_ids) != 1:
                raise ValueError("expected one chat-template sequence")
            prompt_ids = prompt_ids[0]
        prompt_ids = [int(value) for value in prompt_ids]
        encoded = tokenizer(
            completion,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        completion_ids = [int(value) for value in encoded["input_ids"]]
        target_ids = [*completion_ids, int(tokenizer.eos_token_id)]
        sequences.append([*prompt_ids, *target_ids])
        target_rows.append(target_ids)
        offset_rows.append(
            [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        )
    maximum_length = max(len(row) for row in sequences)
    maximum_targets = max(len(row) for row in target_rows)
    input_ids = torch.full(
        (len(sequences), maximum_length),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
        device=model.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        input_ids[index, -len(sequence) :] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=model.device,
        )
        attention_mask[index, -len(sequence) :] = 1
    with torch.inference_mode():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=maximum_targets + 1,
        )
    results: list[dict[str, Any]] = []
    for row_index, (completion, targets, offsets) in enumerate(
        zip(completions, target_rows, offset_rows, strict=True)
    ):
        target_count = len(targets)
        # ``logits_to_keep`` is not implemented consistently across the
        # Transformers versions used by our two Conda environments.  Some
        # Qwen backends return only the requested suffix, while others accept
        # the argument but still return the complete sequence.  Every row is
        # left padded and therefore ends with its target.  Locate the causal
        # target window from the *actual* returned length instead of assuming
        # either backend behaviour.
        returned_length = int(result.logits.shape[1])
        start = returned_length - target_count - 1
        if start < 0:
            raise ValueError(
                "model returned too few logits to score the completion: "
                f"returned={returned_length}, target={target_count}"
            )
        logits = result.logits[row_index, start : start + target_count].float()
        if int(logits.shape[0]) != target_count:
            raise AssertionError("incomplete causal target-logit window")
        target_tensor = torch.tensor(targets, device=logits.device)
        token_logprobs = (
            torch.log_softmax(logits, dim=-1)
            .gather(1, target_tensor.unsqueeze(1))
            .squeeze(1)
        )
        values = [float(value) for value in token_logprobs.cpu().tolist()]
        ordered = sorted(values)
        fields = (*SEMANTIC_CONTRACT_FIELDS, "ordered_slots")
        field_scores: dict[str, float | None] = {}
        for field in fields:
            span = _top_level_json_value_span(completion, field)
            if span is None:
                field_scores[field] = None
                continue
            selected = [
                values[index]
                for index, (span_start, span_end) in enumerate(offsets)
                if span_end > span[0] and span_start < span[1]
            ]
            field_scores[field] = sum(selected) / len(selected) if selected else None
        results.append(
            {
                "tokens": len(values),
                "forward_tokens": len(sequences[row_index]),
                "mean_logprob": sum(values) / len(values),
                "minimum_logprob": min(values),
                "p10_logprob": ordered[max(0, math.ceil(len(ordered) * 0.1) - 1)],
                "field_mean_logprobs": field_scores,
            }
        )
    return results

def _top_level_json_value_span(text: str, field: str) -> tuple[int, int] | None:
    """Return the character span of one compact top-level JSON value."""

    marker = re.search(
        rf"{re.escape(json.dumps(field, ensure_ascii=False))}\s*:",
        text,
    )
    if marker is None:
        return None
    value_start = marker.end()
    while value_start < len(text) and text[value_start].isspace():
        value_start += 1
    try:
        _, relative_end = json.JSONDecoder().raw_decode(text[value_start:])
    except json.JSONDecodeError:
        return None
    return value_start, value_start + relative_end
