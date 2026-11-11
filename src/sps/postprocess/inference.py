"""Batched native auxiliary inference. One base, three adapters, no SQL labels."""
from __future__ import annotations

import gc
import json
import threading
import time

from sps.postprocess.config import PostConfig
from sps.postprocess.contracts import (AnswerContract, CONTRACT_SYSTEM_PROMPT,
    STRUCTURAL_CONTRACT_SYSTEM_PROMPT, render_contract_prompt, render_structural_contract_prompt)
from sps.postprocess.grounding import GROUNDING_SYSTEM_PROMPT, SchemaGroundingTarget, render_grounding_prompt
from sps.postprocess.types import AnswerPriors, ConfidenceScore, PriorPrediction, input_fingerprint

SYSTEM_PROMPTS = {"structural": STRUCTURAL_CONTRACT_SYSTEM_PROMPT,
                  "grounding": GROUNDING_SYSTEM_PROMPT, "semantic": CONTRACT_SYSTEM_PROMPT}


def prediction_prompts(name, questions, schemas):
    renderer = {"structural": render_structural_contract_prompt, "semantic": render_contract_prompt}[name]
    return [renderer(question=q.question, evidence=q.evidence, schema=schemas[q.db_id]) for q in questions]


def parse_prediction(text: str, schema):
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return schema.model_validate(json.loads(text[start:end + 1]))
    except (ValueError, TypeError):
        return None


class AuxiliaryRuntime:
    """Owned by one native auxiliary stage, released even if generation fails."""
    def __init__(self, config: PostConfig, model_sha256: str, schemas: dict):
        self.config, self.model_sha256, self.schemas = config, model_sha256, schemas
        self.model = None
        self.tokenizers = {}
        self.cache = None
        self.lock = threading.Lock()
        self.load_seconds = 0.0

    def _load(self):
        if self.config.mode == "cache":
            if self.cache is None:
                self.cache = {}
                with self.config.cache_path.open() as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        prior = AnswerPriors.model_validate_json(line)
                        if prior.question_id in self.cache:
                            raise ValueError("Duplicate cached auxiliary question")
                        if prior.model_sha256 != self.model_sha256:
                            raise ValueError("Cached prediction uses a different model fingerprint")
                        self.cache[prior.question_id] = prior
            return
        if self.model is not None:
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install sps-datadesigner[post] for fresh auxiliary inference") from error
        started = time.monotonic()
        torch.manual_seed(self.config.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        base = AutoModelForCausalLM.from_pretrained(str(self.config.base_model),
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa",
            local_files_only=True)
        self.model = PeftModel.from_pretrained(base, str(self.config.structural_checkpoint / "adapter"),
                                              adapter_name="structural", local_files_only=True)
        for name in ("grounding", "semantic"):
            self.model.load_adapter(str(getattr(self.config, name + "_checkpoint") / "adapter"),
                                    adapter_name=name, local_files_only=True)
        self.model.to(self.config.device).eval()
        for name in ("structural", "grounding", "semantic"):
            tokenizer = AutoTokenizer.from_pretrained(str(getattr(self.config, name + "_checkpoint") / "tokenizer"),
                                                      use_fast=True, local_files_only=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            self.tokenizers[name] = tokenizer
        self.load_seconds = time.monotonic() - started

    def _generate(self, name, prompts, max_input, max_new, batch_size):
        import torch
        model, tokenizer = self.model, self.tokenizers[name]
        model.set_adapter(name)
        model.eval()
        outputs, usage = [], []
        system = SYSTEM_PROMPTS[name]
        for start in range(0, len(prompts), batch_size):
            texts = [tokenizer.apply_chat_template([
                {"role": "system", "content": system}, {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for p in prompts[start:start + batch_size]]
            encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                max_length=max_input).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=max_new,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            new = generated[:, encoded["input_ids"].shape[1]:]
            outputs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
            for index, ids in enumerate(new.tolist()):
                length = ids.index(tokenizer.eos_token_id) + 1 if tokenizer.eos_token_id in ids else len(ids)
                usage.append({"input_tokens": int(encoded["attention_mask"][index].sum()),
                              "output_tokens": length})
        return outputs, usage

    def predict(self, questions):
        # The full-column scheduler is serialized, and this lock also guards
        # direct use: adapter switching must never race across batches.
        with self.lock:
            self._load()
            if self.cache is not None:
                output = []
                for question in questions:
                    prior = self.cache[question.question_id]
                    prior.validate_input(question, self.schemas[question.db_id])
                    output.append(prior.model_copy(update={"source": "cache"}))
                return output
            from sps.postprocess.model_math import ground_contract, score_contract_completions
            raw, parsed, scores, usage, elapsed = {}, {}, {}, {}, {}
            for name, max_new in (("structural", 320), ("grounding", 128), ("semantic", 640)):
                started = time.monotonic()
                if name == "grounding":
                    valid = [i for i, value in enumerate(parsed["structural"]) if value is not None]
                    ground_prompts = [render_grounding_prompt(question=questions[i].question,
                        evidence=questions[i].evidence, schema=self.schemas[questions[i].db_id],
                        contract=parsed["structural"][i]) for i in valid]
                    ground_raw, ground_usage = self._generate(name, ground_prompts, 9088, max_new,
                                                              self.config.grounding_batch_size)
                    raw[name] = [""] * len(questions)
                    usage[name] = [{"input_tokens": 0, "output_tokens": 0} for _ in questions]
                    for i, text, count in zip(valid, ground_raw, ground_usage, strict=True):
                        raw[name][i], usage[name][i] = text, count
                        parsed["structural"][i], _ = ground_contract(parsed["structural"][i],
                            parse_prediction(text, SchemaGroundingTarget), self.schemas[questions[i].db_id])
                else:
                    prompts = prediction_prompts(name, questions, self.schemas)
                    raw[name], usage[name] = self._generate(name, prompts, 4800, max_new, self.config.batch_size)
                    parsed[name] = [parse_prediction(t, AnswerContract) for t in raw[name]]
                    elapsed[name + "_generation"] = time.monotonic() - started
                    started = time.monotonic()
                    scores[name] = []
                    for start in range(0, len(questions), self.config.confidence_batch_size):
                        stop = start + self.config.confidence_batch_size
                        scores[name].extend(score_contract_completions(self.model, self.tokenizers[name],
                            prompts[start:stop], [t.strip() for t in raw[name][start:stop]],
                            system_prompt=SYSTEM_PROMPTS[name]))
                    elapsed[name + "_confidence"] = time.monotonic() - started
                if name == "grounding":
                    elapsed[name + "_generation"] = time.monotonic() - started
            result = []
            for i, question in enumerate(questions):
                predictions = {name: PriorPrediction(contract=parsed[name][i],
                    confidence=ConfidenceScore(mean_logprob=scores[name][i]["mean_logprob"],
                        field_mean_logprobs=scores[name][i]["field_mean_logprobs"]),
                    raw_prediction=raw[name][i], error=None if parsed[name][i] is not None else "parse_or_grounding_failed")
                    for name in ("structural", "semantic")}
                result.append(AnswerPriors(question_id=question.question_id, db_id=question.db_id,
                    input_sha256=input_fingerprint(question, self.schemas[question.db_id]),
                    model_sha256=self.model_sha256, **predictions, grounding_raw=raw["grounding"][i],
                    timings={"batch_amortized_" + key: value / len(questions) for key, value in elapsed.items()},
                    token_usage={**{name + "_" + key: usage[name][i][key]
                                 for name in usage for key in ("input_tokens", "output_tokens")},
                        **{name + "_confidence_" + key: scores[name][i][key]
                           for name in scores for key in ("tokens", "forward_tokens")}}))
            return result

    def close(self):
        self.model = None
        self.tokenizers.clear()
        self.cache = None
        gc.collect()
        # Do not import CUDA at all for cache-only runs.
        if self.config.mode == "fresh":
            import sys
            torch = sys.modules.get("torch")
            if torch is not None and self.config.device.startswith("cuda") and torch.cuda.is_available():
                with torch.cuda.device(self.config.device):
                    torch.cuda.empty_cache()
