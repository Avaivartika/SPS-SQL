"""One versioned configuration for the original, frozen post-correction path."""
from __future__ import annotations

import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


class PostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["original-14b-v1"] = "original-14b-v1"
    schema_catalog: Path
    mode: Literal["fresh", "cache"] = "fresh"
    base_model: Path | None = None
    structural_checkpoint: Path | None = None
    grounding_checkpoint: Path | None = None
    semantic_checkpoint: Path | None = None
    device: str = "cuda:0"
    batch_size: int = Field(default=8, ge=1)
    grounding_batch_size: int = Field(default=4, ge=1)
    confidence_batch_size: int = Field(default=8, ge=1)
    cpu_workers: int = Field(default=8, ge=1)
    random_seed: int = 42
    cache_path: Path | None = None
    cache_model_sha256: str | None = None

    @model_validator(mode="after")
    def sources(self):
        if self.mode == "fresh":
            if not all((self.base_model, self.structural_checkpoint,
                        self.grounding_checkpoint, self.semantic_checkpoint)):
                raise ValueError("Fresh inference requires the base model and all three trained adapters")
            if self.cache_path is not None or self.cache_model_sha256 is not None:
                raise ValueError("Fresh inference cannot silently reuse cached predictions")
        elif self.cache_path is None or not self.cache_model_sha256:
            raise ValueError("Cache replay requires a cache and its expected model fingerprint")
        return self

    @classmethod
    def read(cls, path: Path):
        values = json.loads(path.read_text())
        for key in ("schema_catalog", "base_model", "structural_checkpoint",
                    "grounding_checkpoint", "semantic_checkpoint", "cache_path"):
            if values.get(key) is not None:
                values[key] = str((path.resolve().parent / values[key]).resolve())
        return cls.model_validate(values)

    def manifest(self) -> dict:
        schema_sha = file_digest(self.schema_catalog)
        versions = {name: importlib.metadata.version(name) for name in ("sqlglot", "pydantic")}
        if self.mode == "cache":
            return {"config": self.model_dump(mode="json"), "schema_sha256": schema_sha,
                    "cache_sha256": file_digest(self.cache_path),
                    "model_sha256": self.cache_model_sha256, "runtime_versions": versions}
        missing = [name for name in ("torch", "transformers", "peft", "accelerate")
                   if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError("Fresh auxiliary dependencies missing: " + ", ".join(missing)
                               + "; install sps-datadesigner[post] before starting search")
        artifacts = {}
        for name, directory in (("base", self.base_model),
                                ("structural", self.structural_checkpoint),
                                ("grounding", self.grounding_checkpoint),
                                ("semantic", self.semantic_checkpoint)):
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            if name != "base":
                for required in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                                 "tokenizer/tokenizer_config.json"):
                    if not (directory / required).is_file():
                        raise FileNotFoundError(directory / required)
            paths = sorted(p for p in directory.rglob("*") if p.is_file()
                           and p.suffix in {".json", ".safetensors", ".txt", ".model"}
                           and ".cache" not in p.relative_to(directory).parts)
            if not any(p.suffix == ".safetensors" for p in paths):
                raise ValueError(f"No model weights in {directory}")
            artifacts[name] = {str(p.relative_to(directory)): file_digest(p) for p in paths}
        for name in ("torch", "transformers", "peft", "accelerate"):
            versions[name] = importlib.metadata.version(name)
        inference_identity = {"model_artifacts": artifacts, "runtime_versions": versions,
            "version": self.version, "batch_size": self.batch_size,
            "grounding_batch_size": self.grounding_batch_size,
            "confidence_batch_size": self.confidence_batch_size, "seed": self.random_seed,
            "implementation": {name: file_digest(Path(__file__).parent / name)
                               for name in ("contracts.py", "grounding.py", "model_math.py", "inference.py")}}
        return {"config": self.model_dump(mode="json"), "schema_sha256": schema_sha,
                "model_artifacts": artifacts, "model_sha256": object_digest(inference_identity),
                "runtime_versions": versions}


def load_schemas(path: Path) -> dict[str, dict]:
    from sps.postprocess.grounding import schema_column_catalog
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("Schema catalog must be a list of public database schemas")
    schemas = {}
    allowed = {"db_id", "table_names", "table_names_original", "column_names", "column_names_original",
               "column_types", "primary_keys", "foreign_keys"}
    for row in rows:
        if set(row) - allowed:
            raise ValueError("Schema catalog contains undeclared fields")
        schema_column_catalog(row)
        if row["db_id"] in schemas:
            raise ValueError("Duplicate schema identity")
        schemas[row["db_id"]] = row
    return schemas
