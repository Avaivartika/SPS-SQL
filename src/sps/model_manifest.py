"""Bind an SPS run to an immutable generator model artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def build(base_model: Path, served_model: str) -> dict:
    base_model = base_model.resolve(strict=True)
    files = [base_model / "config.json", base_model / "model.safetensors.index.json",
             base_model / "tokenizer_config.json"]
    # Some models store their active template outside tokenizer_config.json.
    files += [base_model / name for name in ("chat_template.jinja", "generation_config.json",
              "tokenizer.json", "merges.txt", "vocab.json", "tokenizer.model", "special_tokens_map.json")
              if (base_model / name).is_file()]
    weights = sorted(base_model.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError("No local safetensors weight shards were found")
    index = json.loads((base_model / "model.safetensors.index.json").read_text())
    expected = set(index.get("weight_map", {}).values())
    if not expected or expected != {p.name for p in weights}:
        raise ValueError("Model shard files do not match the weight index")
    files += weights
    manifest = {"artifact_kind": "sps_generator_model_manifest_v1", "served_model": served_model,
                "base_model": str(base_model),
                "weight_fingerprint_scope": "all local safetensors shards and active configuration"}
    manifest["files"] = {str(path): digest(path) for path in files}
    return manifest


def validate(path: Path, model: str, max_tokens: int) -> dict:
    manifest = json.loads(path.read_text())
    if manifest["artifact_kind"] != "sps_generator_model_manifest_v1" or manifest["served_model"] != model:
        raise ValueError("Model manifest does not match requested model")
    if any(digest(filename) != checksum for filename, checksum in manifest["files"].items()):
        raise ValueError("Model artifact changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.base_model, args.served_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
