"""Optional owned vLLM service for sequential reuse of the same GPUs.

External endpoints are never stopped. This class only terminates the process
group it started itself, after the native search stage leaves its context.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ManagedServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_path: Path
    cuda_visible_devices: str
    python: str = sys.executable
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=.82, gt=0, lt=1)
    max_model_len: int = Field(default=32768, ge=1)
    max_num_seqs: int = Field(default=64, ge=1)
    max_num_batched_tokens: int = Field(default=32768, ge=1)
    enforce_eager: bool = False
    language_model_only: bool = False
    default_chat_template_kwargs: dict[str, bool | str | int | float] = Field(default_factory=dict)
    startup_timeout_seconds: int = Field(default=600, ge=1)

    @field_validator("cuda_visible_devices")
    @classmethod
    def devices(cls, value):
        parts = value.split(",")
        if not parts or any(not p.isdigit() for p in parts) or len(set(parts)) != len(parts):
            raise ValueError("Explicit, unique numeric GPU indices are required")
        return value


class OwnedModelServer:
    def __init__(self, config, endpoint, model, log_path, seed):
        self.config, self.endpoint, self.model, self.log_path, self.seed = config, endpoint, model, log_path, seed
        parsed = urlsplit(endpoint)
        if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.scheme != "http" or not parsed.port:
            raise ValueError("A managed server requires an explicit localhost HTTP port")
        if parsed.path.rstrip("/") != "/v1":
            raise ValueError("A managed vLLM endpoint must end in /v1")
        self.port = parsed.port
        self.process = self.log = None

    def environment(self):
        executable = shutil.which(self.config.python)
        if executable is None:
            raise FileNotFoundError(f"Serving Python is not executable: {self.config.python}")
        # A separate vLLM environment also owns compiler helpers such as ninja.
        # Preserve the venv's bin directory rather than resolving its Python
        # symlink back to the system interpreter.
        return {**os.environ, "CUDA_VISIBLE_DEVICES": self.config.cuda_visible_devices,
                "PATH": str(Path(executable).absolute().parent) + os.pathsep + os.environ.get("PATH", "")}

    def __enter__(self):
        with socket.socket() as check:
            try:
                check.bind(("127.0.0.1", self.port))
            except OSError as error:
                raise RuntimeError("Managed port already occupied; existing services will not be stopped") from error
        config = self.config
        command = [config.python, "-m", "vllm.entrypoints.openai.api_server",
            "--host", "127.0.0.1", "--port", str(self.port), "--model", str(config.model_path),
            "--served-model-name", self.model, "--dtype", "bfloat16", "--seed", str(self.seed),
            "--tensor-parallel-size", str(config.tensor_parallel_size),
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--max-model-len", str(config.max_model_len), "--max-num-seqs", str(config.max_num_seqs),
            "--max-num-batched-tokens", str(config.max_num_batched_tokens)]
        if config.enforce_eager:
            command.append("--enforce-eager")
        if config.language_model_only:
            command.append("--language-model-only")
        if config.default_chat_template_kwargs:
            command.extend(["--default-chat-template-kwargs",
                            json.dumps(config.default_chat_template_kwargs, sort_keys=True)])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("ab")
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=self.log,
                stderr=subprocess.STDOUT, env=self.environment(),
                start_new_session=True)
            deadline = time.monotonic() + config.startup_timeout_seconds
            with httpx.Client(timeout=2, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(f"Owned generation server exited; inspect {self.log_path}")
                    try:
                        response = client.get(self.endpoint.rstrip("/") + "/models")
                        if response.status_code == 200:
                            ids = {row["id"] for row in response.json()["data"]}
                            if self.model not in ids:
                                raise RuntimeError("Managed service returned an unexpected model")
                            return self
                    except (httpx.HTTPError, ValueError, KeyError):
                        pass
                    time.sleep(.5)
            raise TimeoutError(f"Owned generation server did not become ready; inspect {self.log_path}")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self.log is not None:
            self.log.close()
