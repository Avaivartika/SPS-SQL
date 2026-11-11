"""Stage-scoped resources on DataDesigner's native CompositeWorkflow."""
from __future__ import annotations

import time
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from data_designer.config.run_config import RunConfig
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.interface.data_designer import DataDesigner

from sps.integration.config import AuxiliaryPredictionConfig, PostCorrectionConfig
from sps.postprocess.inference import AuxiliaryRuntime


class SPSResourceProvider(ResourceProvider):
    auxiliary_runtime: AuxiliaryRuntime | None = None
    post_executor: ProcessPoolExecutor | None = None


class SPSDataDesigner(DataDesigner):
    def __init__(self, *, post_config, post_manifest, schemas, search_run_config,
                 runtime_factory=AuxiliaryRuntime, phase_callback=None, search_service=None, **kwargs):
        super().__init__(**kwargs)
        self.post_config = post_config
        self.post_manifest = post_manifest
        self.schemas = schemas
        self.search_run_config = search_run_config
        self.runtime_factory = runtime_factory
        self.phase_callback = phase_callback or (lambda phase, record: None)
        self.auxiliary_runtime = None
        self.phase_metrics = {}
        self.search_service = search_service
        self.post_executor = None

    def _create_resource_provider(self, *args, **kwargs):
        # DataDesigner 0.9.5's resource construction extension point. Preserve
        # instantiated registries/storage, rather than serializing their objects.
        provider = super()._create_resource_provider(*args, **kwargs)
        return SPSResourceProvider(**{name: getattr(provider, name)
            for name in ResourceProvider.model_fields}, auxiliary_runtime=self.auxiliary_runtime,
            post_executor=self.post_executor)

    def create(self, config_builder, **kwargs):
        columns = config_builder.build().columns
        phase = "auxiliary" if any(isinstance(c, AuxiliaryPredictionConfig) for c in columns) else (
            "post_correction" if any(isinstance(c, PostCorrectionConfig) for c in columns) else "search")
        if phase == "search":
            self.set_run_config(self.search_run_config)
        else:
            self.set_run_config(RunConfig(
                buffer_size=self.post_config.batch_size if phase == "auxiliary" else 1,
                max_concurrent_row_groups=1 if phase == "auxiliary" else self.post_config.cpu_workers,
                max_conversation_restarts=0, max_conversation_correction_steps=0,
                write_scheduler_events=True, otel_metrics_port=None))
        if phase == "auxiliary":
            self.auxiliary_runtime = self.runtime_factory(self.post_config,
                self.post_manifest["model_sha256"], self.schemas)
        if phase == "post_correction":
            # Spawn avoids forking a process that previously initialized CUDA.
            # DataDesigner still schedules and persists each question itself.
            self.post_executor = ProcessPoolExecutor(max_workers=self.post_config.cpu_workers,
                                                      mp_context=get_context("spawn"))
        started = time.monotonic()
        record = {"status": "running"}
        self.phase_callback(phase, record)
        try:
            with self.search_service if phase == "search" and self.search_service is not None else nullcontext():
                result = super().create(config_builder, **kwargs)
            record["status"] = "complete"
            return result
        except BaseException as error:
            record.update(status="interrupted", error_type=type(error).__name__, error=str(error)[:2000])
            raise
        finally:
            if self.auxiliary_runtime is not None:
                record["model_load_seconds"] = self.auxiliary_runtime.load_seconds
                self.auxiliary_runtime.close()
                self.auxiliary_runtime = None
            if self.post_executor is not None:
                self.post_executor.shutdown(wait=True, cancel_futures=True)
                self.post_executor = None
            record["elapsed_seconds"] = time.monotonic() - started
            self.phase_metrics[phase] = record
            self.phase_callback(phase, record)
