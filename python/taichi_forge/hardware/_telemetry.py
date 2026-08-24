"""Passive hardware admission, execution, queue, and lifecycle telemetry."""

from dataclasses import dataclass
from types import MappingProxyType

from taichi_forge import _hardware_telemetry as _execution
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl


HARDWARE_TELEMETRY_SCHEMA_VERSION = 1


def _frozen_mapping(values):
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class HardwareTelemetryReport:
    """Immutable current-generation snapshot; collecting it never loads a provider."""

    schema_version: int
    runtime_initialized: bool
    runtime_generation: int
    backend: object
    automatic_routes: object
    operations: object
    runtime: object
    resources: object
    providers: object

    def __post_init__(self):
        if self.schema_version != HARDWARE_TELEMETRY_SCHEMA_VERSION:
            raise ValueError("hardware telemetry schema version mismatch")
        for name in (
            "automatic_routes",
            "operations",
            "runtime",
            "resources",
            "providers",
        ):
            object.__setattr__(self, name, _frozen_mapping(getattr(self, name)))

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "runtime_initialized": self.runtime_initialized,
            "runtime_generation": self.runtime_generation,
            "backend": self.backend,
            "automatic_routes": {
                name: dict(values) for name, values in self.automatic_routes.items()
            },
            "operations": {
                name: values.to_dict() for name, values in self.operations.items()
            },
            "runtime": dict(self.runtime),
            "resources": {
                name: dict(values) for name, values in self.resources.items()
            },
            "providers": {
                name: dict(values) for name, values in self.providers.items()
            },
        }


def telemetry():
    """Return passive counters for the active runtime generation.

    Execution failures are counted only after a recording is explicitly run.
    Provider status reads existing loader state and never probes or loads a
    vendor library.
    """

    program = impl.get_runtime().prog
    generation = int(impl.runtime_generation())
    if program is None:
        return HardwareTelemetryReport(
            schema_version=HARDWARE_TELEMETRY_SCHEMA_VERSION,
            runtime_initialized=False,
            runtime_generation=generation,
            backend=None,
            automatic_routes={},
            operations=_execution.execution_snapshot(),
            runtime={},
            resources={},
            providers={},
        )

    backend = _ti_core.arch_name(impl.current_cfg().arch)
    if backend in ("x64", "arm64"):
        backend = "cpu"
    raw = dict(program._runtime_statistics_snapshot())
    submission = dict(raw["submission"])
    synchronization = dict(raw["synchronization"])
    runtime = {
        "program_domain": int(raw["program_domain"]),
        "native_submissions": int(submission["native_submissions"]),
        "graph_submissions": int(submission["graph_submissions"]),
        "backend_graph_launches": int(submission["backend_graph_launches"]),
        "synchronize_count": int(synchronization["program_syncs"]),
        "queue_submit_calls": None,
        "submitted_command_buffers": None,
        "physical_queue_counts_exact": False,
    }
    automatic_routes = {}
    resources = {}
    if backend == "cuda":
        automatic_routes["internal.tile.async.cuda"] = _frozen_mapping(
            program._cuda_async_tile_status()
        )
    if backend == "vulkan":
        queue = dict(program._debug_vulkan_queue_submission_stats())
        if queue["supported"]:
            runtime.update(
                queue_submit_calls=int(queue["queue_submit_calls"]),
                submitted_command_buffers=int(queue["submitted_command_buffers"]),
                physical_queue_counts_exact=True,
            )
        resources["vulkan_graphics_pipeline"] = _frozen_mapping(
            program._debug_vulkan_graphics_resource_stats()
        )
        resources["vulkan_triangle_ray"] = _frozen_mapping(
            program._debug_vulkan_ray_resource_stats()
        )

    providers = {}
    if _ti_core.with_cuda():
        for provider_id in ("cublas", "cusparse", "cufft", "cudss"):
            status = dict(_ti_core.cuda_external_library_status(provider_id))
            provider_facts = {
                "library_loaded": bool(status["library_loaded"]),
                "provider_abi": status["provider_abi"],
                "provider_version": status["provider_version"],
            }
            if provider_id == "cufft" and backend == "cuda":
                cache = dict(program._cuda_cufft_plan_cache_statistics())
                provider_facts.update(
                    {f"plan_{name}": int(value) for name, value in cache.items()}
                )
            providers[provider_id] = _frozen_mapping(provider_facts)

    return HardwareTelemetryReport(
        schema_version=HARDWARE_TELEMETRY_SCHEMA_VERSION,
        runtime_initialized=True,
        runtime_generation=generation,
        backend=backend,
        automatic_routes=automatic_routes,
        operations=_execution.execution_snapshot(),
        runtime=runtime,
        resources=resources,
        providers=providers,
    )


__all__ = [
    "HARDWARE_TELEMETRY_SCHEMA_VERSION",
    "HardwareTelemetryReport",
    "telemetry",
]
