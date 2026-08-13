import json
import tempfile
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge import runtime as runtime_api
from tests import test_utils


def test_runtime_contract_manifest_is_immutable_and_source_agnostic():
    manifest = ti.validate_runtime_contract(require_native_manifest=True)
    assert manifest["schema_version"] == 1
    assert manifest["native_abi_revision"] == manifest["required_native_abi_revision"]
    assert manifest["schemas"]["dynamic_work"] == 5
    assert manifest["schemas"]["structured_control"] == 5
    assert manifest["schemas"]["graph_pipeline"] == 2
    assert manifest["runtime"]["source_id"]
    assert manifest["shim"]["source_id"]
    assert manifest["compiler_compatibility"]["runtime"]
    assert manifest["compiler_compatibility"]["shim"]
    assert manifest["features"]["cpu"] is True
    with pytest.raises(TypeError):
        manifest["native_abi_revision"] = 0
    with pytest.raises(TypeError):
        manifest["schemas"]["dynamic_work"] = 0


def test_runtime_contract_manifest_is_safe_before_init():
    ti.reset()
    manifest = ti.validate_runtime_contract(require_native_manifest=True)
    assert manifest["schemas"]["dynamic_work"] == 5
    assert ti.lang.impl.get_runtime().prog is None


def test_runtime_public_api_requires_an_initialized_program():
    ti.reset()
    with pytest.raises(ti.TaichiRuntimeError, match=r"requires ti\.init"):
        ti.runtime.stats()
    with pytest.raises(ti.TaichiRuntimeError, match=r"requires ti\.init"):
        ti.runtime.capabilities()
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(ti.TaichiRuntimeError, match=r"requires ti\.init"):
            with ti.runtime.trace(Path(directory) / "missing.json"):
                pass


def test_runtime_public_api_rejects_mismatched_statistics_schema():
    with pytest.raises(
        ti.TaichiRuntimeError,
        match=r"unsupported runtime statistics schema 1.*matching",
    ):
        runtime_api._statistics_from_raw({"schema_version": 1})


def test_startup_profile_is_opt_in_and_available_before_program_init():
    ti.reset()
    ti.runtime.configure_startup_profile(True, clear=True)
    before = ti.runtime.startup_profile()
    assert before.enabled
    assert before.events == ()

    ti.init(arch=ti.cpu, startup_profile=True)
    snapshot = ti.runtime.startup_profile()
    phase_names = {phase.name for phase in snapshot.phases}
    assert snapshot.schema_version == 1
    assert "ti_init.total" in phase_names
    assert "ti_init.reset" in phase_names
    assert "ti_init.program_create" in phase_names
    assert "ti_init.runtime_materialize" in phase_names
    assert all(phase.duration_ns >= 0 for phase in snapshot.phases)
    ti.reset()
    ti.runtime.configure_startup_profile(False, clear=True)
    disabled = ti.runtime.startup_profile()
    assert not disabled.enabled
    assert disabled.events == ()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_runtime_stats_and_capabilities_are_immutable_and_exact():
    @ti.kernel
    def step(value: ti.types.ndarray()):
        value[0] += 1

    value = ti.ndarray(ti.i32, shape=1)
    step(value)
    ti.sync()

    prog = ti.lang.impl.get_runtime().prog
    raw = prog._runtime_statistics_snapshot()
    snapshot = ti.runtime.stats()
    assert snapshot.schema_version == raw["schema_version"] == 3
    assert snapshot.backend == raw["backend"]
    assert snapshot.program_domain == raw["program_domain"]
    assert asdict(snapshot.submission) == raw["submission"]
    assert asdict(snapshot.synchronization) == raw["synchronization"]
    assert asdict(snapshot.memory) == raw["memory"]
    assert asdict(snapshot.transfer) == raw["transfer"]
    assert asdict(snapshot.graph) == raw["graph"]
    assert asdict(snapshot.display) == raw["display"]
    assert asdict(snapshot.trace) == raw["trace"]
    assert asdict(snapshot.fault) == raw["fault"]
    with pytest.raises(FrozenInstanceError):
        snapshot.program_domain = 0

    capabilities = ti.runtime.capabilities()
    assert capabilities.schema_version == 1
    assert capabilities.backend == snapshot.backend
    assert capabilities.program_domain == snapshot.program_domain
    assert capabilities.statistics is True
    assert capabilities.statistics_schema_version == snapshot.schema_version
    assert capabilities.bounded_trace is True
    assert capabilities.trace_schema_version == 1
    assert capabilities.chrome_trace_export is True
    assert capabilities.backend_wait_telemetry is (raw["synchronization"]["backend_waits"] is not None)
    assert capabilities.backend_lock_telemetry is (raw["synchronization"]["backend_lock_samples"] is not None)
    assert capabilities.device_memory_telemetry is (raw["memory"]["device_raw_bytes"] is not None)
    assert capabilities.cuda_mempool_telemetry is (raw["memory"]["cuda_mempool_reserved_bytes"] is not None)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_runtime_trace_context_exports_and_accounts_exactly():
    @ti.kernel
    def increment(values: ti.types.ndarray()):
        for i in values:
            values[i] += 1

    values = ti.ndarray(ti.i32, shape=4)
    increment(values)
    ti.sync()
    before = ti.runtime.stats()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "runtime.json"
        context = ti.runtime.trace(path, max_threads=2, events_per_thread=32)
        with context as active:
            assert active is context
            assert active.started.enabled is True
            assert active.summary is None
            values.from_numpy(np.arange(4, dtype=np.int32))
            increment(values)
            result = values.to_numpy()
            ti.sync()
        assert context.exported is True
        assert context.summary.enabled is False
        exported = json.loads(path.read_text(encoding="utf-8"))

    np.testing.assert_array_equal(result, np.arange(4, dtype=np.int32) + 1)
    assert exported["taichiRuntimeTrace"]["programDomain"] == before.program_domain
    assert exported["taichiRuntimeTrace"]["recordedEvents"] == context.summary.recorded_events
    assert len(exported["traceEvents"]) == context.summary.recorded_events
    after = ti.runtime.stats()
    assert after.trace.recorded_events - before.trace.recorded_events == context.summary.recorded_events
    assert after.trace.dropped_events - before.trace.dropped_events == context.summary.dropped_events
    with pytest.raises(ti.TaichiRuntimeError, match="one-shot"):
        with context:
            pass


@test_utils.test(arch=ti.cpu)
def test_runtime_trace_rejects_nesting_and_preserves_workload_exception():
    @ti.kernel
    def step(value: ti.types.ndarray()):
        value[0] += 1

    value = ti.ndarray(ti.i32, shape=1)
    step(value)
    ti.sync()

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        outer_path = directory / "outer.json"
        with ti.runtime.trace(outer_path) as outer:
            with pytest.raises(ti.TaichiRuntimeError, match="nested or concurrent"):
                with ti.runtime.trace(directory / "inner.json"):
                    pass
            step(value)
        assert outer.exported is True

        error_path = directory / "error.json"
        error_context = ti.runtime.trace(error_path)
        with pytest.raises(ValueError, match="workload failed"):
            with error_context:
                step(value)
                raise ValueError("workload failed")
        assert error_context.exported is True
        assert error_context.summary.recorded_events >= 1
        assert error_path.exists()


@test_utils.test(arch=ti.cpu)
def test_runtime_trace_cleanup_failure_does_not_leak_active_context():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        invalid_path = directory / "missing" / "trace.json"
        with pytest.raises(ti.TaichiRuntimeError, match="unable to export"):
            with ti.runtime.trace(invalid_path):
                pass

        valid_path = directory / "valid.json"
        with ti.runtime.trace(valid_path) as recovered:
            pass
        assert recovered.exported is True
        assert valid_path.exists()


def test_runtime_stats_generation_and_trace_survive_reset_boundary():
    ti.init(arch=ti.cpu)

    @ti.kernel
    def step(value: ti.types.ndarray()):
        value[0] += 1

    value = ti.ndarray(ti.i32, shape=1)
    step(value)
    ti.sync()
    old_stats = ti.runtime.stats()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reset.json"
        with ti.runtime.trace(path) as old_trace:
            step(value)
            ti.reset()
        assert old_trace.exported is True
        assert old_trace.summary.program_domain == old_stats.program_domain
        assert path.exists()

    ti.init(arch=ti.cpu)
    new_stats = ti.runtime.stats()
    assert new_stats.program_domain != old_stats.program_domain
    assert old_stats.program_domain == old_trace.summary.program_domain
    ti.reset()
