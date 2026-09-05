"""NVML boundary/ABI contracts and lossless public-search report integration."""

import ctypes
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import PureWindowsPath
from types import SimpleNamespace

import numpy as np
import pytest
import taichi_forge as ti

from taichi_forge.hardware import _gpu_environment as env
from tests import test_utils

_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


class _Function:
    def __init__(self, invoke):
        self.invoke = invoke

    def __call__(self, *args):
        assert len(self.argtypes) == len(args)
        assert self.restype is ctypes.c_int
        return self.invoke(*args)


class _Nvml:
    def __init__(self, *, memory_error=0, handle_error=0, init_error=0):
        self.calls = []
        self.init_error = init_error
        self.handle_error = handle_error
        self.memory_error = memory_error
        self.nvmlInit_v2 = self._function("init", lambda: init_error)
        self.nvmlShutdown = self._function("shutdown", lambda: 0)
        self.nvmlDeviceGetHandleByUUID = self._function("handle", self.handle)
        self.nvmlSystemGetDriverVersion = self._function("driver", lambda output, size: self.string(output, b"610.62"))
        self.nvmlSystemGetNVMLVersion = self._function(
            "version", lambda output, size: self.string(output, b"13.610.62")
        )
        self.nvmlDeviceGetName = self._function(
            "name", lambda handle, output, size: self.string(output, b"fixture GPU")
        )
        self.nvmlDeviceGetMemoryInfo = self._function("memory", self.memory)
        self.nvmlDeviceGetClockInfo = self._function(
            "clock", lambda handle, kind, output: self.scalar(output, 1000 + kind)
        )
        self.nvmlDeviceGetPowerUsage = self._function("power", lambda handle, output: self.scalar(output, 120000))
        # NOT_SUPPORTED must remain unavailable, not a made-up zero degrees.
        self.nvmlDeviceGetTemperature = self._function("temperature", lambda handle, kind, output: 3)
        # Exercise the stable older driver symbol and retain unknown reason bits.
        self.nvmlDeviceGetCurrentClocksThrottleReasons = self._function(
            "reasons", lambda handle, output: self.scalar(output, (1 << 63) | 4)
        )

    def _function(self, name, invoke):
        def call(*args):
            self.calls.append(name)
            return invoke(*args)

        return _Function(call)

    @staticmethod
    def string(output, value):
        output.value = value
        return 0

    @staticmethod
    def scalar(output, value):
        output._obj.value = value
        return 0

    def handle(self, value, output):
        assert value == _UUID.encode("ascii")
        output._obj.value = 123 if not self.handle_error else None
        return self.handle_error

    def memory(self, handle, output):
        assert handle.value == 123
        assert ctypes.sizeof(output._obj) == 24
        output._obj.total = 8 << 30
        output._obj.free = 6 << 30
        output._obj.used = env._UNAVAILABLE
        return self.memory_error


def _install(monkeypatch, **kwargs):
    library = _Nvml(**kwargs)
    monkeypatch.setattr(env, "_load_nvml", lambda: (library, "fixture-driver-library", None))
    return library


def test_nvml_resolves_known_driver_paths_without_current_directory_fallback(monkeypatch):
    attempts = []

    def load(reference):
        attempts.append(reference)
        raise OSError("absent")

    monkeypatch.setattr(env.ctypes, "CDLL", load)
    monkeypatch.setattr(env.sys, "platform", "win32")
    monkeypatch.setattr(env, "Path", PureWindowsPath)
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setenv("ProgramW6432", r"C:\Program Files")
    assert env._load_nvml() == (None, None, "driver_library_not_found")
    assert attempts == [r"C:\Windows\System32\nvml.dll", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"]
    attempts.clear()
    monkeypatch.setattr(env.sys, "platform", "linux")
    assert env._load_nvml() == (None, None, "driver_library_not_found")
    assert attempts == ["libnvidia-ml.so.1"]


@pytest.mark.parametrize("memory_error", (0, 3))
def test_nvml_partial_fields_keep_units_sentinels_and_balanced_lifetime(monkeypatch, memory_error):
    library = _install(monkeypatch, memory_error=memory_error)
    report = ti.hardware.gpu_environment(device_uuid=_UUID)
    assert report["status"] == "partial"
    assert report["device_uuid"] == _UUID
    assert report["scope"] == "device_wide_boundary_snapshot"
    assert report["synchronizes_device"] is False
    assert report["values"]["power_milliwatts"] == 120000
    assert report["values"]["memory_clock_mhz"] == 1002
    assert report["values"]["temperature_celsius"] is None
    assert report["errors"]["temperature_celsius"] == "nvml_error:3"
    assert report["values"]["device_memory_used_bytes"] is None
    assert report["values"]["device_memory_total_bytes"] == (None if memory_error else 8 << 30)
    assert report["values"]["clock_event_reasons_mask"] == (1 << 63) | 4
    assert report["clock_event_reasons_symbol"] == "nvmlDeviceGetCurrentClocksThrottleReasons"
    assert library.calls[0:2] == ["init", "handle"]
    assert library.calls.count("shutdown") == 1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", ("missing_driver", "missing_symbol", "init_error", "handle_error"))
def test_nvml_unavailable_does_not_leak_initialization(monkeypatch, failure):
    library = _install(monkeypatch, **({failure: 9} if failure.endswith("error") else {}))
    if failure == "missing_driver":
        monkeypatch.setattr(env, "_load_nvml", lambda: (None, None, "driver_library_not_found"))
    elif failure == "missing_symbol":
        del library.nvmlDeviceGetHandleByUUID
    report = ti.hardware.gpu_environment(device_uuid=_UUID)
    assert report["status"] == "unavailable"
    assert report["errors"]
    assert library.calls.count("shutdown") == (1 if failure == "handle_error" else 0)
    assert "memory" not in library.calls


@test_utils.test(arch=ti.cpu)
def test_nvml_stays_out_of_passive_reports_and_does_not_guess_device_zero(monkeypatch):
    def forbidden():
        raise AssertionError("NVML must not load here")

    monkeypatch.setattr(env, "_load_nvml", forbidden)
    ti.hardware.telemetry()
    assert ti.hardware.gpu_environment()["status"] == "unavailable"
    with ti.hardware.capture_trial_environment():
        assert env._trial_environment.get().sample()["status"] == "unavailable"
    with pytest.raises(ValueError, match="device_uuid"):
        ti.hardware.gpu_environment(device_uuid=0)
    from taichi_forge.lang import impl

    monkeypatch.setattr(impl, "current_cfg", lambda: SimpleNamespace(arch=ti.cuda))
    monkeypatch.setattr(ti.interop, "current_cuda_device_uuid", lambda: uuid.UUID(_UUID[4:]).bytes)
    assert env._device_uuid(None) == (_UUID, None)


def test_nvml_scope_is_nested_thread_local_and_closes_on_exception(monkeypatch):
    library = _install(monkeypatch)
    assert env._trial_environment.get() is None
    with ti.hardware.capture_trial_environment(device_uuid=_UUID):
        outer = env._trial_environment.get()
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(env._trial_environment.get).result() is None
        with pytest.raises(RuntimeError, match="objective"):
            with ti.hardware.capture_trial_environment(device_uuid=_UUID):
                assert env._trial_environment.get() is not outer
                raise RuntimeError("objective")
        assert env._trial_environment.get() is outer
        assert not outer._closed
    assert env._trial_environment.get() is None
    assert outer.sample()["errors"]["session"] == "observation_scope_closed"
    assert library.calls.count("init") == library.calls.count("shutdown") == 2


@test_utils.test(arch=ti.cpu, offline_cache=False)
@pytest.mark.parametrize("objective_fails", (False, True))
def test_trial_environment_survives_public_report_without_changing_objectives_or_reuse(monkeypatch, objective_fails):
    pytest.importorskip("compileiq.forge_support")
    library = _install(monkeypatch)

    @ti.kernel
    def fill(output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in output:
            output[i] = i * 3 - 7

    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1))
    definition = builder.freeze()
    output = ti.ndarray(ti.i32, shape=71)
    target = ti.graph.GraphOptimizationTarget(objectives=(("score", "min"),))
    budget = ti.graph.GraphSearchBudget(evaluation_limit=4)
    contracts = {
        "workload_context": ti.graph.GraphWorkloadContext({"fixture": "nvml-boundary", "count": 71}),
        "evaluation_contract": ti.graph.GraphEvaluationContract({"correctness": "numpy-exact", "score": "constant"}),
        "backend_environment": ti.graph.GraphBackendEnvironment({"backend": "test-cpu"}),
    }

    def evaluate(graph, recipe):
        boundary_count = library.calls.count("memory")
        for _ in range(3):
            graph.run({"output": output})
        np.testing.assert_array_equal(output.to_numpy(), np.arange(71) * 3 - 7)
        assert library.calls.count("memory") == boundary_count, "no sample inside the evaluator or replay"
        if objective_fails:
            raise RuntimeError("injected objective failure")
        return {"score": 1.0}

    plain = definition.search_recipes(target=target, budget=budget, **contracts).run(evaluate)
    assert not library.calls
    with ti.hardware.capture_trial_environment(device_uuid=_UUID):
        observed = definition.search_recipes(target=target, budget=budget, **contracts).run(evaluate)
    assert library.calls.count("init") == library.calls.count("shutdown") == 1
    report = observed.report
    records = report.checkpoint.compileiq_checkpoint["records"]
    actual = [record for record in records if record["source"] == "objective"]
    assert actual
    assert library.calls.count("memory") == 2 * len(actual)
    for record in actual:
        payload = json.loads(record["outcome"]["provenance"][env._PROVENANCE_KEY])
        assert payload["measurement_key"] == record["request"]["measurement_key"]
        assert payload["observation_index"] == record["request"]["observation_index"]
        assert (
            payload["after_cleanup_attempt"]["observed_at_unix_ns"]
            >= payload["before_materialization"]["observed_at_unix_ns"]
        )
        assert payload["before_materialization"]["device_uuid"] == _UUID
        assert (record["outcome"]["failure"] is not None) == objective_fails
    assert report.selected_recipe_id == plain.report.selected_recipe_id
    assert report.checkpoint.contract == plain.report.checkpoint.contract
    assert report.compileiq_report.target == plain.report.compileiq_report.target
    for observed_candidate, plain_candidate in zip(
        report.compileiq_report.candidates, plain.report.compileiq_report.candidates
    ):
        assert observed_candidate.metrics == plain_candidate.metrics
        assert observed_candidate.materialized_physical_ids == plain_candidate.materialized_physical_ids
        assert [item.failure for item in observed_candidate.failures] == [
            item.failure for item in plain_candidate.failures
        ]
    assert any(item["environment_observations"] for item in report.recipe_annotations)
    assert all(
        observation["trial_failed"] == objective_fails
        for item in report.recipe_annotations
        for observation in item["environment_observations"]
    )
    assert "not recipe/process peak memory" in report.to_markdown()
    restored = ti.graph.GraphOptimizationReport.from_json(report.to_json())
    assert restored.to_dict() == report.to_dict()
    assert restored.to_markdown() == report.to_markdown()
    if not objective_fails:
        previous_calls = tuple(library.calls)

        def forbidden(*args):
            raise AssertionError("a completed checkpoint must not re-evaluate")

        resumed = definition.search_recipes(
            target=target, budget=budget, checkpoint=report.checkpoint, **contracts
        ).run(forbidden)
        assert tuple(library.calls) == previous_calls
        assert resumed.report.selected_recipe_id == report.selected_recipe_id
        assert resumed.report.recipe_annotations == report.recipe_annotations
