import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
import taichi_forge._hardware_telemetry as execution_telemetry
from taichi_forge._hardware_telemetry import (
    hardware_failure_phase,
    hardware_provider_call,
    instrument_hardware_recording,
)
from taichi_forge.hardware._external_providers import (
    external_provider_ids,
    passive_external_provider_status,
)
from tests import test_utils


@instrument_hardware_recording("test.hardware_failure_phases")
class _FailurePhaseRecording:
    command_count = 1

    def __init__(self, phase):
        self.phase = phase

    def execute(self):
        if self.phase is None:
            raise RuntimeError("unattributed validation failure")
        with hardware_failure_phase(self.phase):
            raise RuntimeError(f"injected {self.phase}")


@instrument_hardware_recording("test.hardware_provider_call")
class _ProviderCallFailureRecording:
    command_count = 1

    def execute(self):
        with hardware_provider_call("test-provider"):
            raise RuntimeError("injected provider failure")


@test_utils.test(arch=ti.cpu)
def test_hardware_telemetry_is_passive_and_classifies_explicit_failure():
    provider_loaded_before = {
        name: bool(passive_external_provider_status(name)["library_loaded"])
        for name in external_provider_ids()
    }
    before = ti.hardware.telemetry()
    assert before.schema_version == ti.hardware.HARDWARE_TELEMETRY_SCHEMA_VERSION
    assert before.runtime_initialized
    assert before.backend == "cpu"
    assert before.runtime["native_submissions"] == 0
    assert before.resources == {}
    assert "matrix.mma.cuda" in before.operations
    if ti_core.with_cuda():
        assert tuple(before.providers) == external_provider_ids()

    recording = ti.hardware.matrix.CudaMatrixMmaRecording(1)
    memory = recording.memory_report()
    assert not memory.components[0].resident
    assert memory.known_resident_requested_bytes == 0
    with pytest.raises(RuntimeError, match="requires the CUDA backend"):
        recording.execute({"a": object(), "b": object(), "output": object()})

    after = ti.hardware.telemetry()
    operation = after.operations["matrix.mma.cuda"]
    assert operation.recordings == before.operations["matrix.mma.cuda"].recordings + 1
    assert operation.attempted == before.operations["matrix.mma.cuda"].attempted + 1
    assert operation.executed == before.operations["matrix.mma.cuda"].executed
    assert operation.unsupported == before.operations["matrix.mma.cuda"].unsupported + 1
    assert (
        operation.contract_failure
        == before.operations["matrix.mma.cuda"].contract_failure
    )
    assert operation.provider_load_failure == 0
    assert operation.provider_plan_failure == 0
    assert operation.provider_execution_failure == 0
    assert operation.completion_failure == 0
    assert operation.fallback == 0
    assert operation.declared_backend_commands >= 1
    assert not recording.memory_report().components[0].resident

    provider_loaded_after = {
        name: bool(passive_external_provider_status(name)["library_loaded"])
        for name in external_provider_ids()
    }
    assert provider_loaded_after == provider_loaded_before


@test_utils.test(arch=ti.cpu)
def test_hardware_telemetry_uses_explicit_failure_phases():
    before = ti.hardware.telemetry().operations["test.hardware_failure_phases"]
    phases = (
        "provider_load_failure",
        "provider_plan_failure",
        "provider_execution_failure",
        "completion_failure",
    )
    for phase in phases:
        with pytest.raises(RuntimeError, match=phase):
            _FailurePhaseRecording(phase).execute()
    with pytest.raises(RuntimeError, match="validation failure"):
        _FailurePhaseRecording(None).execute()

    after = ti.hardware.telemetry().operations["test.hardware_failure_phases"]
    assert after.recordings == before.recordings + len(phases) + 1
    assert after.attempted == before.attempted + len(phases) + 1
    assert after.executed == before.executed
    assert after.contract_failure == before.contract_failure + 1
    for phase in phases:
        assert getattr(after, phase) == getattr(before, phase) + 1


@test_utils.test(arch=ti.cpu)
def test_hardware_telemetry_classifies_lazy_provider_load_without_error_text(
    monkeypatch,
):
    states = iter((False, False, True, True))
    monkeypatch.setattr(
        execution_telemetry,
        "_provider_library_loaded",
        lambda provider_id: next(states),
    )
    before = ti.hardware.telemetry().operations["test.hardware_provider_call"]

    with pytest.raises(RuntimeError, match="provider failure"):
        _ProviderCallFailureRecording().execute()
    with pytest.raises(RuntimeError, match="provider failure"):
        _ProviderCallFailureRecording().execute()

    after = ti.hardware.telemetry().operations["test.hardware_provider_call"]
    assert after.provider_load_failure == before.provider_load_failure + 1
    assert after.provider_execution_failure == before.provider_execution_failure + 1


@test_utils.test(arch=ti.cpu)
def test_hardware_memory_component_defaults_to_nonresident():
    component = ti.hardware.HardwareMemoryComponent(
        "opaque", None, False, "runtime", "driver"
    )
    assert not component.resident
    with pytest.raises(TypeError, match="resident"):
        ti.hardware.HardwareMemoryComponent(
            "opaque", None, False, "runtime", "driver", resident=1
        )
