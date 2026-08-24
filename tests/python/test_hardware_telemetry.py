import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_hardware_telemetry_is_passive_and_classifies_explicit_failure():
    provider_loaded_before = {
        name: bool(ti_core.cuda_external_library_status(name)["library_loaded"])
        for name in ("cublas", "cusparse", "cufft", "cudss")
    }
    before = ti.hardware.telemetry()
    assert before.schema_version == ti.hardware.HARDWARE_TELEMETRY_SCHEMA_VERSION
    assert before.runtime_initialized
    assert before.backend == "cpu"
    assert before.runtime["native_submissions"] == 0
    assert before.resources == {}
    assert "matrix.mma.cuda" in before.operations

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
    assert operation.incompatible == before.operations["matrix.mma.cuda"].incompatible
    assert operation.fallback == 0
    assert operation.declared_backend_commands >= 1
    assert not recording.memory_report().components[0].resident

    provider_loaded_after = {
        name: bool(ti_core.cuda_external_library_status(name)["library_loaded"])
        for name in ("cublas", "cusparse", "cufft", "cudss")
    }
    assert provider_loaded_after == provider_loaded_before


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
