"""OptiX programs and prepared launches in source-independent AOT plans."""

import os
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_optix_program_factory_is_cuda_only():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.native_resource_capabilities("optix_program")
        assert not runtime.probe_native_resource("optix_program")["available"]
        assert not runtime.native_resource_capabilities("optix_prepared_launch")
        assert not runtime.native_operation_capabilities("optix_program_launch")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_program_resources_and_failed_compile_retirement(tmp_path):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit common OptiX adapter required")
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_optix_program_resource_probe.py")),
                mode,
                adapter,
                str(tmp_path / f"{mode}.json"),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_optix_launches_sbt_updates_gradients_and_retirement(tmp_path):
    from tests.python.aot_optix_launch_probe import packet, plan_parts, program_parameters

    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit common OptiX adapter required")

    @ti.kernel
    def convert(source: ti.types.ndarray(ti.u32, 1), values: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in source:
            values[i] = ti.cast(source[i], ti.f32)

    @ti.kernel
    def energy(
        values: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in values:
            loss[0] += 0.5 * values[i] * values[i]

    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(convert)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    code = (Path(__file__).with_name("assets") / "hardware_optix_program.ptx").read_bytes()
    p = program_parameters(code)
    for name, indirect, public in (("trace", False, False), ("indirect", True, False), ("public", False, True)):
        operations, resources = plan_parts(p, indirect, public)
        if not public:
            data = packet(p, indirect)[0]
            resources += [
                dict(kind="host_blob", name="ptx", data=module.add_data(code), byte_size=len(code)),
                dict(kind="host_blob", name="packet", data=module.add_data(data), byte_size=len(data)),
            ]
        module.add_execution_plan(
            name, ti.aot.ExecutionPlan(operations, resources=resources, gradients=dict(values="dvalues", loss="dloss"))
        )
    artifact = tmp_path / "program.tcm"
    module.archive(str(artifact))
    assert len(module._data_payloads) == 3  # Shared PTX plus two address-free packets.
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(compile_ms=(time.perf_counter() - start) * 1000, bytes=artifact.stat().st_size))
    )
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_optix_launch_probe.py")),
                mode,
                str(artifact),
                adapter,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
