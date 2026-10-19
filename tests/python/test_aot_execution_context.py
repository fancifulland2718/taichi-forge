"""Load actual AOT kernels in an independent runtime and interleave instances."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Scalar
from tests import test_utils


def export_execution_context_module(tmp_path, artifact_format, explicit_target, tier):
    @ti.func
    def transform(value: ti.types.vector(2, ti.f32)) -> ti.f32:
        return value[0] + value[1] * 2.0 + 1.0

    @ti.real_func
    def carried_value(value: ti.types.vector(2, ti.f32)) -> ti.f32:
        return transform(value)

    @ti.kernel(opt_level=tier)
    def stage(out: ti.types.ndarray(ti.f32, 1), seed: ti.f32, n: ti.i32):
        carried = carried_value(ti.Vector([seed, seed * 2.0]))
        total = 0.0
        for i in range(n):
            total += carried * (i + 1)
        for j in range(n):
            out[j] = total + j

    fields = [ti.field(ti.f32, shape=64) for _ in range(2)]

    @ti.kernel(opt_level=tier)
    def field_stage(out: ti.types.ndarray(ti.f32, 1), seed: ti.f32, n: ti.i32, target: ti.template()):
        carried = carried_value(ti.Vector([seed, seed * 2.0]))
        total = 0.0
        for i in range(n):
            total += carried * (i + 1)
        for j in target:
            target[j] += total + j
        for j in target:
            out[j] = target[j]

    caps = (
        [ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())]
        if explicit_target
        else None
    )
    module = ti.aot.Module(artifact_format=artifact_format, caps=caps)
    module.add_kernel(stage, name="stage")
    module._aot_builder._add_with_launch_abi("gated_stage", module._exported_kernels["stage"], "cuda_graph_gate_v1")
    module._content.append("kernel:gated_stage")
    output_layout = ti.ndarray(ti.f32, 64)
    for i, field in enumerate(fields):
        module.add_field(f"field_{i}", field)
        module.add_kernel(
            field_stage, name=f"field_stage_{i}", template_args={"out": output_layout, "target": field}
        )
    module.add_execution_plan(
        "public_stage",
        ti.aot.ExecutionPlan(
            [dict(kind="kernel", entry="stage", arguments=["temporary", "seed", "n"]),
             dict(kind="copy", source="temporary", destination="out")],
            resources=[dict(kind="ndarray", name="temporary", dtype="f32", shape=[64]),
                       Scalar("i32", 64).resource("n")],
            initialization=[dict(kind="fill32", destination="temporary", value=0)],
        ),
    )
    module.add_execution_plan(
        "public_field",
        ti.aot.ExecutionPlan(
            [dict(kind="kernel", entry="field_stage_0", arguments=["out", "seed", "n"])],
            resources=[Scalar("i32", 64).resource("n")],
        ),
    )
    directory = tmp_path / "module"
    module.save(directory)
    metadata = json.loads((directory / "aot_metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 8
    assert metadata["kernel_format"] == artifact_format
    if not explicit_target:
        assert {item["key"]: item["value"] for item in metadata["required_caps"]} == {
            "cuda_compute_capability": 80,
            "cuda_ptx_version": 70,
        }
        if artifact_format == "ptx":
            assert b".version 7.0" in (directory / "stage.ptx").read_bytes()
    assert {item["key"]: item["value"] for item in metadata["kernel_launch_abis"]} == {
        "stage": "runtime_context_v2",
        "gated_stage": "cuda_graph_gate_context_v2",
        "field_stage_0": "runtime_context_v2",
        "field_stage_1": "runtime_context_v2",
    }
    assert len(metadata["launch_abi_sha256"]) == 64
    assert {item["key"]: item["value"] for item in metadata["kernel_runtime_state"]} == {
        name: 1 for name in ("stage", "gated_stage", "field_stage_0", "field_stage_1")
    }
    assert len(metadata["runtime_state_sha256"]) == 64
    return directory


@pytest.mark.parametrize(
    "artifact_format,explicit_target", (("bc", False), ("ptx", False), ("ptx", True), ("cubin", True))
)
@pytest.mark.parametrize("tier", ("fast", "balanced", "full"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_saved_execution_context_isolates_temporary_storage(artifact_format, explicit_target, tier, tmp_path):
    executable = Path(
        os.environ.get(
            "TI_TEST_CPP_EXECUTABLE",
            str(
                Path(__file__).resolve().parents[2]
                / "build"
                / (
                    "taichi_cuda_execution_domain_tests.exe"
                    if os.name == "nt"
                    else "taichi_cuda_execution_domain_tests"
                )
            ),
        )
    )
    if not executable.is_file():
        pytest.skip("Requires the native CUDA execution-domain test executable")
    if artifact_format == "cubin" and not (os.environ.get("TI_CUDA_PTXAS_PATH") or shutil.which("ptxas")):
        pytest.skip("Requires an external ptxas for cubin export")

    directory = export_execution_context_module(tmp_path, artifact_format, explicit_target, tier)
    result = subprocess.run(
        [str(executable), "--gtest_filter=CudaExecutionContextAot.*:CudaPublicExecutionAot.*:CudaExecutionOwner.ExternalStorageLoanPreservesCompleteAotEpoch"],
        env=dict(os.environ, TAICHI_F3_CONTEXT_AOT_FOLDER=str(directory)),
        capture_output=True,
        text=True,
        timeout=90,
    )
    (tmp_path / "native-consumer.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0 and "[  PASSED  ] 11 tests." in result.stdout, result.stdout + result.stderr
    consumer = Path(__file__).with_name("aot_execution_domain_probe.py")
    result = subprocess.run([sys.executable, "-S", str(consumer), str(directory)],
                            env=os.environ.copy(), capture_output=True, text=True, timeout=90)
    (tmp_path / "public-consumer.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0 and "PUBLIC_AOT_DOMAIN_OK" in result.stdout, result.stdout + result.stderr
    external_consumer = Path(__file__).with_name("aot_external_memory_probe.py")
    result = subprocess.run([sys.executable, "-S", str(external_consumer), str(directory)],
                            env=os.environ.copy(), capture_output=True, text=True, timeout=90)
    (tmp_path / "external-consumer.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0 and "PUBLIC_AOT_EXTERNAL_MEMORY_OK" in result.stdout, result.stdout + result.stderr
    from tests.python.aot_execution_domain_probe import check

    with ti.aot.borrow_runtime() as borrower:
        check(borrower, directory)
