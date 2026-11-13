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


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("cleanup", [False, True])
def test_bitcode_library_shares_real_function_and_validates_dependencies(tmp_path, cleanup):
    import numpy as np
    from zipfile import ZipFile, ZIP_DEFLATED
    from taichi_forge_runtime.aot import Runtime

    @ti.real_func
    def shared(value: ti.f32, bias: ti.types.ndarray(ti.f32, 0)) -> ti.f32:
        x = value
        for _ in range(5):
            x = ti.sin(x) * 0.25 + bias[None]
        return x

    @ti.kernel
    def first(data: ti.types.ndarray(ti.f32, 1), bias: ti.types.ndarray(ti.f32, 0)):
        for i in range(data.shape[0]):
            data[i] = shared(shared(data[i], bias), bias)

    @ti.kernel
    def second(data: ti.types.ndarray(ti.f32, 1), bias: ti.types.ndarray(ti.f32, 0)):
        for i in range(data.shape[0]):
            data[i] = shared(data[i], bias) + 1

    module = ti.aot.Module(artifact_format="bc-library")
    if cleanup:
        module._aot_builder._set_ir_cleanup_for_testing(True)
    module.add_kernel(first)
    module.add_kernel(second)
    artifact = tmp_path / "library.tcm"
    module.archive(str(artifact))
    if cleanup:
        stats = module._aot_builder._streaming_export_statistics()
        assert stats['aot_cleaned_units'] == 1
        assert stats['aot_cleaned_roots'] == 2
        assert stats['aot_unit_finalization_hits'] == 1
        # A second save reuses finalized bodies and recomputes no Pass chain.
        module.archive(str(tmp_path / 'again.tcm'))
        again = module._aot_builder._streaming_export_statistics()
        assert again['aot_cleaned_units'] == 1
        assert again['aot_cleaned_roots'] == 2
        assert again['aot_root_finalization_hits'] == 2
    with ZipFile(artifact) as archive:
        units = [name for name in archive.namelist() if name.startswith("code_units/")]
        assert len(units) == 1  # Actual shared storage, not two equal copies.

    with Runtime("cuda") as runtime:
        loaded = runtime.load_module(str(artifact))
        a = runtime.ndarray("f32", (37,))
        b = runtime.ndarray("f32", (37,))
        bias = runtime.ndarray("f32", ())
        bias_value = np.array(0.5, dtype=np.float32)
        bias.write(bias_value.tobytes())
        for name, reference in (("first", first), ("second", second)):
            source = np.linspace(0, 1, 37, dtype=np.float32)
            expected = source.copy()
            reference(expected, bias_value)
            peer_expected = source + 7
            reference(peer_expected, bias_value)
            a.write(source.tobytes())
            b.write((source + 7).tobytes())
            with loaded.kernel(name).bind(a, bias).prepare(specialize=True) as prepared:
                with loaded.kernel(name).bind(b, bias).prepare(specialize=True) as peer:
                    prepared.run()
                    runtime.wait()
                    np.testing.assert_array_equal(np.frombuffer(b.read(), np.float32), source + 7)
                    peer.run()
                    runtime.wait()
                    np.testing.assert_allclose(np.frombuffer(a.read(), np.float32), expected, rtol=1e-6, atol=1e-6)
                    np.testing.assert_allclose(np.frombuffer(b.read(), np.float32), peer_expected, rtol=1e-6, atol=1e-6)
        a.close()
        b.close()
        bias.close()

    for failure, message in (("missing", "Missing AOT code unit"), ("corrupt", "checksum mismatch")):
        broken = tmp_path / (failure + ".tcm")
        with ZipFile(artifact) as original, ZipFile(broken, "w", ZIP_DEFLATED) as output:
            for name in original.namelist():
                if name == units[0] and failure == "missing":
                    continue
                contents = original.read(name)
                if name == units[0]:
                    contents += b"corrupted"
                output.writestr(name, contents)
        with Runtime("cuda") as runtime:
            with pytest.raises(RuntimeError, match=message):
                runtime.load_module(str(broken)).kernel("first")


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
    assert metadata["schema_version"] == 22
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


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_individual_instance_retirement_before_pending_peer(tmp_path):
    executable = Path(os.environ.get(
        "TI_TEST_CPP_EXECUTABLE",
        str(Path(__file__).resolve().parents[2] / "build" / "taichi_cuda_execution_domain_tests.exe"),
    ))
    if not executable.is_file():
        pytest.skip("Requires the native CUDA execution-domain test executable")
    directory = export_execution_context_module(tmp_path, "bc", False, "fast")
    result = subprocess.run(
        [str(executable), "--gtest_filter=CudaExecutionInstanceRetirement.*:CudaPublicInstanceRetirement.*"],
        env=dict(os.environ, TAICHI_F3_CONTEXT_AOT_FOLDER=str(directory)),
        capture_output=True, text=True, timeout=30,
    )
    (tmp_path / "native-retirement.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode == 0 and "[  SKIPPED ]" in result.stdout:
        pytest.skip("Native instance retirement requires stream-ordered allocation support")
    assert result.returncode == 0 and "[  PASSED  ] 2 tests." in result.stdout, result.stdout + result.stderr


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
