"""AOT retains compiler state facts without changing its task/context ABI."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("artifact_format", ("bc", "ptx", "cubin"))
@pytest.mark.parametrize("dense_lists", (False, True))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_shared_runtime_state_is_not_admitted(artifact_format, dense_lists, tmp_path):
    executable = Path(os.environ.get("TI_TEST_CPP_EXECUTABLE", "build/taichi_cuda_execution_domain_tests.exe"))
    if not executable.is_file():
        pytest.skip("Requires the native execution-domain test executable")
    if artifact_format == "cubin" and not (os.environ.get("TI_CUDA_PTXAS_PATH") or shutil.which("ptxas")):
        pytest.skip("Requires an external ptxas for cubin export")
    ti.init(arch=ti.cuda, offline_cache=False, demote_dense_struct_fors=not dense_lists)

    @ti.real_func
    def random_number() -> ti.f32:
        return ti.random(ti.f32)

    @ti.kernel
    def random_direct(out: ti.types.ndarray(ti.f32, 1)):
        out[0] = ti.random(ti.f32)

    @ti.kernel
    def random_callee(out: ti.types.ndarray(ti.f32, 1)):
        out[0] = random_number()

    @ti.kernel
    def print_value(out: ti.types.ndarray(ti.f32, 1)):
        print(out[0])

    field = ti.field(ti.i32)
    ti.root.pointer(ti.i, 4).dense(ti.i, 4).place(field)

    @ti.kernel
    def sparse_value():
        field[0] = 1

    dense = ti.field(ti.i32, shape=8)

    @ti.kernel
    def dense_value():
        for i in dense:
            dense[i] += 1

    caps = [ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())]
    module = ti.aot.Module(artifact_format=artifact_format, caps=caps)
    for kernel in (random_direct, random_callee, print_value, sparse_value, dense_value):
        module.add_kernel(kernel)
    directory = tmp_path / "module"
    module.save(directory)
    metadata = json.loads((directory / "aot_metadata.json").read_text())
    assert metadata["schema_version"] == 17
    assert {item["key"]: item["value"] for item in metadata["kernel_runtime_state"]} == {
        "random_direct": 3, "random_callee": 3, "print_value": 65, "sparse_value": 9,
        "dense_value": 9 if dense_lists else 1,
    }
    result = subprocess.run(
        [str(executable), "--gtest_filter=CudaRuntimeStateAot.*"],
        env=dict(os.environ, TAICHI_F3_RUNTIME_STATE_AOT_FOLDER=str(directory),
                 TAICHI_F3_DENSE_LIST_MODE=str(int(dense_lists))),
        capture_output=True, text=True, timeout=90,
    )
    (tmp_path / "native-consumer.log").write_text(result.stdout + result.stderr, encoding="utf8")
    assert result.returncode == 0 and "[  PASSED  ] 1 test." in result.stdout, result.stdout + result.stderr

    from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime

    # The public plan entry must reject the entire recipe before a preceding
    # fill can run or private resources/initialization can consume a sequence.
    with Runtime("cuda") as runtime:
        loaded = runtime.load_module(directory)
        out = runtime.ndarray("f32", 1)
        out.write(b"\x00\x00\x80\x3f")
        domain = runtime.execution_domain()
        blocked = {
            "random_direct": (["out"], "shared random-number state"),
            "random_callee": (["out"], "shared random-number state"),
            "print_value": (["out"], "shared device print state"),
            "sparse_value": ([], "shared sparse list or allocator state"),
        }
        if dense_lists:
            blocked["dense_value"] = ([], "shared sparse list or allocator state")
        for entry, (arguments, reason) in blocked.items():
            plan = ExecutionPlan(
                [dict(kind="fill32", destination="private", value=0),
                 dict(kind="copy", source="private", destination="out", size_bytes=4),
                 dict(kind="kernel", entry=entry, arguments=arguments)],
                resources=[dict(kind="ndarray", name="private", dtype="f32", shape=[16], zero_initialize=True)],
            )
            with pytest.raises(AotError, match=reason):
                loaded.prepare_execution(plan, dict(out=out), execution_domain=domain)
            with domain.latest_completion() as completion:
                assert completion.info["sequence"] == 0
            assert domain.info["accepting"]
            assert out.read() == b"\x00\x00\x80\x3f"
        with Runtime("cuda") as foreign:
            wrong_domain = foreign.execution_domain()
            plan = ExecutionPlan([dict(kind="fill32", destination="out", value=0)])
            with pytest.raises(RuntimeError, match="another runtime generation"):
                loaded.prepare_execution(plan, dict(out=out), execution_domain=wrong_domain)
            assert wrong_domain.info["accepting"]

    # Runtime-state facts are independent from the original launch checksum.
    # Corrupting only this contract must fail before loading a kernel.

    metadata["kernel_runtime_state"][0]["value"] ^= 2
    (directory / "aot_metadata.json").write_text(json.dumps(metadata), encoding="utf8")
    with Runtime("cuda") as runtime:
        with pytest.raises(RuntimeError, match="runtime-state inventory checksum"):
            runtime.load_module(directory)
