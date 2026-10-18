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
    assert metadata["schema_version"] == 8
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

    # Runtime-state facts are independent from the original launch checksum.
    # Corrupting only this contract must fail before loading a kernel.
    from taichi_forge_runtime.aot import Runtime

    metadata["kernel_runtime_state"][0]["value"] ^= 2
    (directory / "aot_metadata.json").write_text(json.dumps(metadata), encoding="utf8")
    with Runtime("cuda") as runtime:
        with pytest.raises(RuntimeError, match="runtime-state inventory checksum"):
            runtime.load_module(directory)
