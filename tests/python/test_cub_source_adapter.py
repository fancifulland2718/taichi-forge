"""The common provider protocol can lease and execute an existing CUB addon."""

import ctypes as C
import hashlib
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.hardware import _bundled_runtime_provider as bridge
from taichi_forge.hardware._external_cuda_submission import external_cuda_submission
from tests import test_utils


class PlanDesc(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("operation", C.c_uint32), ("num_items", C.c_uint64)]


class PlanInfo(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("reserved", C.c_uint32), ("workspace_bytes", C.c_uint64)]


class ExecDesc(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("reserved", C.c_uint32)] + [
        (name, C.c_uint64)
        for name in ("input0", "input1", "output0", "output1", "workspace", "workspace_bytes", "cuda_stream")
    ]


class ExecutionApi(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("execution_abi_version", C.c_uint32),
        (
            "create_plan",
            C.CFUNCTYPE(
                C.c_uint32, C.c_void_p, C.POINTER(PlanDesc), C.POINTER(C.c_void_p), C.POINTER(PlanInfo)
            ),
        ),
        ("execute", C.CFUNCTYPE(C.c_uint32, C.c_void_p, C.POINTER(ExecDesc))),
        ("destroy_plan", C.CFUNCTYPE(C.c_uint32, C.c_void_p)),
    ]


def load():
    adapter = os.environ.get("TI_CUB_SOURCE_TEST_ADAPTER")
    library = os.environ.get("TI_CUB_SOURCE_TEST_LIBRARY")
    if not adapter or not library:
        pytest.skip("set the CUB common adapter and existing addon paths")
    definition = bridge.BundledRuntimeProviderDefinition(
        "cub_source",
        "CUB source addon",
        "unused",
        "taichi_forge_cub_source_provider_query",
        "forge-runtime-provider-abi2",
        "UNUSED_CUB_TEST",
        (),
        (),
        "source addon C ABI 1",
    )
    return definition, bridge._query_provider(definition, adapter), library


def test_cub_adapter_cold_identity_and_errors(tmp_path):
    definition, loaded, library = load()
    expected = "sha256:" + hashlib.sha256(Path(library).read_bytes()).hexdigest()
    target = tmp_path / "非 ASCII addon" / Path(library).name
    target.parent.mkdir()
    shutil.copy2(library, target)
    info = bridge._probe_runtime(definition, loaded, str(target))
    assert Path(info["library_path"]).samefile(target)
    assert info["build_version"] == expected
    assert info["version_major"] > 0 and info["cuda_runtime_version"] > 0
    for path, message in ((None, "explicit library path"), (str(target) + ".missing", "could not be loaded")):
        with pytest.raises(RuntimeError, match=message):
            bridge._create_runtime(definition, loaded, path)
    with bridge._create_runtime(definition, loaded, library) as runtime:
        with pytest.raises(RuntimeError, match="execution ABI"):
            runtime.query_execution_api(ExecutionApi, version=9)
        api = runtime.query_execution_api(ExecutionApi)
        for operation, n in ((0, 1), (7, 1), (1, 2**31)):
            handle, result = C.c_void_p(123), PlanInfo(C.sizeof(PlanInfo))
            desc = PlanDesc(C.sizeof(PlanDesc), operation, n)
            with pytest.raises(RuntimeError, match="int ABI"):
                runtime.check_result(
                    api.create_plan(runtime.handle, C.byref(desc), C.byref(handle), C.byref(result))
                )
            assert not handle.value


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("empty", (False, True))
def test_cub_adapter_executes_existing_addon_and_preserves_lifetime(empty):
    definition, loaded, library = load()
    program = ti.lang.impl.get_runtime().prog
    n = 0 if empty else 37

    def array(values, dtype=ti.u32):
        result = ti.ndarray(dtype, shape=len(values))
        result.from_numpy(np.asarray(values, dtype=np.uint64 if dtype == ti.u64 else np.uint32))
        return result

    def pointer(value):
        return int(program.get_ndarray_data_ptr_as_int(value.arr))

    with bridge._create_runtime(definition, loaded, library) as runtime:
        api = runtime.query_execution_api(ExecutionApi)
        for operation in ((4, 6) if empty else range(1, 7)):
            desc, result, handle = (
                PlanDesc(C.sizeof(PlanDesc), operation, n),
                PlanInfo(C.sizeof(PlanInfo)),
                C.c_void_p(),
            )
            runtime.check_result(
                api.create_plan(runtime.handle, C.byref(desc), C.byref(handle), C.byref(result))
            )
            try:
                with pytest.raises(RuntimeError, match="live execution resources"):
                    runtime.close()
                workspace = ti.ndarray(ti.u8, shape=max(1, result.workspace_bytes))
                keys = [((i * 7) % 11) + (2**33 if operation == 2 else 0) for i in range(max(1, n))]
                values = list(range(max(1, n)))
                data = array(keys, ti.u64 if operation == 2 else ti.u32)
                auxiliary = array(values if operation < 3 else [i % 2 for i in range(max(1, n))])
                if operation >= 5:
                    auxiliary = array([1 | (1 << 5) | (1 << 31), 0] if n else [1])
                output = array([991] * max(1, n), ti.u64 if operation == 2 else ti.u32)
                second = array([992] * (max(1, n) if operation < 3 else 1))
                invocation = ExecDesc(
                    C.sizeof(ExecDesc),
                    0,
                    pointer(data),
                    pointer(auxiliary),
                    pointer(output),
                    pointer(second),
                    pointer(workspace),
                    result.workspace_bytes,
                    0,
                )
                assert np.all(output.to_numpy() == 991)
                for _ in range(2):
                    with external_cuda_submission(
                        program, (data, auxiliary, output, second, workspace)
                    ) as submission:
                        runtime.check_result(submission.invoke(api.execute, handle, C.byref(invocation)))
                    ti.sync()
                    if empty:
                        assert output.to_numpy()[0] == 991
                        if operation == 4:
                            assert second.to_numpy()[0] == 0
                    elif operation < 3:
                        order = sorted(range(n), key=lambda i: keys[i])
                        assert list(output.to_numpy()) == [keys[i] for i in order]
                        assert list(second.to_numpy()) == order
                    elif operation == 4:
                        chosen = [v for i, v in enumerate(keys) if i % 2]
                        assert second.to_numpy()[0] == len(chosen)
                        assert list(output.to_numpy()[: len(chosen)]) == chosen
                    else:
                        expected, total = [], 0
                        for i, value in enumerate(keys):
                            if operation >= 5 and i in (0, 5, 31):
                                total = 0
                            expected.append(total + value if operation == 5 else total)
                            total += value
                        assert list(output.to_numpy()) == expected
                invocation.input0 = 0
                if empty:
                    # The existing ABI permits empty inputs; empty select still
                    # requires its output count, and empty scan is a no-op.
                    runtime.check_result(api.execute(handle, C.byref(invocation)))
                    ti.sync()
                    if operation == 4:
                        invocation.output1 = 0
                        with pytest.raises(RuntimeError, match="null device pointer"):
                            runtime.check_result(api.execute(handle, C.byref(invocation)))
                else:
                    with pytest.raises(RuntimeError, match="null device pointer"):
                        runtime.check_result(api.execute(handle, C.byref(invocation)))
                    invocation.input0 = pointer(data)
                    with external_cuda_submission(
                        program, (data, auxiliary, output, second, workspace)
                    ) as submission:
                        runtime.check_result(submission.invoke(api.execute, handle, C.byref(invocation)))
                    ti.sync()
            finally:
                runtime.check_result(api.destroy_plan(handle))
