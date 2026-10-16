"""Lease an existing CUTLASS addon through the common provider protocol."""

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
    _fields_ = (
        [(name, C.c_uint32) for name in ("struct_size", "strategy", "transpose_a", "transpose_b")]
        + [(name, C.c_int32) for name in ("m", "n", "k", "activation")]
        + [(name, C.c_float) for name in ("alpha", "beta")]
    )


class PlanInfo(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("reserved", C.c_uint32), ("workspace_bytes", C.c_uint64)]


class ExecDesc(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("reserved", C.c_uint32)] + [
        (name, C.c_uint64) for name in ("a", "b", "output", "workspace", "workspace_bytes", "cuda_stream")
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
    adapter = os.environ.get("TI_CUTLASS_SOURCE_TEST_ADAPTER")
    library = os.environ.get("TI_CUTLASS_SOURCE_TEST_LIBRARY")
    if not adapter or not library:
        pytest.skip("set the CUTLASS common adapter and existing addon paths")
    definition = bridge.BundledRuntimeProviderDefinition(
        "cutlass_source",
        "CUTLASS source addon",
        "unused",
        "taichi_forge_cutlass_source_provider_query",
        "forge-runtime-provider-abi2",
        "UNUSED_CUTLASS_TEST",
        (),
        (),
        "source addon C ABI 1",
    )
    return definition, bridge._query_provider(definition, adapter), library


def test_cutlass_adapter_cold_identity_and_errors(tmp_path):
    definition, loaded, library = load()
    target = tmp_path / "非 ASCII addon" / Path(library).name
    target.parent.mkdir()
    shutil.copy2(library, target)
    info = bridge._probe_runtime(definition, loaded, str(target))
    assert Path(info["library_path"]).samefile(target)
    assert info["build_version"] == "sha256:" + hashlib.sha256(Path(library).read_bytes()).hexdigest()
    assert info["version_major"] > 0
    assert info["cuda_runtime_version"] == 0  # Not queryable through the original ABI.
    for path, message in ((None, "explicit library path"), (str(target) + ".missing", "could not be loaded")):
        with pytest.raises(RuntimeError, match=message):
            bridge._create_runtime(definition, loaded, path)
    with bridge._create_runtime(definition, loaded, library) as runtime:
        with pytest.raises(RuntimeError, match="execution ABI"):
            runtime.query_execution_api(ExecutionApi, version=9)
        api = runtime.query_execution_api(ExecutionApi)
        for key, value in (
            ("strategy", 3),
            ("m", 0),
            ("n", -1),
            ("transpose_a", 2),
            ("activation", 2),
            ("alpha", float("nan")),
            ("beta", float("inf")),
        ):
            desc = PlanDesc(C.sizeof(PlanDesc), 0, 0, 0, 17, 19, 129, 0, 0.75, 0.25)
            setattr(desc, key, value)
            handle, info = C.c_void_p(123), PlanInfo(C.sizeof(PlanInfo))
            with pytest.raises(RuntimeError, match="plan descriptor"):
                runtime.check_result(
                    api.create_plan(runtime.handle, C.byref(desc), C.byref(handle), C.byref(info))
                )
            assert not handle.value
        desc = PlanDesc(C.sizeof(PlanDesc), 2, 0, 0, 2**31 - 1, 2**31 - 1, 1, 0, 1, 0)
        with pytest.raises(RuntimeError, match="workspace exceeds platform ABI"):
            runtime.check_result(api.create_plan(runtime.handle, C.byref(desc), C.byref(handle), C.byref(info)))
        assert not handle.value


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("transpose,activation", [((0, 0), 0), ((1, 1), 1), ((0, 1), 1), ((1, 0), 0)])
def test_cutlass_adapter_strategies_feedback_and_lifetime(transpose, activation):
    definition, loaded, library = load()
    program = ti.lang.impl.get_runtime().prog
    m, n, k = 17, 19, 129
    rng = np.random.default_rng(1016)
    host = [
        rng.uniform(-0.5, 0.5, shape).astype(np.float32)
        for shape in ((k, m) if transpose[0] else (m, k), (n, k) if transpose[1] else (k, n), (m, n))
    ]
    arrays = [ti.ndarray(ti.f32, shape=x.shape) for x in host]
    for value, data in zip(arrays, host):
        value.from_numpy(data)

    def pointer(value):
        return int(program.get_ndarray_data_ptr_as_int(value.arr))

    product = 0.75 * (
        (host[0].T if transpose[0] else host[0]).astype(np.float64)
        @ (host[1].T if transpose[1] else host[1]).astype(np.float64)
    )
    with bridge._create_runtime(definition, loaded, library) as runtime:
        api = runtime.query_execution_api(ExecutionApi)
        for strategy in range(3):
            desc = PlanDesc(C.sizeof(PlanDesc), strategy, *transpose, m, n, k, activation, 0.75, 0.25)
            handle, info = C.c_void_p(), PlanInfo(C.sizeof(PlanInfo))
            runtime.check_result(api.create_plan(runtime.handle, C.byref(desc), C.byref(handle), C.byref(info)))
            try:
                with pytest.raises(RuntimeError, match="live execution resources"):
                    runtime.close()
                assert info.workspace_bytes == m * n * 4 * (0, 16, 128)[strategy]
                workspace = ti.ndarray(ti.u8, shape=max(1, info.workspace_bytes))
                arrays[2].from_numpy(host[2])
                np.testing.assert_array_equal(arrays[2].to_numpy(), host[2])
                invocation = ExecDesc(
                    C.sizeof(ExecDesc), 0, *map(pointer, arrays), pointer(workspace), info.workspace_bytes, 0
                )
                invocation.workspace_bytes += 1
                with pytest.raises(RuntimeError, match="workspace differs"):
                    runtime.check_result(api.execute(handle, C.byref(invocation)))
                invocation.workspace_bytes -= 1
                saved_a = invocation.a
                invocation.a = 0
                with pytest.raises(RuntimeError, match="capture pointers"):
                    runtime.check_result(api.execute(handle, C.byref(invocation)))
                invocation.a = saved_a
                expected = host[2].astype(np.float64)
                for _ in range(3):
                    with external_cuda_submission(program, (*arrays, workspace)) as submission:
                        runtime.check_result(submission.invoke(api.execute, handle, C.byref(invocation)))
                    ti.sync()
                    expected = product + 0.25 * expected
                    if activation:
                        expected = np.maximum(expected, 0)
                    np.testing.assert_allclose(arrays[2].to_numpy(), expected, rtol=5e-5, atol=5e-5)
            finally:
                runtime.check_result(api.destroy_plan(handle))
