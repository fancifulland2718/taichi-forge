"""Optional, explicitly requested native Vulkan CiG qualification."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(not os.environ.get("TI_TEST_CIG_FIXTURE"), reason="Requires native CiG Vulkan fixture")
@pytest.mark.parametrize("scenario", ["invalid", "auto", "disabled", "foreign"])
def test_fresh_process_context_contract(scenario):
    probe = Path(__file__).with_name("cuda_cig_contract_probe.py")
    result = subprocess.run(
        [sys.executable, "-S", str(probe), scenario], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not os.environ.get("TI_TEST_CIG_FIXTURE"), reason="Requires native CiG Vulkan fixture")
@pytest.mark.parametrize("stage", ["initialize", "submit"])
def test_context_fault_has_bounded_failure(stage):
    probe = Path(__file__).with_name("cuda_cig_fault_probe.py")
    result = subprocess.run(
        [sys.executable, "-S", str(probe), stage], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not os.environ.get("TI_TEST_CIG"), reason="Requires configured CiG context")
def test_jit_field_graph_and_worker_restore_foreign_context():
    import ctypes as C
    import threading
    import taichi_forge as ti
    from taichi_forge_runtime.aot import cuda_context_info

    assert cuda_context_info()["context_mode"] == "vulkan_cig"
    driver = C.WinDLL("nvcuda.dll")
    driver.cuDevicePrimaryCtxRetain.argtypes = [C.POINTER(C.c_void_p), C.c_int]
    driver.cuCtxGetCurrent.argtypes = [C.POINTER(C.c_void_p)]
    driver.cuCtxSetCurrent.argtypes = [C.c_void_p]
    driver.cuDevicePrimaryCtxRelease_v2.argtypes = [C.c_int]
    primary = C.c_void_p()
    assert driver.cuDevicePrimaryCtxRetain(C.byref(primary), 0) == 0

    def current():
        context = C.c_void_p()
        assert driver.cuCtxGetCurrent(C.byref(context)) == 0
        return context.value

    try:
        assert driver.cuCtxSetCurrent(primary) == 0
        ti.init(arch=ti.cuda, offline_cache=False)
        assert current() == primary.value
        field = ti.field(ti.i32, shape=16)
        field.fill(7)
        assert current() == primary.value
        output = ti.ndarray(ti.i32, 16)
        output.fill(0)
        assert current() == primary.value

        @ti.kernel
        def increment(out: ti.types.ndarray(ti.i32, 1)):
            for i in out:
                field[i] += 1
                out[i] = field[i]

        builder = ti.graph.GraphBuilder()
        builder.dispatch(increment, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1))
        graph = builder.compile()
        domain = ti.graph.execution_domain(instance_capacity=2)
        prepared = graph.prepare(dict(out=output), execution_domain=domain)
        prepared.initialize()
        assert current() == primary.value
        errors = []

        def worker():
            try:
                assert driver.cuCtxSetCurrent(primary) == 0
                for _ in range(5):
                    ticket = prepared.submit()
                    assert current() == primary.value
                    ticket.wait()
                    assert current() == primary.value
                    ticket.close()
            except BaseException as error:
                errors.append(error)
            finally:
                driver.cuCtxSetCurrent(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert not errors, errors
        assert output.to_numpy().tolist() == [12] * 16
        assert current() == primary.value
        ti.reset()
        assert current() == primary.value
    finally:
        driver.cuCtxSetCurrent(None)
        driver.cuDevicePrimaryCtxRelease_v2(0)
