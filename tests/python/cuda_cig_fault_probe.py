"""Destructive CUDA-context fault qualification; always run in a subprocess."""

import ctypes as C
import json
import sys

from cuda_cig_test_owner import NativeCigOwner
from taichi_forge_runtime.aot import Runtime, configure_cuda_context, cuda_context_info


def check(stage):
    owner = NativeCigOwner()
    configure_cuda_context(owner.binding)
    runtime = Runtime("cuda")
    domain = runtime.execution_domain()
    context = domain.info["context"]
    if stage == "initialize":
        runtime.close()
    driver = C.WinDLL("nvcuda.dll")
    driver.cuCtxGetCurrent.argtypes = [C.POINTER(C.c_void_p)]
    driver.cuCtxSetCurrent.argtypes = [C.c_void_p]
    driver.cuModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
    driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
    driver.cuLaunchKernel.argtypes = [C.c_void_p, *([C.c_uint] * 7), C.c_void_p, C.c_void_p, C.c_void_p]
    previous, module, function = C.c_void_p(), C.c_void_p(), C.c_void_p()
    assert driver.cuCtxGetCurrent(C.byref(previous)) == 0
    assert driver.cuCtxSetCurrent(context) == 0
    ptx = b".version 7.0\n.target sm_80\n.address_size 64\n.visible .entry fail() { .reg .b64 a; mov.u64 a,0; st.global.u32 [a],1; ret; }\n"
    assert driver.cuModuleLoadData(C.byref(module), ptx) == 0
    assert driver.cuModuleGetFunction(C.byref(function), module, b"fail") == 0
    assert driver.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, None, None, None) == 0
    driver.cuCtxSetCurrent(previous)
    errors = []
    operations = ([lambda: Runtime("cuda"), lambda: Runtime("cuda")] if stage == "initialize"
                  else [runtime.wait, lambda: runtime.ndarray("i32", 16), runtime.close])
    for operation in operations:
        try:
            operation()
        except RuntimeError as error:
            errors.append(str(error))
        else:
            raise AssertionError("A failed context accepted new work")
    assert all("700" in message or "ILLEGAL_ADDRESS" in message or "LAUNCH_FAILED" in message or "illegal memory" in message
               for message in errors), errors
    if stage == "initialize":
        assert errors[0] == errors[1], errors
    info = cuda_context_info()
    assert owner.references == 2
    owner.close()
    # Native objects of a context-fatal failure remain quarantined until process
    # exit. Do not attempt another CUDA free/unload merely to tidy up the probe.
    return dict(stage=stage, errors=errors, context_mode=info["context_mode"], bounded_failure=True)


if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1])), flush=True)
