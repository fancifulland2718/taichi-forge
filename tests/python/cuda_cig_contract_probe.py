"""Fresh-process CiG startup and caller-context contracts, usable with -S."""

import ctypes as C
import json
import sys
import threading

from cuda_cig_test_owner import NativeCigOwner
from taichi_forge_runtime.aot import Runtime, configure_cuda_context, cuda_context_info
from taichi_forge_runtime.aot._cuda_context import _Config, _functions


def rejected(operation, text):
    try:
        operation()
    except (RuntimeError, ValueError) as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError("Invalid operation was accepted")


def main(scenario):
    assert cuda_context_info()["context_mode"] == "uninitialized"
    if scenario in ("auto", "disabled"):
        info = configure_cuda_context(mode=scenario)
        assert info["context_mode"] == "ordinary"
        assert bool(info["fallback_reason"]) == (scenario == "auto")
        with Runtime("cuda") as runtime:
            output = runtime.ndarray("i32", (4,))
            output.write(bytes(range(16)))
            assert output.read() == bytes(range(16))
        rejected(lambda: configure_cuda_context(mode="disabled"), "first use")
        return info

    owner = NativeCigOwner()
    try:
        if scenario == "invalid":
            api, configure, _ = _functions(None)
            config = _Config.from_buffer_copy(C.string_at(owner.address, C.sizeof(_Config)))
            config.schema_version = 9
            rejected(lambda: api.call(configure, C.byref(config)), "descriptor")
            assert owner.references == 1
            assert cuda_context_info()["context_mode"] == "uninitialized"
            config.schema_version = 1
            config.device_uuid[0] ^= 255
            rejected(lambda: api.call(configure, C.byref(config)), "UUID")
            assert owner.references == 1
            assert cuda_context_info()["context_mode"] == "uninitialized"
            return {"invalid_schema_and_uuid_rejected": True, "owner_references": owner.references}

        driver = C.WinDLL("nvcuda.dll")
        driver.cuInit.argtypes = [C.c_uint32]
        driver.cuDevicePrimaryCtxRetain.argtypes = [C.POINTER(C.c_void_p), C.c_int]
        driver.cuCtxGetCurrent.argtypes = [C.POINTER(C.c_void_p)]
        driver.cuCtxSetCurrent.argtypes = [C.c_void_p]
        driver.cuDevicePrimaryCtxRelease_v2.argtypes = [C.c_int]
        assert driver.cuInit(0) == 0
        primary = C.c_void_p()
        assert driver.cuDevicePrimaryCtxRetain(C.byref(primary), 0) == 0
        assert driver.cuCtxSetCurrent(primary) == 0

        def current():
            value = C.c_void_p()
            assert driver.cuCtxGetCurrent(C.byref(value)) == 0
            return value.value

        try:
            info = configure_cuda_context(owner.binding)
            assert info["context_mode"] == "vulkan_cig"
            assert current() == primary.value
            assert owner.references == 2
            rejected(lambda: configure_cuda_context(owner.binding), "first use")
            assert owner.references == 2
            errors = []

            def worker(expected):
                try:
                    assert driver.cuCtxSetCurrent(expected) == 0
                    for _ in range(3):
                        with Runtime("cuda") as runtime:
                            assert current() == expected
                            output = runtime.ndarray("i32", (4,))
                            assert current() == expected
                            output.write(bytes(range(16)))
                            assert current() == expected
                            assert output.read() == bytes(range(16))
                            assert current() == expected
                            domains = [runtime.execution_domain(capacity=2) for _ in range(3)]
                            assert len({domain.info["stream"] for domain in domains}) == 3
                            assert len({domain.info["context"] for domain in domains}) == 1
                            assert current() == expected
                            for domain in domains:
                                ticket = domain.latest_completion()
                                ticket.wait()
                                ticket.close()
                                domain.close()
                            assert current() == expected
                        assert current() == expected
                except BaseException as error:
                    errors.append(error)
                finally:
                    driver.cuCtxSetCurrent(None)

            # Separate threads cover both an empty context stack and a foreign
            # primary context. Runtime reconstruction keeps the native CiG lease.
            for previous in (None, primary.value):
                thread = threading.Thread(target=worker, args=(previous,), daemon=True)
                thread.start()
                thread.join(timeout=30)
                assert not thread.is_alive(), "Worker did not retire in bounded time"
                assert not errors, errors
            assert current() == primary.value
            assert owner.references == 2
            assert cuda_context_info()["generation"] == info["generation"]
            return dict(info, caller_context_restored=True, runtime_reconstruction_count=6)
        finally:
            driver.cuCtxSetCurrent(None)
            driver.cuDevicePrimaryCtxRelease_v2(0)
    finally:
        owner.close()


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1])), flush=True)
