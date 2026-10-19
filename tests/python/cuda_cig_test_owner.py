"""Native queue/device lease fixture; safe in python -S consumers without DSL."""
import ctypes as C
import os
from pathlib import Path

_capsule_name = b"taichi.cuda_vulkan_context_binding.v1"
_libraries = []


class NativeCigOwner:
    def __init__(self):
        from taichi_forge_runtime.aot import CudaVulkanContextBinding

        path = os.environ.get("TI_TEST_CIG_FIXTURE", str(
            Path(__file__).resolve().parents[2] / "build/taichi_vulkan_cig_fixture.dll"
        ))
        self.library = C.CDLL(path)
        _libraries.append(self.library)
        self.library.cig_fixture_create.restype = C.c_void_p
        self.library.cig_fixture_error.restype = C.c_char_p
        self.library.cig_fixture_references.argtypes = [C.c_void_p]
        self.library.cig_fixture_references.restype = C.c_uint
        self.library.cig_fixture_release.argtypes = [C.c_void_p]
        self.library.cig_fixture_release.restype = None
        self.library.cig_fixture_vulkan.argtypes = [C.c_void_p]
        self.library.cig_fixture_vulkan.restype = C.c_void_p
        self.address = self.library.cig_fixture_create()
        if not self.address:
            raise RuntimeError(self.library.cig_fixture_error().decode())
        self.closed = False
        capsule = C.pythonapi.PyCapsule_New
        capsule.argtypes, capsule.restype = [C.c_void_p, C.c_char_p, C.c_void_p], C.py_object
        self.binding = CudaVulkanContextBinding(capsule(self.address, _capsule_name, None))

    @property
    def references(self):
        return self.library.cig_fixture_references(self.address)

    def close(self):
        if not self.closed:
            self.library.cig_fixture_release(self.address)
            self.closed = True


def configure_test_context():
    if not os.environ.get("TI_TEST_CIG"):
        return None
    from taichi_forge_runtime.aot import configure_cuda_context, cuda_context_info
    current = cuda_context_info()
    if current["context_mode"] == "vulkan_cig":
        return current
    assert current["context_mode"] == "uninitialized", current
    owner = NativeCigOwner()
    try:
        assert owner.references == 1
        current = configure_cuda_context(owner.binding)
        assert owner.references == 2
        assert current["context_mode"] == "vulkan_cig", current
        checks = current["allocation_checks"]
        assert checks["device"] and checks["managed"] and checks["capture"], current
        assert not checks["async"] and not checks["owned_pool"], current
        assert not current["fallback_reason"], current
    finally:
        owner.close()
    # The callback and C++ allocation remain live without this Python owner.
    assert owner.references == 1
    return current


if __name__ == "__main__":
    import json
    import sys
    os.environ["TI_TEST_CIG"] = "1"
    print(json.dumps(configure_test_context()), flush=True)
    if len(sys.argv) > 1:
        import pytest
        raise SystemExit(pytest.main(sys.argv[1:]))
