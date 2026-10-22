"""F3 declarations come from the native implementation, including old cores."""

import ctypes as C
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import taichi_forge as ti


NAMES = (
    "cuda_execution_domains",
    "cuda_aot_execution_domains",
    "cuda_jit_graph_execution_domains",
    "cuda_external_storage_execution_domains",
    "cuda_canvas_execution_domains",
)


def test_domain_manifest_matches_runtime_only_without_initializing_a_device():
    from taichi_forge_runtime import runtime_contract_manifest

    ti.reset()
    native = runtime_contract_manifest()
    jit = ti.runtime_contract_manifest()
    assert native["native_abi_revision"] == jit["native_abi_revision"]
    assert dict(native["features"]) == dict(jit["features"])
    cuda_api = jit["features"]["cuda"] and hasattr(ti._lib.core, "_borrow_aot_runtime")
    assert native["features"][NAMES[0]] == cuda_api
    assert native["features"][NAMES[1]] == cuda_api
    assert native["features"][NAMES[2]] == (cuda_api and jit["features"]["llvm"])
    assert native["features"][NAMES[3]] == cuda_api
    assert native["features"][NAMES[4]] == (
        cuda_api and jit["features"]["vulkan"] and jit["features"]["ggui"]
    )
    assert native["features"]["cuda_submission_parameters_u64"] == cuda_api
    assert native["features"]["cuda_completion_gpu_timing"] == cuda_api
    assert ti.lang.impl.get_runtime().prog is None
    with pytest.raises(TypeError):
        native["features"][NAMES[0]] = False


def test_runtime_only_manifest_does_not_import_the_dsl():
    code = """
import importlib.abc, json, sys
class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'numpy'):
            raise AssertionError('Unexpected frontend import: '+fullname)
sys.meta_path.insert(0, NoFrontend())
from taichi_forge_runtime import runtime_contract_manifest, _native
assert _native.loaded_path() is None
value = runtime_contract_manifest()
print(json.dumps(dict(features=dict(value['features']), abi=value['native_abi_revision'])))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    assert observed["features"] == dict(ti.runtime_contract_manifest()["features"])


@pytest.mark.parametrize("bitmap", [0, (1 << 12) - 1, ((1 << 17) - 1)])
def test_old_native_bits_do_not_infer_domains_from_version_or_abi(monkeypatch, bitmap):
    from taichi_forge_runtime import _contracts

    class Query:
        def __call__(self, pointer, size):
            native = C.cast(pointer, C.POINTER(_contracts._Bootstrap)).contents
            assert size == C.sizeof(native)
            native.struct_size = size
            native.manifest_schema_version = 1
            native.native_abi_revision = 87
            native.feature_bitmap = bitmap
            native.compiler_abi = b"legacy-test-compiler"
            return 0

    library = SimpleNamespace(taichi_forge_runtime_bootstrap_v1=Query())
    monkeypatch.setattr(_contracts._native, "load_library", lambda: library)
    manifest = _contracts.runtime_contract_manifest()
    assert manifest["native_abi_revision"] == 87
    assert not manifest["features"]["cuda_submission_parameters_u64"]
    assert not manifest["features"]["cuda_completion_gpu_timing"]
    assert all(
        manifest["features"][name] == bool(bitmap & (1 << (12 + i)))
        for i, name in enumerate(NAMES)
    )
