"""Primitive type captures keep typed identity and source-free selection."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti

from taichi_forge.aot import Component
from taichi_forge_runtime.aot import Runtime, StaticDType
from taichi_forge_runtime._static_values import read_static_key, static_key
from tests import test_utils


@ti.data_oriented
class _State:
    def __init__(self, n, dtype):
        self.n, self.dtype = n, dtype
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.static_dtype("dtype", choices=(ti.f32, ti.f64))
        scope.output = scope.field(ti.f64, shape=scope.dim("n"))
        self.tree = builder.finalize()

    @ti.kernel
    def fill(self):
        for i in range(self.n):
            self.output[i] = ti.cast(16777217, self.dtype) + ti.cast(i, self.dtype)

    def record(self, builder):
        builder.dispatch(self.fill)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_static_dtype_identity_and_loaded_contract(tmp_path):
    first, other, wide = _State(3, ti.f32), _State(17, ti.f32), _State(17, ti.f64)
    env = Component.from_owner(first)
    code = env.compile(first.fill)
    assert Component.from_owner(other).compile(other.fill) is code
    wide_env = Component.from_owner(wide)
    wide_code = wide_env.compile(wide.fill)
    assert wide_code.semantic_key != code.semantic_key
    assert dict(wide_code.contract["static"])["dtype"] == ["dtype", "f64"]
    wide_code.run(wide_env, wide)
    np.testing.assert_array_equal(wide.output.to_numpy(), 16777217 + np.arange(17))
    code.run(Component.from_owner(other), other)
    expected = (np.float32(16777217) + np.arange(17, dtype=np.float32)).astype(np.float64)
    np.testing.assert_array_equal(other.output.to_numpy(), expected)
    wide.dtype = "f64"
    with pytest.raises(ti.TaichiCompilationError, match="outside its declared domain"):
        wide_env.compile(wide.fill)
    wide.dtype = StaticDType("f64")
    with pytest.raises(ti.TaichiCompilationError, match="consumer configuration"):
        wide_env.compile(wide.fill)
    wide.dtype = ti.f64
    assert wide_env.compile(wide.fill) is wide_code
    root = wide_env.compile_graph(wide.record)
    module = ti.aot.Module()
    root.export(module, "typed")
    path = tmp_path / "typed.tcm"
    module.archive(str(path))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(path).component("typed")
        with pytest.raises(ValueError):
            component.instantiate(dict(n=7), configuration=dict(dtype="f64"))
        with component.instantiate(dict(n=7), configuration=dict(dtype=StaticDType("f64"))) as instance:
            instance.initialize(zero=("output",))
            with instance.prepare() as call:
                call.run()
            np.testing.assert_array_equal(np.frombuffer(instance.resource("output").read(), np.float64),
                                          16777217 + np.arange(7))


def test_static_dtype_serialization_needs_no_frontend():
    key = static_key((StaticDType("f64"), "f64"))
    assert key[1][0] != key[1][1]
    assert static_key(read_static_key(json.loads(json.dumps(key)))) == key
    for invalid in ("unknown", "float64", 64):
        with pytest.raises(ValueError):
            read_static_key(["dtype", invalid])
    # A fresh runtime-only process must not import any numerical frontend.
    script = (
        "import sys; from taichi_forge_runtime.aot import StaticDType; "
        "from taichi_forge_runtime._static_values import static_key, read_static_key; "
        "assert static_key(read_static_key(['dtype', 'f64'])) == ('dtype', 'f64'); "
        "assert not any(n == 'taichi_forge' or n.startswith('taichi_forge.') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], env=dict(os.environ), check=True)
