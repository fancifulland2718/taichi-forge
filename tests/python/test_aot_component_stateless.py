"""Stateless ordinary functions use the existing component Graph contract."""

from contextlib import closing

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import RuntimeValue
from tests import test_utils


@ti.kernel
def _scale(src: ti.types.ndarray(ti.f32, 1), dst: ti.types.ndarray(ti.f32, 1), factor: ti.template()):
    for i in src:
        dst[i] = src[i] * factor


_UNDECLARED = 7


@ti.kernel
def _unclassified(dst: ti.types.ndarray(ti.i32, 1)):
    dst[0] = _UNDECLARED


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_stateless_graph_owner_and_global_guards(tmp_path):
    src = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "src", ti.f32, ndim=1)
    dst = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.f32, ndim=1)

    def record(builder):
        builder.dispatch(_scale, src, dst, template_args=dict(factor=3))

    code = ti.aot.Component().compile_graph(record)
    same = ti.aot.Component().compile_graph(record)
    assert code.contract["semantic_key"] == same.contract["semantic_key"]
    assert not code.contract["resources"]
    assert not code.contract["dimensions"]
    exported = ti.aot.Module()
    code.export(exported, "scale")
    path = tmp_path / "stateless.tcm"
    exported.archive(str(path))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(path) as module:
        with module.component("scale").instantiate({}) as instance:
            instance.initialize({})
            for size in (1, 17):
                with closing(runtime.ndarray("f32", size)) as x, closing(runtime.ndarray("f32", size)) as y:
                    data = np.arange(size, dtype=np.float32)
                    x.write(data.tobytes())
                    with instance.prepare(arguments=dict(src=x, dst=y)) as call:
                        call.run()
                        np.testing.assert_array_equal(np.frombuffer(y.read(), np.float32), data * 3)

    with pytest.raises(ti.TaichiCompilationError, match="undeclared global/closure"):
        ti.aot.Component().compile(_unclassified, ti.ndarray(ti.i32, 1))
    with pytest.raises(ti.TaichiCompilationError, match="declared resource owner"):
        ti.aot.Component(missing=RuntimeValue(ti.i32)).compile_graph(record)
