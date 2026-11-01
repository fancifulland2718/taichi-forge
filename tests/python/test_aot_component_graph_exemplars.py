"""Component recording shares normal Graph runtime ABI exemplar validation."""

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class _Output:
    def __init__(self):
        scope = ti.experimental.ResourceScope(self)
        scope.bias = scope.field(ti.i32, shape=(), initialization="zero")
        self.bias[None] = 99

    @ti.kernel
    def apply(self, output: ti.types.ndarray(ndim=1), scale: ti.i32):
        for i in output:
            output[i] = scale * i + self.bias[None]

    def record(self, builder, exemplar):
        output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, 1)
        scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.i32)
        builder.dispatch(
            self.apply, output, scale, template_args={"self": self, "output": exemplar}
        )


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_runtime_exemplars_remain_dynamic_and_keep_validation(tmp_path):
    model = _Output()
    component = ti.aot.Component.from_owner(model)
    with pytest.raises(
        ti.TaichiCompilationError, match="does not match its symbolic ndarray"
    ):
        component.compile_graph(model.record, ti.ndarray(ti.f32, shape=3))
    a = component.compile_graph(model.record, ti.ndarray(ti.i32, shape=3))
    b = component.compile_graph(model.record, ti.ndarray(ti.i32, shape=11))
    assert a.contract["semantic_key"] == b.contract["semantic_key"]
    array = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, 1)
    scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.i32)
    with pytest.raises(
        ti.TaichiCompilationError, match="Unknown Graph template arguments"
    ):
        component.compile_graph(
            lambda builder: builder.dispatch(
                model.apply, array, scale, template_args={"self": model, "unknown": 1}
            )
        )
    with pytest.raises(ti.TaichiCompilationError, match="can only bind"):
        component.compile_graph(
            lambda builder: builder.dispatch(
                model.apply, array, scale, template_args={"self": model, "scale": 2}
            )
        )
    module = ti.aot.Module()
    a.export(module, "output")
    artifact = tmp_path / "output.tcm"
    module.archive(str(artifact))
    arch = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(arch) as runtime:
        with runtime.load_module(artifact) as module:
            with module.component("output").instantiate({}) as state:
                array = runtime.ndarray("i32", (37,))
                try:
                    with state.prepare(arguments=dict(output=array, scale=7)) as call:
                        call.run()
                    np.testing.assert_array_equal(
                        np.frombuffer(array.read(), np.int32), 7 * np.arange(37)
                    )
                finally:
                    array.close()
