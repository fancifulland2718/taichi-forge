import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.aot._component import Component
from tests import test_utils


@ti.data_oriented
class _Declared:
    def __init__(self, count):
        self.count = count
        builder = ti.FieldsBuilder()
        state = ti.experimental.ResourceScope(self, builder=builder)
        state.x = state.field(ti.f32, shape=state.dim("count").maximum(1))
        state.y = state.field(ti.f32, shape=state.dim("count") + 1)
        state.alias = state.x
        self.tree = builder.finalize()

    @ti.kernel
    def fill(self):
        for i in range(self.count):
            self.alias[i] = i + 2
        for i in self.y:
            self.y[i] = i * 3


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_normal_declarations_preserve_relations_and_actual_root():
    from taichi_forge.aot._component import Dimension
    from taichi_forge.lang._resource_declaration import Extent

    expression = ["max", ["dim", "count"], ["const", 1]]
    extent, dimension = Extent(expression), Dimension(expression=expression)
    expression[2][1] = 9
    assert extent.expression == dimension.expression == ("max", ("dim", "count"), ("const", 1))
    small = _Declared(3)
    env = Component.from_owner(small)
    code = env.compile(small.fill)
    large = _Declared(17)
    large.unrelated_count = 11
    independent = ti.FieldsBuilder()
    unrelated = ti.experimental.ResourceScope(large, builder=independent)
    unrelated.unused = unrelated.field(ti.i32, shape=unrelated.dim("unrelated_count"))
    large.independent_tree = independent.finalize()
    large_env = Component.from_owner(large)
    assert large_env.compile(large.fill) is code
    code.run(large_env, large)
    np.testing.assert_array_equal(large.x.to_numpy(), np.arange(17) + 2)
    np.testing.assert_array_equal(large.y.to_numpy(), np.arange(18) * 3)
    assert len(code.contract["groups"]) == 1
    assert "count" in code.contract["dimensions"]
    assert "unrelated_count" not in code.contract["dimensions"]
    assert len(code.contract["captures"]) == 3
    assert code.contract["declaration_sources"]["alias"][0].endswith("test_resource_declaration.py")
    empty = _Declared(0)
    empty_env = Component.from_owner(empty)
    assert empty_env.compile(empty.fill) is code
    code.run(empty_env, empty)
    assert empty.x.shape == (1,)
    ordinary = _Declared(17)
    ordinary.fill()
    np.testing.assert_array_equal(large.y.to_numpy(), ordinary.y.to_numpy())
    large.x = ti.field(ti.f32, shape=17)
    with pytest.raises(ti.TaichiCompilationError, match="replaced without redeclaration"):
        Component.from_owner(large)
