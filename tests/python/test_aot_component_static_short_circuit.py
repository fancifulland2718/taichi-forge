"""Static short circuit excludes unreachable operations and capture dependencies."""
import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@ti.data_oriented
class State:
    def __init__(self, n, enabled=False):
        self.n, self.enabled = n, enabled
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.static("enabled", choices=(False, True))
        scope.out = scope.field(ti.i32, shape=scope.dim("n"))
        self.tree = builder.finalize()

    @ti.kernel
    def run(self):
        if ti.static(self.enabled and self.n == 1):
            self.out[0] = 101
        if ti.static(not self.enabled or self.missing):
            for i in range(self.n):
                self.out[i] = i + ti.static(0 or 7 or self.missing)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_component_static_short_circuit_preserves_actual_capture_contract():
    first, second = State(3), State(9)
    env = ti.aot.Component.from_owner(first)
    code = env.compile(first.run)
    other = ti.aot.Component.from_owner(second)
    assert other.compile(second.run) is code
    code.run(other, second)
    np.testing.assert_array_equal(second.out.to_numpy(), np.arange(9) + 7)
    # The same expression is still invalid when its dynamic operand is reached.
    second.enabled = True
    with pytest.raises(ti.TaichiCompilationError, match="runtime capture.*n.*ti.static"):
        other.compile(second.run)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_normal_jit_static_short_circuit_does_not_evaluate_dead_operations():
    @ti.kernel
    def values() -> ti.i32:
        return ti.static((0 and (1 // 0)) + (4 or (1 // 0)))
    assert values() == 4
    state = State(4)
    state.run()
    np.testing.assert_array_equal(state.out.to_numpy(), np.arange(4) + 7)
