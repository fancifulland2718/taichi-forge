import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge_runtime._dimensions import evaluate_extent, extent_dependencies, freeze_extent
from tests import test_utils


def test_power_of_two_relation_validates_arity_and_integer_range():
    expression = freeze_extent(["ceil_pow2", ["dim", "count"]])
    assert extent_dependencies(expression) == {"count"}
    for count, expected in ((0, 1), (1, 1), (2, 2), (3, 4), (32, 32), (33, 64)):
        assert evaluate_extent(expression, lambda _: count) == expected
    with pytest.raises(ValueError, match="nonnegative"):
        evaluate_extent(expression, lambda _: -1)
    with pytest.raises(ValueError, match="i64"):
        evaluate_extent(expression, lambda _: 2**62 + 1)
    for invalid in (("ceil_pow2",), ("ceil_pow2", ("const", 1), ("const", 2))):
        with pytest.raises(ValueError, match="Invalid dimension"):
            freeze_extent(invalid)


@ti.data_oriented
class _Table:
    def __init__(self, count):
        self.count = count
        self.capacity = 1 << (max(2, 2 * count) - 1).bit_length()
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        capacity = scope.power_of_two_dim("capacity", (scope.dim("count") * 2).maximum(2))
        scope.table = scope.field(ti.i32, shape=capacity)
        scope.tail = scope.field(ti.i32, shape=scope.dim("count").maximum(1))
        self.tree = builder.finalize()

    @ti.kernel
    def fill(self):
        for i in range(self.capacity):
            self.table[i] = (i + self.count) & (self.capacity - 1)
        for i in self.tail:
            self.tail[i] = self.table[self.capacity - 1] + i

    def record(self, builder):
        builder.dispatch(self.fill)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_power_of_two_capacity_reuses_code_and_frozen_consumer_relation(tmp_path):
    from taichi_forge_runtime.aot import Runtime

    producer = _Table(3)
    env = ti.aot.Component.from_owner(producer)
    code = env.compile(producer.fill)
    other = _Table(17)
    other_env = ti.aot.Component.from_owner(other)
    assert other_env.compile(other.fill) is code
    code.run(other_env, other)
    np.testing.assert_array_equal(other.table.to_numpy(), (np.arange(64) + 17) & 63)
    np.testing.assert_array_equal(other.tail.to_numpy(), np.arange(17) + 16)
    other.capacity -= 1
    with pytest.raises(ti.TaichiCompilationError, match="Derived dimension self.capacity"):
        code.run(other_env, other)
    other.capacity += 1
    module = ti.aot.Module()
    env.compile_graph(producer.record).export(module, "table")
    archive = tmp_path / "table.tcm"
    module.archive(str(archive))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(archive).component("table")
        with pytest.raises(ValueError, match="dimension inputs differ"):
            component.instantiate(dict(count=37, capacity=75))
        for count in (0, 37):
            capacity = 1 << (max(2, 2 * count) - 1).bit_length()
            with component.instantiate(dict(count=count)) as instance:
                instance.initialize(zero=component.contract["resources"])
                with instance.prepare() as prepared:
                    prepared.run()
                np.testing.assert_array_equal(
                    np.frombuffer(instance.resource("table").read(), dtype=np.int32),
                    (np.arange(capacity) + count) & (capacity - 1),
                )
                np.testing.assert_array_equal(
                    np.frombuffer(instance.resource("tail").read(), dtype=np.int32),
                    np.arange(max(count, 1)) + ((capacity - 1 + count) & (capacity - 1)),
                )
