"""Ordered Dense axes preserve JIT layout and per-instance affine addressing."""

import json

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils
from tests.python.test_aot_dense_snode_group import _compiled_nodes


@ti.data_oriented
class _Ordered:
    def __init__(self, count):
        self.count = count
        scope = ti.experimental.ResourceScope(self, initialization="zero")
        n = scope.dim("count")
        scope.a = scope.field(ti.i32, shape=(3, 2, n), order="kij")
        scope.v = scope.vector_field(2, ti.i32, shape=(2, n), order="ji")
        scope.tail = scope.field(ti.i32, shape=(1, 2, 2, n), order="lijk")

    @ti.kernel
    def step(self):
        for i, j, k in self.a:
            self.a[i, j, k] += 100 * i + 10 * j + k
        for j, k in self.v:
            self.v[j, k] += ti.Vector([self.a[2, j, k], k + 7])
        for a, i, j, k in self.tail:
            self.tail[a, i, j, k] += i * 10 + j + k

    def record(self, builder):
        builder.dispatch(self.step)


def _expected(count):
    i, j, k = np.indices((3, 2, count))
    a = (100 * i + 10 * j + k).astype(np.int32)
    v = np.stack([a[2], np.broadcast_to(np.arange(count) + 7, (2, count))], axis=-1)
    _, i, j, k = np.indices((1, 2, 2, count))
    return dict(a=a, v=v.astype(np.int32), tail=(i * 10 + j + k).astype(np.int32))


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_ordered_dense_preserves_layout_and_frozen_multi_instance_execution(tmp_path):
    model = _Ordered(3)
    env = Component.from_owner(model)
    code = env.compile(model.step)
    root = env.compile_graph(model.record)
    module = ti.aot.Module()
    root.export(module, "step")
    artifact = tmp_path / "ordered.tcm"
    module.archive(str(artifact))
    backend = ti.cfg.arch.name
    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("step")
        instances = []
        try:
            for count in (7, 37):
                normal = _Ordered(count)
                current = Component.from_owner(normal)
                assert current.compile(normal.step) is code
                normal.step()
                expected = _expected(count)
                for name, data in expected.items():
                    np.testing.assert_array_equal(getattr(normal, name).to_numpy(), data)
                    getattr(normal, name).fill(0)
                code.run(current, normal)
                for name, data in expected.items():
                    np.testing.assert_array_equal(getattr(normal, name).to_numpy(), data)
                with ti.aot.borrow_runtime() as borrowed:
                    view = borrowed.import_dense_view(ti.experimental.ndarray_view(normal.a))
                    view.write(expected["a"] + 1)
                    np.testing.assert_array_equal(normal.a.to_numpy(), expected["a"] + 1)
                    np.testing.assert_array_equal(
                        np.frombuffer(view.read(), np.int32).reshape(expected["a"].shape), expected["a"] + 1)
                    view.close()
                instance = component.instantiate(dict(count=count))
                instances.append(instance)
                actual_root = ti.lang.impl.get_runtime().prog.get_snode_root(
                    normal.a.snode.ptr.get_snode_tree_id())
                assert list(instance.layout.values())[0]["nodes"] == _compiled_nodes(actual_root)
                with instance.prepare() as call:
                    call.run()
                for name, data in expected.items():
                    actual = np.frombuffer(instance.resource(name).read(), np.int32).reshape(data.shape)
                    np.testing.assert_array_equal(actual, data)
            # Reusing one loaded code owner never redirects the first state.
            with instances[1].prepare() as call:
                call.run()
            np.testing.assert_array_equal(
                np.frombuffer(instances[0].resource("a").read(), np.int32).reshape(3, 2, 7),
                _expected(7)["a"],
            )
        finally:
            for instance in instances:
                instance.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_ordered_dense_path_product_rejected_before_allocation():
    model = _Ordered(3)
    root = Component.from_owner(model).compile_graph(model.record)
    schema = next(iter(root.contract["groups"].values()))
    declaration = json.loads(schema)
    parameters = {e["parameter"] for node in declaration["nodes"]
                  for e in node["extents"] if e["parameter"]}
    program = ti.lang.impl.get_runtime().prog
    before = program.get_active_snode_tree_ids()
    with pytest.raises(ValueError, match="i32 indexing range"):
        core.DenseSNodeSchema.from_json(schema).instantiate(program, dict.fromkeys(parameters, 2**30))
    assert program.get_active_snode_tree_ids() == before
