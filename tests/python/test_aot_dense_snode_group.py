import ctypes as C

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge_runtime.aot import AotError, Runtime
from tests import test_utils
from tests.python.test_dense_snode_schema import _group


def _compiled_nodes(root):
    nodes = []

    def visit(node, parent):
        index = len(nodes)
        nodes.append(
            dict(
                parent=parent,
                type=int(node.type),
                dtype=str(node.data_type()),
                shape=[node.get_shape_along_axis(i) for i in range(node.num_active_indices())],
                cell_size=node.cell_size_bytes,
                offset=node.offset_bytes_in_parent_cell,
            )
        )
        for child in range(node.get_num_ch()):
            visit(node.get_ch(child), index)

    visit(root, -1)
    return nodes


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_standalone_group_uses_jit_layout_and_owns_member_storage():
    schema, producer, _ = _group(7)
    frozen = schema.to_json()
    arch = "cpu" if ti.cfg.arch == ti.cpu else "cuda"
    with Runtime(arch) as runtime:
        first = runtime.dense_snode_group(frozen, {"N": 37})
        second = runtime.dense_snode_group(frozen, {"N": 113})
        for group, n in ((first, 37), (second, 113)):
            _, expected_tree, root = _group(n)
            try:
                assert group.layout["nodes"] == _compiled_nodes(root)
                assert group.nbytes == root.cell_size_bytes
            finally:
                expected_tree.destroy()
        assert first.layout["nodes"][3]["offset"] != second.layout["nodes"][3]["offset"]
        x = first.member([[0, 0]], "f32")
        alias = first.member([[0, 0]], "f32")
        v = first.member([[1, i] for i in range(3)], "f64", element_shape=(3,))
        m = first.member([[2, i] for i in range(4)], "f32", element_shape=(2, 2))
        scalar = first.member([[3, 0]], "i32")
        other = second.member([[0, 0]], "f32")
        assert x.shape == (37,) and v.shape == m.shape == x.shape
        assert scalar.shape == ()
        for array in (x, v, m, scalar, other):
            assert not any(array.read())
        x.write(np.arange(37, dtype=np.float32))
        np.testing.assert_array_equal(np.frombuffer(alias.read(), np.float32), np.arange(37))
        vector_data = np.arange(111, dtype=np.float64)
        v.write(vector_data)
        np.testing.assert_array_equal(np.frombuffer(v.read(), np.float64), vector_data)
        m.write(np.arange(148, dtype=np.float32))
        scalar.write(np.array([91], dtype=np.int32))
        assert np.frombuffer(scalar.read(), np.int32)[0] == 91
        assert not any(other.read())

        resources = len(runtime._resources)
        for paths, dtype, shape in (([[99]], "f32", ()), ([[0, 0]], "i32", ()), ([[1, 0], [1, 0]], "f64", (2,))):
            with pytest.raises(AotError):
                first.member(paths, dtype, element_shape=shape)
            assert len(runtime._resources) == resources
        first.close()
        with pytest.raises(AotError, match="retired"):
            alias.read()
        assert not any(other.read())
        # Revoked member shells remain closeable, in arbitrary order.
        for array in (alias, scalar, x, m, v):
            array.close()
        second.close()
        other.close()
    producer.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_dense_group_rejects_stale_or_other_runtime_and_invalid_dimensions():
    schema, producer, _ = _group(7)
    with Runtime("cpu") as runtime, Runtime("cpu") as wrong:
        for dimensions in ({}, {"N": 0}, {"N": -1}, {"N": 7, "unknown": 3}):
            with pytest.raises(AotError):
                runtime.dense_snode_group(schema.to_json(), dimensions)
            assert not runtime._resources
        group = runtime.dense_snode_group(schema.to_json(), {"N": 11})
        handle = group._memory._handle
        wrong._api.load_dense_snode_groups()
        with pytest.raises(AotError, match="runtime"):
            wrong._api.call(
                wrong._api.ti_get_dense_snode_group_layout_ext, wrong._handle, handle, C.byref(C.c_uint64()), None
            )
        group.close()
        with pytest.raises(AotError, match="runtime"):
            runtime._api.call(
                runtime._api.ti_get_dense_snode_group_layout_ext, runtime._handle, handle, C.byref(C.c_uint64()), None
            )
    producer.destroy()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_group_members_bind_loaded_code_and_retire_prepared_users(tmp_path):
    @ti.kernel
    def advance(x: ti.types.ndarray(ti.f32, 1)):
        for i in range(x.shape[0]):
            x[i] += i + 1

    schema, producer, _ = _group(7)
    exported = ti.aot.Module()
    exported.add_kernel(advance)
    artifact = tmp_path / "group.tcm"
    exported.archive(str(artifact))
    with Runtime("cpu" if ti.cfg.arch == ti.cpu else "cuda") as runtime:
        module = runtime.load_module(artifact)
        entry = module.kernel("advance")
        a = runtime.dense_snode_group(schema.to_json(), {"N": 37})
        b = runtime.dense_snode_group(schema.to_json(), {"N": 113})
        ax = a.member([[0, 0]], "f32")
        bx = b.member([[0, 0]], "f32")
        with entry.prepare(ax) as pa, entry.prepare(bx) as pb:
            pa.run()
            pb.run()
            pa.run()
            a.close()  # Cold drain/revocation, not an asynchronous close claim.
            with pytest.raises(AotError, match="retired|invalid|live|closed"):
                pa.run()
            pb.run()
            np.testing.assert_array_equal(np.frombuffer(bx.read(), np.float32), 2 * np.arange(1, 114))
    producer.destroy()
