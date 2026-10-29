import ctypes as C
import json
import zipfile

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge_runtime.aot import AotError, Runtime
from taichi_forge_runtime.aot import _c_api as abi
from taichi_forge._lib import core
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
def test_dense_group_uses_actual_tree_bounds_above_legacy_node_limit():
    # Ordinary SNodes already support this tree. Deployment must not introduce
    # a separate 4096-node, child-index or tensor-lane limit.
    count = 4098
    builder = ti.FieldsBuilder()
    block = builder.dense(ti.i, 2)
    fields = [ti.field(ti.i32) for _ in range(count)]
    for field in fields:
        block.place(field)
    schema = core.DenseSNodeSchema.capture(builder.ptr, [])
    producer = builder.finalize()
    arch = "cpu" if ti.cfg.arch == ti.cpu else "cuda"
    try:
        with Runtime(arch) as runtime, runtime.dense_snode_group(schema.to_json(), {}) as group:
            assert group.layout["nodes"] == _compiled_nodes(builder.ptr)
            paths = [[0, lane] for lane in range(count)]
            vector = group.member(paths, "i32", element_shape=(count,))
            tail = group.member([paths[-1]], "i32")
            values = np.arange(2 * count, dtype=np.int32).reshape(2, count)
            vector.write(values)
            np.testing.assert_array_equal(np.frombuffer(tail.read(), np.int32), values[:, -1])
            tail.write(np.array([-17, 29], dtype=np.int32))
            values[:, -1] = [-17, 29]
            np.testing.assert_array_equal(np.frombuffer(vector.read(), np.int32).reshape(2, count), values)

            resources = len(runtime._resources)
            for bad_paths, shape in (
                ([[0, count]], ()),
                ([[0, 2**31]], ()),
                ([paths[-1]], (2**31, 2**31)),
                ([paths[-1]], (0,)),
                ([paths[-1]], (2,)),
            ):
                with pytest.raises((AotError, ValueError)):
                    group.member(bad_paths, "i32", element_shape=shape)
                assert len(runtime._resources) == resources
            tail.close()
            vector.close()
    finally:
        producer.destroy()


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


def _interleaved_group(n):
    before, after = ti.field(ti.i32), ti.field(ti.i32)
    vector, matrix = ti.Vector.field(3, ti.f32), ti.Matrix.field(2, 2, ti.f64)
    builder = ti.FieldsBuilder()
    block = builder.dense(ti.ij, (n, 3))
    block.place(before, vector, matrix, after)
    schema = core.DenseSNodeSchema.capture(builder.ptr, [(block.ptr, 0, "N")])
    tree = builder.finalize()
    return schema, tree, builder.ptr


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_interleaved_members_transfer_logical_elements_without_touching_neighbors():
    schema, producer, _ = _interleaved_group(7)
    with Runtime("cpu" if ti.cfg.arch == ti.cpu else "cuda") as runtime:
        with runtime.dense_snode_group(schema.to_json(), {"N": 37}) as group:
            _, expected_tree, root = _interleaved_group(37)
            assert group.layout["nodes"] == _compiled_nodes(root)
            expected_tree.destroy()
            before = group.member([[0, 0]], "i32")
            vector = group.member([[0, i] for i in range(1, 4)], "f32", element_shape=(3,))
            matrix = group.member([[0, i] for i in range(4, 8)], "f64", element_shape=(2, 2))
            after = group.member([[0, 8]], "i32")
            alias = group.member([[0, i] for i in range(1, 4)], "f32", element_shape=(3,))
            before.write(np.full((37, 3), 71, dtype=np.int32))
            after.write(np.full((37, 3), 73, dtype=np.int32))
            vector_data = np.arange(37 * 3 * 3, dtype=np.float32)
            matrix_data = np.arange(37 * 3 * 4, dtype=np.float64)
            vector.write(vector_data)
            matrix.write(matrix_data)
            np.testing.assert_array_equal(np.frombuffer(alias.read(), np.float32), vector_data)
            np.testing.assert_array_equal(np.frombuffer(matrix.read(), np.float64), matrix_data)
            assert (np.frombuffer(before.read(), np.int32) == 71).all()
            assert (np.frombuffer(after.read(), np.int32) == 73).all()
            # Legacy packed transfers and forged shape/type must fail before IO.
            raw = abi.MemorySlice(vector._memory._handle, 0, vector.nbytes)
            output = (C.c_char * vector.nbytes)()
            with pytest.raises(AotError, match="logical ndarray"):
                runtime._api.call(runtime._api.ti_copy_memory_to_host_ext, runtime._handle, output, C.byref(raw))
            for change in ("shape", "type"):
                descriptor = vector._descriptor(runtime)
                if change == "shape":
                    descriptor.shape.dims[0] -= 1
                else:
                    descriptor.elem_type = before._code
                with pytest.raises(AotError, match="differs"):
                    runtime._api.call(runtime._api.ti_copy_ndarray_to_host_ext, runtime._handle, output,
                                      C.byref(descriptor), vector.nbytes)
        with pytest.raises(AotError, match="retired"):
            alias.read()
    producer.destroy()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_interleaved_first_member_uses_kernel_strides_and_rejects_packed_graph(tmp_path):
    @ti.kernel
    def increment(x: ti.types.ndarray(ti.i32, 2)):
        for i, j in x:
            x[i, j] += i * 3 + j

    schema, producer, _ = _interleaved_group(7)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.i32, ndim=2))
    # Component tests cover bitcode; exercise the metadata-only CUDA code path.
    exported = ti.aot.Module(artifact_format="ptx" if ti.cfg.arch == ti.cuda else "bc")
    exported.add_kernel(increment)
    exemplar = ti.ndarray(ti.i32, (14, 3))
    exported.add_kernel(increment, name="affine", template_args={
        "x": ti.experimental.ndarray_view(exemplar, slices=(slice(0, 14, 2), slice(None)))
    })
    exported.add_graph("increment", builder.compile())
    archive = tmp_path / "interleaved.tcm"
    exported.archive(str(archive))
    if ti.cfg.arch == ti.cpu:
        # The loader must not accept a changed stride qualification independently
        # of the compiled ABI. Reuse this small artifact; no second compilation.
        corrupted = tmp_path / "missing_layout_checksum.tcm"
        with zipfile.ZipFile(archive) as source, zipfile.ZipFile(corrupted, "w") as target:
            for name in source.namelist():
                payload = source.read(name)
                if name == "aot_metadata.json":
                    metadata = json.loads(payload)
                    assert metadata["kernel_runtime_affine_arrays"]
                    metadata["array_layout_sha256"] = ""
                    payload = json.dumps(metadata).encode()
                target.writestr(name, payload)
        with Runtime("cpu") as runtime:
            with pytest.raises(AotError, match="layout qualification checksum"):
                runtime.load_module(corrupted)
    with Runtime("cpu" if ti.cfg.arch == ti.cpu else "cuda") as runtime:
        module = runtime.load_module(archive)
        with runtime.dense_snode_group(schema.to_json(), {"N": 37}) as group:
            first = group.member([[0, 0]], "i32")
            neighbor = group.member([[0, i] for i in range(1, 4)], "f32", element_shape=(3,))
            neighbor.write(np.full(37 * 3 * 3, 17, dtype=np.float32))
            with pytest.raises(AotError, match="compiler-proven runtime stride"):
                module.kernel("increment")(first)
            module.kernel("affine")(first)
            np.testing.assert_array_equal(np.frombuffer(first.read(), np.int32), np.arange(37 * 3))
            bound = module.graph("increment").bind(dict(x=first))
            for operation in (bound, bound.prepare):
                with pytest.raises(AotError, match="Strided views require"):
                    operation()
            np.testing.assert_array_equal(np.frombuffer(neighbor.read(), np.float32), np.full(37 * 3 * 3, 17))
    producer.destroy()
