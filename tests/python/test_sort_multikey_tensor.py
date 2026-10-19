"""Lexicographic sorting moves packed tensor payloads as whole records."""

import numpy as np
import taichi_forge as ti
from tests import test_utils


def tensor_case(dtype, shape, n=17, shift=0):
    keys = [
        np.asarray([(i * 7 + shift) % 3 for i in range(n)], np.int32),
        np.asarray([(i * 5 + shift) % 4 for i in range(n)], np.int32),
    ]
    npdtype = {ti.f32: np.float32, ti.i64: np.int64, ti.i32: np.int32}[dtype]
    values = np.arange(n * np.prod(shape), dtype=npdtype).reshape((n, *shape)) + shift
    if dtype == ti.f32 and n > 1:
        values.view(np.uint32).reshape(-1)[:4] = [
            0x80000000,
            0x7FC00017,
            0xFFC00023,
            0x7F800000,
        ]
    elif dtype == ti.i64:
        values += np.int64(1 << 60)
    elif dtype == ti.i32:
        values.reshape(n, -1)[:, 0] = np.arange(n)
    order = sorted(range(n), key=lambda i: (keys[0][i], keys[1][i]))
    return keys, values, order


def tensor_array(dtype, shape, n):
    return (
        ti.Vector.ndarray(shape[0], dtype, n)
        if len(shape) == 1
        else ti.Matrix.ndarray(*shape, dtype, n)
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_multikey_tensor_payload_replay_and_workspace_reuse():
    workspace = ti.algorithms.SortWorkspace()
    method = "cpu_native" if ti.cfg.arch == ti.cpu else "auto"
    for dtype, shape in ((ti.f32, (3,)), (ti.f32, (1, 3)), (ti.i64, (2, 2))):
        keys = [ti.ndarray(ti.i32, 17), ti.ndarray(ti.i32, 17)]
        values = tensor_array(dtype, shape, 17)
        sequence = ti.algorithms.primitive_sequence().sort_by_key(
            keys, values, method=method, workspace=workspace
        )
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        graph = builder.compile()
        size = None
        for shift in (0, 1, 4):
            host_keys, host_values, order = tensor_case(dtype, shape, shift=shift)
            for owner, data in zip(keys, host_keys):
                owner.from_numpy(data)
            values.from_numpy(host_values)
            if shift == 0:
                sequence.run()
            else:
                graph.run({})
            for owner, data in zip(keys, host_keys):
                assert owner.to_numpy().tobytes() == data[order].tobytes()
            assert values.to_numpy().tobytes() == host_values[order].tobytes()
            if size is None:
                size = workspace.workspace_bytes_current
            assert workspace.workspace_bytes_current == size
        graph.close()
    workspace.clear()
    assert workspace.workspace_bytes_current == 0
    # Clearing retires the identity and temporary buffers; the same workspace
    # must also be usable for the next independent sort.
    host_keys, host_values, order = tensor_case(dtype, shape, shift=2)
    for owner, data in zip(keys, host_keys):
        owner.from_numpy(data)
    values.from_numpy(host_values)
    ti.algorithms.sort_by_key(keys, values, method=method, workspace=workspace)
    assert values.to_numpy().tobytes() == host_values[order].tobytes()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_multikey_tensor_singleton_keeps_bits_without_scratch():
    keys = [ti.ndarray(ti.i32, 1), ti.ndarray(ti.i32, 1)]
    values = ti.Vector.ndarray(3, ti.f32, 1)
    data = (
        np.asarray([0x80000000, 0x7FC00017, 0xFFC00023], np.uint32)
        .view(np.float32)
        .reshape(1, 3)
    )
    values.from_numpy(data)
    workspace = ti.algorithms.SortWorkspace()
    ti.algorithms.sort_by_key(
        keys,
        values,
        method="cpu_native" if ti.cfg.arch == ti.cpu else "auto",
        workspace=workspace,
    )
    assert values.to_numpy().tobytes() == data.tobytes()
    assert workspace.workspace_bytes_current == 0
