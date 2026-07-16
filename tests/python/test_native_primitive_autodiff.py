import os

import pytest
import numpy as np

import taichi_forge as ti
import taichi_forge.algorithms._autodiff as _autodiff
from taichi_forge.lang import impl
from tests import test_utils


def _require_cuda_capability(name, description):
    prog = impl.get_runtime().prog
    available = hasattr(prog, name) and getattr(prog, name)()
    if available:
        return
    message = f"CUDA {description} is unavailable. Native Tape coverage requires " "the corresponding runtime provider."
    if os.environ.get("TI_REQUIRE_CUDA_NATIVE_AD_CAPABILITIES") == "1":
        pytest.fail(message)
    pytest.skip(message)


def _native_copy_method_for_arch():
    arch = impl.current_cfg().arch
    if arch == ti.cuda:
        # Forward device copy kernels are available without the CUDA Toolkit,
        # but their Tape records still need native add/merge accumulation.
        # Treat the forward/backward pair as one capability contract.
        _require_cuda_capability(
            "cuda_device_add_merge_available", "native AD accumulation"
        )
        return "cuda_device"
    if arch == ti.vulkan:
        return "vulkan_native"
    return "cpu_native"


def _native_reduce_method_for_arch():
    arch = impl.current_cfg().arch
    if arch == ti.cuda:
        _require_cuda_capability("cuda_device_reduce_available", "native reduce")
        return "cuda_device"
    if arch == ti.vulkan:
        prog = impl.get_runtime().prog
        if not (hasattr(prog, "vulkan_reduce_available") and prog.vulkan_reduce_available()):
            pytest.skip("Vulkan native reduce is unavailable.")
        return "vulkan_native"
    return "cpu_native"


def _require_native_scatter_add_for_arch():
    arch = impl.current_cfg().arch
    prog = impl.get_runtime().prog
    if arch == ti.cuda:
        _require_cuda_capability(
            "cuda_device_scatter_add_available", "native scatter-add"
        )
        _require_cuda_capability(
            "cuda_device_add_merge_available", "native AD accumulation"
        )
    elif arch == ti.vulkan:
        if not (
            hasattr(prog, "vulkan_scatter_add_available")
            and prog.vulkan_scatter_add_available()
        ):
            pytest.skip("Vulkan native scatter-add is unavailable.")
        if not prog.vulkan_scatter_add_value_type_available(1):
            pytest.skip("Vulkan f32 scatter-add atomics are unavailable.")


def _require_native_grouped_reduce_for_arch():
    arch = impl.current_cfg().arch
    prog = impl.get_runtime().prog
    if arch == ti.cuda:
        _require_cuda_capability(
            "cuda_device_grouped_reduce_available", "native grouped-reduce"
        )
    elif arch == ti.vulkan:
        if not (
            hasattr(prog, "vulkan_grouped_reduce_available")
            and prog.vulkan_grouped_reduce_available()
        ):
            pytest.skip("Vulkan native grouped-reduce is unavailable.")
        if not prog.vulkan_grouped_reduce_atomic_value_type_available(1):
            pytest.skip("Vulkan f32 grouped-reduce atomics are unavailable.")


def _require_native_scan_for_arch():
    if impl.current_cfg().arch == ti.cuda:
        _require_cuda_capability("cuda_device_scan_available", "native scan")


def test_native_autodiff_no_tape_keeps_method(monkeypatch):
    monkeypatch.setattr(_autodiff, "is_tape_active", lambda: False)

    assert _autodiff.native_autodiff_method("transform", "auto") == "auto"
    assert (
        _autodiff.native_autodiff_method("transform", "vulkan_native")
        == "vulkan_native"
    )


def test_native_autodiff_auto_uses_kernel_fallback_under_tape(monkeypatch):
    monkeypatch.setattr(_autodiff, "is_tape_active", lambda: True)

    assert _autodiff.native_autodiff_method("transform", "auto") == "kernel"
    assert _autodiff.native_autodiff_method("gather", "auto") == "kernel"
    assert _autodiff.native_autodiff_method("scatter", "auto") == "kernel"
    assert _autodiff.native_autodiff_method("scan", "auto") == "kernel"
    assert _autodiff.native_autodiff_method("scatter_add", "auto") == "kernel"
    assert (
        _autodiff.native_autodiff_method("grouped_reduce", "auto", op="sum")
        == "kernel"
    )
    assert (
        _autodiff.native_autodiff_method("reduce", "auto", op="sum")
        == "field_atomic"
    )


def test_native_autodiff_explicit_native_rejected_under_tape(monkeypatch):
    monkeypatch.setattr(_autodiff, "is_tape_active", lambda: True)

    with pytest.raises(RuntimeError, match="method='cuda_device'.*ti.ad.Tape"):
        _autodiff.native_autodiff_method("transform", "cuda_device")
    with pytest.raises(RuntimeError, match="method='vulkan_native'.*ti.ad.Tape"):
        _autodiff.native_autodiff_method("gather", "vulkan_native")
    with pytest.raises(RuntimeError, match="method='cpu_native'.*ti.ad.Tape"):
        _autodiff.native_autodiff_method("scatter", "cpu_native")
    with pytest.raises(RuntimeError, match="method='cuda_cub'.*ti.ad.Tape"):
        _autodiff.native_autodiff_method("scan", "cuda_cub")
    with pytest.raises(RuntimeError, match="method='cuda_two_level'.*ti.ad.Tape"):
        _autodiff.native_autodiff_method("scatter_add", "cuda_two_level")


def test_native_autodiff_rejects_non_differentiable_native_reduce_op(monkeypatch):
    monkeypatch.setattr(_autodiff, "is_tape_active", lambda: True)

    with pytest.raises(RuntimeError, match="op='max'.*no native autodiff policy"):
        _autodiff.native_autodiff_method("reduce", "auto", op="max")
    assert (
        _autodiff.native_autodiff_method("reduce", "field_atomic", op="max")
        == "field_atomic"
    )


def test_native_ad_record_and_bridge_insert(monkeypatch):
    calls = []

    class FakeTape:
        def __init__(self):
            self.records = []

        def insert_native(self, record):
            self.records.append(record)

    def backward(a, *, scale):
        calls.append(a * scale)

    tape = FakeTape()
    bridge = _autodiff.NativePrimitiveADBridge()
    bridge.register_backward("transform", "cuda_device", backward)
    monkeypatch.setattr(_autodiff, "active_tape", lambda: tape)

    assert bridge.record("transform", "cuda_device", 3, scale=2)
    assert len(tape.records) == 1
    tape.records[0].grad()
    assert calls == [6]


def test_native_ad_record_suppresses_nested_tape_routing(monkeypatch):
    states = []

    class FakeTape:
        def __init__(self):
            self.records = []

        def insert_native(self, record):
            self.records.append(record)

    def backward():
        states.append(_autodiff.is_tape_active())

    tape = FakeTape()
    monkeypatch.setattr(_autodiff, "active_tape", lambda: tape)
    assert _autodiff.is_tape_active()
    assert _autodiff.native_primitive_ad.record_callable(
        "transform", "cpu_native", backward
    )
    tape.records[0].grad()
    assert states == [False]
    assert _autodiff.is_tape_active()


@test_utils.test(arch=[ti.cpu])
def test_transform_auto_preserves_tape_gradients():
    x = ti.field(ti.f32, shape=4, needs_grad=True)
    y = ti.field(ti.f32, shape=4, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def init():
        for i in x:
            x[i] = i + 1

    @ti.kernel
    def sum_y():
        for i in y:
            loss[None] += y[i]

    init()
    with ti.ad.Tape(loss):
        ti.algorithms.experimental_transform(
            x, y, scale=3.0, bias=1.0, method="auto"
        )
        sum_y()

    assert loss[None] == pytest.approx(34.0)
    for i in range(4):
        assert x.grad[i] == pytest.approx(3.0)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_transform_cpu_native_tape_gradients_ndarray():
    n = 8
    x = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    y = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    x.from_numpy(np.arange(n, dtype=np.float32))

    @ti.kernel
    def sum_y(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            loss[None] += arr[i]

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_transform(
            x, y, scale=2.5, bias=1.0, method=_native_copy_method_for_arch()
        )
        sum_y(y)

    grads = x.grad.to_numpy()
    for i in range(n):
        assert grads[i] == pytest.approx(2.5)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_gather_cpu_native_tape_gradients_ndarray():
    _require_native_scatter_add_for_arch()

    src = ti.ndarray(ti.f32, shape=5, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=6)
    dst = ti.ndarray(ti.f32, shape=6, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    src.from_numpy(np.arange(5, dtype=np.float32))
    indices.from_numpy(np.array([0, 1, 2, 2, 4, 1], dtype=np.int32))

    @ti.kernel
    def sum_dst(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(6):
            loss[None] += arr[i]

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_gather(
            src, indices, dst, method=_native_copy_method_for_arch()
        )
        sum_dst(dst)

    np.testing.assert_allclose(
        src.grad.to_numpy(), np.array([1, 2, 2, 0, 1], dtype=np.float32)
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_scatter_add_cpu_native_tape_gradients_ndarray():
    _require_native_scatter_add_for_arch()

    src = ti.ndarray(ti.f32, shape=6, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=6)
    dst = ti.ndarray(ti.f32, shape=5, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    src.from_numpy(np.ones(6, dtype=np.float32))
    indices.from_numpy(np.array([0, 1, 2, 2, 4, 1], dtype=np.int32))
    dst.fill(0)

    @ti.kernel
    def sum_dst(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(5):
            loss[None] += arr[i]

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_scatter_add(
            src, indices, dst, method=_native_copy_method_for_arch()
        )
        sum_dst(dst)

    np.testing.assert_allclose(src.grad.to_numpy(), np.ones(6, dtype=np.float32))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scatter_cpu_native_tape_gradients_ndarray():
    src = ti.ndarray(ti.f32, shape=5, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=5)
    dst = ti.ndarray(ti.f32, shape=7, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    src.from_numpy(np.arange(5, dtype=np.float32))
    indices.from_numpy(np.array([2, 5, 1, 6, 3], dtype=np.int32))
    dst.fill(0)

    @ti.kernel
    def weighted_sum(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        loss[None] += arr[1] * 2.0
        loss[None] += arr[2] * 3.0
        loss[None] += arr[3] * 5.0
        loss[None] += arr[5] * 7.0
        loss[None] += arr[6] * 11.0

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_scatter(
            src, indices, dst, method=_native_copy_method_for_arch()
        )
        weighted_sum(dst)

    np.testing.assert_allclose(
        src.grad.to_numpy(), np.array([3, 7, 2, 11, 5], dtype=np.float32)
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scan_native_tape_gradients_ndarray():
    _require_native_scan_for_arch()
    n = 7
    values = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    values.from_numpy(np.arange(1, n + 1, dtype=np.float32))
    scanner = ti.algorithms.PrefixSumExecutor(n)

    @ti.kernel
    def sum_values(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            loss[None] += arr[i]

    with ti.ad.Tape(loss):
        scanner.run(values)
        sum_values(values)

    np.testing.assert_allclose(
        values.grad.to_numpy(),
        np.arange(n, 0, -1, dtype=np.float32),
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_reduce_cpu_native_tape_gradients_ndarray():
    values = ti.ndarray(ti.f32, shape=7, needs_grad=True)
    output = ti.ndarray(ti.f32, shape=1, needs_grad=True)
    values.from_numpy(np.arange(7, dtype=np.float32))

    with ti.ad.Tape(output):
        ti.algorithms.experimental_reduce(
            values, output, op="sum", method=_native_reduce_method_for_arch()
        )

    np.testing.assert_allclose(values.grad.to_numpy(), np.ones(7, dtype=np.float32))


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_transform_vulkan_native_tape_gradients_ndarray_chain():
    n = 8
    x = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    y = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    loss = ti.ndarray(ti.f32, shape=1, needs_grad=True)
    x.from_numpy(np.arange(n, dtype=np.float32))

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_transform(
            x, y, scale=2.5, bias=1.0, method="vulkan_native"
        )
        ti.algorithms.experimental_reduce(y, loss, op="sum", method="vulkan_native")

    np.testing.assert_allclose(x.grad.to_numpy(), np.full(n, 2.5, dtype=np.float32))


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_gather_vulkan_native_tape_gradients_ndarray_chain():
    _require_native_scatter_add_for_arch()

    src = ti.ndarray(ti.f32, shape=5, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=6)
    dst = ti.ndarray(ti.f32, shape=6, needs_grad=True)
    loss = ti.ndarray(ti.f32, shape=1, needs_grad=True)
    src.from_numpy(np.arange(5, dtype=np.float32))
    indices.from_numpy(np.array([0, 1, 2, 2, 4, 1], dtype=np.int32))

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_gather(src, indices, dst, method="vulkan_native")
        ti.algorithms.experimental_reduce(dst, loss, op="sum", method="vulkan_native")

    np.testing.assert_allclose(
        src.grad.to_numpy(), np.array([1, 2, 2, 0, 1], dtype=np.float32)
    )


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scatter_add_vulkan_native_tape_gradients_ndarray_chain():
    _require_native_scatter_add_for_arch()

    src = ti.ndarray(ti.f32, shape=6, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=6)
    dst = ti.ndarray(ti.f32, shape=5, needs_grad=True)
    loss = ti.ndarray(ti.f32, shape=1, needs_grad=True)
    src.from_numpy(np.ones(6, dtype=np.float32))
    indices.from_numpy(np.array([0, 1, 2, 2, 4, 1], dtype=np.int32))
    dst.fill(0)

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_scatter_add(
            src, indices, dst, method="vulkan_native"
        )
        ti.algorithms.experimental_reduce(dst, loss, op="sum", method="vulkan_native")

    np.testing.assert_allclose(src.grad.to_numpy(), np.ones(6, dtype=np.float32))


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scatter_vulkan_native_tape_gradients_ndarray_chain():
    src = ti.ndarray(ti.f32, shape=5, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=5)
    dst = ti.ndarray(ti.f32, shape=7, needs_grad=True)
    loss = ti.ndarray(ti.f32, shape=1, needs_grad=True)
    src.from_numpy(np.arange(5, dtype=np.float32))
    indices.from_numpy(np.array([2, 5, 1, 6, 3], dtype=np.int32))
    dst.fill(0)

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_scatter(src, indices, dst, method="vulkan_native")
        ti.algorithms.experimental_reduce(dst, loss, op="sum", method="vulkan_native")

    np.testing.assert_allclose(src.grad.to_numpy(), np.ones(5, dtype=np.float32))


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scan_vulkan_native_tape_gradients_ndarray_chain():
    n = 7
    values = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    loss = ti.ndarray(ti.f32, shape=1, needs_grad=True)
    values.from_numpy(np.arange(1, n + 1, dtype=np.float32))
    scanner = ti.algorithms.PrefixSumExecutor(n)

    with ti.ad.Tape(loss):
        scanner.run(values)
        ti.algorithms.experimental_reduce(values, loss, op="sum", method="vulkan_native")

    np.testing.assert_allclose(
        values.grad.to_numpy(), np.arange(n, 0, -1, dtype=np.float32)
    )


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_grouped_reduce_vulkan_native_tape_gradients_ndarray_chain():
    _require_native_grouped_reduce_for_arch()

    keys = ti.ndarray(ti.i32, shape=6)
    values = ti.ndarray(ti.f32, shape=6, needs_grad=True)
    output = ti.ndarray(ti.f32, shape=3, needs_grad=True)
    loss = ti.ndarray(ti.f32, shape=1, needs_grad=True)
    keys.from_numpy(np.array([0, 1, 2, 2, 1, 0], dtype=np.int32))
    values.from_numpy(np.ones(6, dtype=np.float32))

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_grouped_reduce(
            keys, values, output, op="sum", method="vulkan_native"
        )
        ti.algorithms.experimental_reduce(output, loss, op="sum", method="vulkan_native")

    np.testing.assert_allclose(values.grad.to_numpy(), np.ones(6, dtype=np.float32))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_reduce_native_tape_gradients_dense_field():
    n = 9
    values = ti.field(ti.f32, shape=n, needs_grad=True)
    output = ti.field(ti.f32, shape=(), needs_grad=True)
    values.from_numpy(np.arange(n, dtype=np.float32))

    with ti.ad.Tape(output):
        ti.algorithms.experimental_reduce(
            values, output, op="sum", method=_native_reduce_method_for_arch()
        )

    actual = np.array([values.grad[i] for i in range(n)], dtype=np.float32)
    np.testing.assert_allclose(actual, np.ones(n, dtype=np.float32))


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_grouped_reduce_cpu_native_tape_gradients_ndarray():
    _require_native_grouped_reduce_for_arch()

    keys = ti.ndarray(ti.i32, shape=6)
    values = ti.ndarray(ti.f32, shape=6, needs_grad=True)
    output = ti.ndarray(ti.f32, shape=3, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    keys.from_numpy(np.array([0, 1, 2, 2, 1, 0], dtype=np.int32))
    values.from_numpy(np.ones(6, dtype=np.float32))

    @ti.kernel
    def weighted_sum(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        loss[None] += arr[0] * 2.0
        loss[None] += arr[1] * 3.0
        loss[None] += arr[2] * 5.0

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_grouped_reduce(
            keys, values, output, op="sum", method=_native_copy_method_for_arch()
        )
        weighted_sum(output)

    np.testing.assert_allclose(
        values.grad.to_numpy(), np.array([2, 3, 5, 5, 3, 2], dtype=np.float32)
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_transform_native_tape_gradients_dense_field():
    n = 8
    x = ti.field(ti.f32, shape=n, needs_grad=True)
    y = ti.field(ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    x.from_numpy(np.arange(n, dtype=np.float32))

    @ti.kernel
    def sum_y():
        for i in y:
            loss[None] += y[i]

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_transform(
            x, y, scale=2.5, bias=1.0, method=_native_copy_method_for_arch()
        )
        sum_y()

    actual = np.array([x.grad[i] for i in range(n)], dtype=np.float32)
    np.testing.assert_allclose(actual, np.full(n, 2.5, dtype=np.float32))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_transform_native_tape_gradients_dense_matrix_field():
    n = 8
    x = ti.Vector.field(2, ti.f32, shape=n, needs_grad=True)
    y = ti.Vector.field(2, ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    x.from_numpy(np.arange(n * 2, dtype=np.float32).reshape(n, 2))

    @ti.kernel
    def weighted_sum():
        for i in y:
            loss[None] += y[i][0] * 2.0 + y[i][1] * 3.0

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_transform(
            x, y, scale=2.5, bias=1.0, method=_native_copy_method_for_arch()
        )
        weighted_sum()

    expected = np.tile(np.array([5.0, 7.5], dtype=np.float32), (n, 1))
    np.testing.assert_allclose(x.grad.to_numpy(), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_gather_native_tape_gradients_dense_field():
    _require_native_scatter_add_for_arch()

    src = ti.field(ti.f32, shape=5, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=6)
    dst = ti.field(ti.f32, shape=6, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    src.from_numpy(np.arange(5, dtype=np.float32))
    indices.from_numpy(np.array([0, 1, 2, 2, 4, 1], dtype=np.int32))

    @ti.kernel
    def sum_dst():
        for i in dst:
            loss[None] += dst[i]

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_gather(
            src, indices, dst, method=_native_copy_method_for_arch()
        )
        sum_dst()

    actual = np.array([src.grad[i] for i in range(5)], dtype=np.float32)
    np.testing.assert_allclose(actual, np.array([1, 2, 2, 0, 1], dtype=np.float32))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_gather_native_tape_gradients_dense_matrix_field():
    _require_native_scatter_add_for_arch()

    src = ti.Vector.field(2, ti.f32, shape=5, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=6)
    dst = ti.Vector.field(2, ti.f32, shape=6, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    src.from_numpy(np.arange(10, dtype=np.float32).reshape(5, 2))
    index_data = np.array([0, 1, 2, 2, 4, 1], dtype=np.int32)
    indices.from_numpy(index_data)

    @ti.kernel
    def weighted_sum():
        for i in dst:
            loss[None] += dst[i][0] * ti.cast(i + 1, ti.f32)
            loss[None] += dst[i][1] * ti.cast(i + 2, ti.f32)

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_gather(
            src, indices, dst, method=_native_copy_method_for_arch()
        )
        weighted_sum()

    expected = np.zeros((5, 2), dtype=np.float32)
    for i, target in enumerate(index_data):
        expected[target, 0] += i + 1
        expected[target, 1] += i + 2
    np.testing.assert_allclose(src.grad.to_numpy(), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scatter_add_native_tape_gradients_dense_field():
    _require_native_scatter_add_for_arch()

    src = ti.field(ti.f32, shape=6, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=6)
    dst = ti.field(ti.f32, shape=5, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    src.from_numpy(np.ones(6, dtype=np.float32))
    indices.from_numpy(np.array([0, 1, 2, 2, 4, 1], dtype=np.int32))
    dst.fill(0)

    @ti.kernel
    def sum_dst():
        for i in dst:
            loss[None] += dst[i]

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_scatter_add(
            src, indices, dst, method=_native_copy_method_for_arch()
        )
        sum_dst()

    actual = np.array([src.grad[i] for i in range(6)], dtype=np.float32)
    np.testing.assert_allclose(actual, np.ones(6, dtype=np.float32))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scatter_add_native_tape_gradients_dense_matrix_field():
    _require_native_scatter_add_for_arch()

    src = ti.Vector.field(2, ti.f32, shape=6, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=6)
    dst = ti.Vector.field(2, ti.f32, shape=5, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    src.from_numpy(np.ones((6, 2), dtype=np.float32))
    index_data = np.array([0, 1, 2, 2, 4, 1], dtype=np.int32)
    indices.from_numpy(index_data)
    dst.fill(0)

    @ti.kernel
    def weighted_sum():
        for i in dst:
            loss[None] += dst[i][0] * ti.cast(i + 1, ti.f32)
            loss[None] += dst[i][1] * ti.cast(i + 2, ti.f32)

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_scatter_add(
            src, indices, dst, method=_native_copy_method_for_arch()
        )
        weighted_sum()

    expected = np.zeros((6, 2), dtype=np.float32)
    for i, target in enumerate(index_data):
        expected[i, 0] = target + 1
        expected[i, 1] = target + 2
    np.testing.assert_allclose(src.grad.to_numpy(), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scatter_native_tape_gradients_dense_field():
    src = ti.field(ti.f32, shape=5, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=5)
    dst = ti.field(ti.f32, shape=7, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    src.from_numpy(np.arange(5, dtype=np.float32))
    indices.from_numpy(np.array([2, 5, 1, 6, 3], dtype=np.int32))
    dst.fill(0)

    @ti.kernel
    def weighted_sum():
        loss[None] += dst[1] * 2.0
        loss[None] += dst[2] * 3.0
        loss[None] += dst[3] * 5.0
        loss[None] += dst[5] * 7.0
        loss[None] += dst[6] * 11.0

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_scatter(
            src, indices, dst, method=_native_copy_method_for_arch()
        )
        weighted_sum()

    actual = np.array([src.grad[i] for i in range(5)], dtype=np.float32)
    np.testing.assert_allclose(actual, np.array([3, 7, 2, 11, 5], dtype=np.float32))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scatter_native_tape_gradients_dense_matrix_field():
    src = ti.Vector.field(2, ti.f32, shape=5, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=5)
    dst = ti.Vector.field(2, ti.f32, shape=7, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    src.from_numpy(np.arange(10, dtype=np.float32).reshape(5, 2))
    index_data = np.array([2, 5, 1, 6, 3], dtype=np.int32)
    indices.from_numpy(index_data)
    dst.fill(0)

    @ti.kernel
    def weighted_sum():
        for i in dst:
            loss[None] += dst[i][0] * ti.cast(i + 1, ti.f32)
            loss[None] += dst[i][1] * ti.cast(i + 2, ti.f32)

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_scatter(
            src, indices, dst, method=_native_copy_method_for_arch()
        )
        weighted_sum()

    expected = np.zeros((5, 2), dtype=np.float32)
    for i, target in enumerate(index_data):
        expected[i, 0] = target + 1
        expected[i, 1] = target + 2
    np.testing.assert_allclose(src.grad.to_numpy(), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scan_native_tape_gradients_dense_field():
    _require_native_scan_for_arch()
    n = 7
    values = ti.field(ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    values.from_numpy(np.arange(1, n + 1, dtype=np.float32))
    scanner = ti.algorithms.PrefixSumExecutor(n)

    @ti.kernel
    def sum_values():
        for i in values:
            loss[None] += values[i]

    with ti.ad.Tape(loss):
        scanner.run(values)
        sum_values()

    actual = np.array([values.grad[i] for i in range(n)], dtype=np.float32)
    np.testing.assert_allclose(actual, np.arange(n, 0, -1, dtype=np.float32))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scan_native_tape_gradients_dense_matrix_field():
    _require_native_scan_for_arch()
    n = 7
    values = ti.Vector.field(2, ti.f32, shape=n, needs_grad=True)
    weight_field = ti.Vector.field(2, ti.f32, shape=n)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    initial = np.arange(1, n * 2 + 1, dtype=np.float32).reshape(n, 2)
    weights = np.stack(
        (
            np.arange(1, n + 1, dtype=np.float32),
            np.arange(2, n + 2, dtype=np.float32),
        ),
        axis=1,
    )
    values.from_numpy(initial)
    weight_field.from_numpy(weights)
    scanner = ti.algorithms.PrefixSumExecutor(n)

    @ti.kernel
    def weighted_sum():
        for i in values:
            loss[None] += values[i][0] * weight_field[i][0]
            loss[None] += values[i][1] * weight_field[i][1]

    with ti.ad.Tape(loss):
        scanner.run(values)
        weighted_sum()

    expected = np.flip(np.cumsum(np.flip(weights, axis=0), axis=0), axis=0)
    np.testing.assert_allclose(values.grad.to_numpy(), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_grouped_reduce_native_tape_gradients_dense_field():
    _require_native_grouped_reduce_for_arch()

    keys = ti.field(ti.i32, shape=6)
    values = ti.field(ti.f32, shape=6, needs_grad=True)
    output = ti.field(ti.f32, shape=3, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    keys.from_numpy(np.array([0, 1, 2, 2, 1, 0], dtype=np.int32))
    values.from_numpy(np.ones(6, dtype=np.float32))

    @ti.kernel
    def weighted_sum():
        loss[None] += output[0] * 2.0
        loss[None] += output[1] * 3.0
        loss[None] += output[2] * 5.0

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_grouped_reduce(
            keys, values, output, op="sum", method=_native_copy_method_for_arch()
        )
        weighted_sum()

    actual = np.array([values.grad[i] for i in range(6)], dtype=np.float32)
    np.testing.assert_allclose(
        actual, np.array([2, 3, 5, 5, 3, 2], dtype=np.float32)
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_grouped_reduce_native_tape_gradients_dense_matrix_field():
    _require_native_grouped_reduce_for_arch()

    key_data = np.array([0, 1, 2, 2, 1, 0], dtype=np.int32)
    expected = np.zeros((6, 2), dtype=np.float32)
    for i, key in enumerate(key_data):
        expected[i, 0] = key + 2
        expected[i, 1] = key + 5

    for keys_as_field in (False, True):
        keys = ti.field(ti.i32, shape=6) if keys_as_field else ti.ndarray(ti.i32, shape=6)
        values = ti.Vector.field(2, ti.f32, shape=6, needs_grad=True)
        output = ti.Vector.field(2, ti.f32, shape=3, needs_grad=True)
        loss = ti.field(ti.f32, shape=(), needs_grad=True)
        keys.from_numpy(key_data)
        values.from_numpy(np.ones((6, 2), dtype=np.float32))

        @ti.kernel
        def weighted_sum():
            for i in output:
                loss[None] += output[i][0] * ti.cast(i + 2, ti.f32)
                loss[None] += output[i][1] * ti.cast(i + 5, ti.f32)

        with ti.ad.Tape(loss):
            ti.algorithms.experimental_grouped_reduce(
                keys, values, output, op="sum", method=_native_copy_method_for_arch()
            )
            weighted_sum()

        np.testing.assert_allclose(values.grad.to_numpy(), expected)
