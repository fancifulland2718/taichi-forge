"""Native addon relocation, retained binding and feedback capture contracts."""

import ctypes
import os

import numpy as np
import pytest
import taichi_forge as ti

from taichi_forge.hardware._cub_segmented_capture import _CubSegmentedScanExecutable
from taichi_forge.hardware._cub_source_provider import load_cub_source_provider
from taichi_forge.lang import impl
from tests import test_utils


def _provider():
    path = os.environ.get("TI_FORGE_TEST_CUB_SOURCE_PROVIDER_MANIFEST")
    if not path:
        pytest.skip("an explicit reset-monoid source addon was not supplied")
    provider = load_cub_source_provider(path)
    if provider._library.info.features & 0x30 != 0x30:
        pytest.skip("the source addon predates reset-monoid scan")
    return provider


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("inclusive", (True, False))
@pytest.mark.parametrize("with_kernel", (True, False))
def test_addon_capture_retains_fixed_storage_and_preserves_feedback(
    inclusive, with_kernel, monkeypatch
):
    provider = _provider()
    n, capacity = 4097, 4104
    offsets = (0, 0, 1, 31, 32, 32, 4096, 4097)
    head_bits = np.zeros((n + 31) // 32, dtype=np.uint32)
    for begin, end in zip(offsets, offsets[1:]):
        if begin < end:
            head_bits[begin // 32] |= np.uint32(1 << (begin % 32))
    heads = ti.ndarray(ti.u32, len(head_bits))
    heads.from_numpy(head_bits)
    state, scratch = (ti.ndarray(ti.u32, capacity) for _ in range(2))
    initial = np.random.default_rng(613).integers(
        0, 2**32, size=capacity, dtype=np.uint32
    )
    state.from_numpy(initial)
    scratch.fill(77)
    expected = initial.copy()
    executables = tuple(
        _CubSegmentedScanExecutable(
            provider,
            source,
            heads,
            target,
            num_items=n,
            inclusive=inclusive,
            binding_prefix=f"scan{index}",
        )
        for index, (source, target) in enumerate(((state, scratch), (scratch, state)))
    )
    builder = ti.graph.GraphBuilder()
    for executable in executables:
        builder._append_native_executable(executable, admission="explicit")
    arguments = {}
    if with_kernel:
        observed = ti.ndarray(ti.u32, capacity)

        @ti.kernel
        def observe(
            source: ti.types.ndarray(dtype=ti.u32),
            output: ti.types.ndarray(dtype=ti.u32),
        ):
            for i in source:
                output[i] = source[i]

        builder.dispatch(
            observe,
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "observed_state", ti.u32, ndim=1),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "observed_output", ti.u32, ndim=1),
        )
        arguments = {"observed_state": state, "observed_output": observed}
    graph = builder.compile()
    assert len(graph._spec.fixed_runtime_args) == 8
    bound = graph.bind(arguments)
    np.testing.assert_array_equal(state.to_numpy(), initial)

    def unexpected(*args, **kwargs):
        pytest.fail("native addon Graph execution re-entered Python addon preparation")

    monkeypatch.setattr(provider._library, "execute", unexpected)
    monkeypatch.setattr(provider._library, "workspace_bytes", unexpected)
    monkeypatch.setattr(type(executables[0].recording), "execute", unexpected)
    monkeypatch.setattr(type(executables[0].plan), "_invocation", unexpected)
    for _ in range(4):
        graph.run(bound)
        for _pass in range(2):
            for begin, end in zip(offsets, offsets[1:]):
                segment = expected[begin:end].copy()
                if inclusive:
                    expected[begin:end] = np.cumsum(segment, dtype=np.uint32)
                elif end > begin:
                    expected[begin] = 0
                    expected[begin + 1 : end] = np.cumsum(segment[:-1], dtype=np.uint32)
        np.testing.assert_array_equal(state.to_numpy(), expected)
        np.testing.assert_array_equal(
            scratch.to_numpy()[n:], np.full(capacity - n, 77, dtype=np.uint32)
        )
        if with_kernel:
            np.testing.assert_array_equal(observed.to_numpy(), expected)
    assert graph._graph_stats[0]["last_path"] == "cuda_exact_replay"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_addon_capture_rejects_conflicting_fixed_binding_before_append():
    provider = _provider()
    values, output = (ti.ndarray(ti.u32, 32) for _ in range(2))
    heads = ti.ndarray(ti.u32, 1)
    first, second = (
        _CubSegmentedScanExecutable(
            provider,
            values,
            heads,
            output,
            num_items=32,
            inclusive=True,
            binding_prefix="same",
        )
        for _ in range(2)
    )
    builder = ti.graph.GraphBuilder()
    builder._append_native_executable(first, admission="explicit")
    count = builder._dispatch_count
    with pytest.raises(RuntimeError, match="conflicting fixed binding"):
        builder._append_native_executable(second, admission="explicit")
    assert builder._dispatch_count == count


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize(
    "offsets,stream_offset,counts", (((8,), 8, (1,)), ((32,), 8, (1,)), ((0,), 8, ()))
)
def test_native_addon_relocation_metadata_rejected_without_calling_callback(
    offsets, stream_offset, counts
):
    # A real, retained function pointer must never be invoked for invalid
    # metadata. This test does not require a Toolkit or load an external DLL.
    calls = []
    callback = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p)(
        lambda _: calls.append(True) or 0
    )
    builder = ti.graph.GraphBuilder()
    argument = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "input", ti.u32, ndim=1)
    with pytest.raises(RuntimeError, match="relocations|metadata"):
        builder._runtime_graph_builder._dispatch_cuda_addon_capture_recipe(
            impl.get_runtime().prog,
            ctypes.cast(callback, ctypes.c_void_p).value,
            bytes(24),
            stream_offset,
            (argument,),
            offsets,
            counts,
            (False,),
        )
    assert calls == []
