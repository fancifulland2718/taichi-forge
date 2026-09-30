import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl


@test_utils.test(arch=ti.cuda)
def test_global_thread_idx():
    n = 128
    x = ti.field(ti.i32, shape=n)

    @ti.kernel
    def func():
        for i in range(n):
            tid = ti.global_thread_idx()
            x[tid] = tid

    func()
    assert np.arange(n).sum() == x.to_numpy().sum()


def _compile_vulkan_before_launch(kernel, *args):
    # Detect invalid SPIR-V before handing it to a driver. The original bug
    # produced ID zero and could crash during pipeline creation after opt failed.
    key = kernel._primal.ensure_compiled(*args)
    program = impl.get_runtime().prog
    program.compile_kernel(program.config(), program.get_device_caps(), kernel._primal.compiled_kernels[key])
    stats = _ti_core.get_last_vulkan_spv_stats()
    assert stats
    assert all(not task["opt_run"] or task["opt_ok"] for task in stats)


@pytest.mark.parametrize("block_dim,start,count", [(32, 0, 65), (64, 17, 257), (128, -13, 513)])
@test_utils.test(
    arch=[ti.cuda, ti.amdgpu, ti.vulkan], offline_cache=False, vulkan_spv_stats=True, vulkan_spv_stats_filter="all"
)
def test_simt_global_thread_idx_across_blocks(block_dim, start, count):
    values = ti.ndarray(ti.i32, count)

    @ti.kernel
    def run(out: ti.types.ndarray(ti.i32, ndim=1)):
        ti.loop_config(block_dim=block_dim)
        for i in range(start, start + count):
            out[i - start] = ti.simt.block.global_thread_idx()

    if impl.current_cfg().arch == ti.vulkan:
        _compile_vulkan_before_launch(run, values)
    run(values)
    # Thread indices are zero based within the launch, independent of loop start.
    np.testing.assert_array_equal(values.to_numpy(), np.arange(count))


@pytest.mark.parametrize("optimization_level", [0, 3])
@test_utils.test(arch=ti.vulkan, offline_cache=False, vulkan_spv_stats=True, vulkan_spv_stats_filter="all")
def test_vulkan_global_thread_idx_internal_and_graph(optimization_level):
    ti.cfg.external_optimization_level = optimization_level
    count, block_dim, start = 257, 64, 19
    values = ti.ndarray(ti.i32, (count, 3))

    @ti.kernel
    def run(out: ti.types.ndarray(ti.i32, ndim=2)):
        ti.loop_config(block_dim=block_dim)
        for i in range(start, start + count):
            out[i - start, 0] = ti.simt.block.global_thread_idx()
            out[i - start, 1] = impl.call_internal("vkGlobalThreadIdx", with_runtime_context=False)
            out[i - start, 2] = ti.simt.block.thread_idx()

    _compile_vulkan_before_launch(run, values)
    expected = np.stack((np.arange(count), np.arange(count), np.arange(count) % block_dim), axis=1)
    run(values)
    np.testing.assert_array_equal(values.to_numpy(), expected)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(run, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=2))
    graph = builder.compile()
    try:
        bound = graph.bind({"out": values})
        for _ in range(3):
            values.fill(-1)
            graph.submit(bound).wait()
            np.testing.assert_array_equal(values.to_numpy(), expected)
    finally:
        graph.close()


@pytest.mark.parametrize("optimization_level", [0, 3])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_unimplemented_internal_op_fails_before_shader_creation(optimization_level):
    ti.cfg.external_optimization_level = optimization_level
    out = ti.ndarray(ti.i32, 1)

    @ti.kernel
    def unsupported(values: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(1):
            values[i] = impl.call_internal("linear_thread_idx", with_runtime_context=False)

    with pytest.raises(
        RuntimeError, match="Internal operation 'linear_thread_idx' is not implemented by SPIR-V codegen"
    ):
        unsupported(out)

    # A rejected intrinsic must leave the runtime usable for subsequent work.
    @ti.kernel
    def supported(values: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(1):
            values[i] = ti.simt.block.global_thread_idx() + 7

    supported(out)
    np.testing.assert_array_equal(out.to_numpy(), [7])


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_simt_global_thread_idx_rejects_unsupported_arch():
    @ti.kernel
    def run() -> ti.i32:
        return ti.simt.block.global_thread_idx()

    with pytest.raises(ti.TaichiCompilationError, match="global_thread_idx is not supported for arch"):
        run()
