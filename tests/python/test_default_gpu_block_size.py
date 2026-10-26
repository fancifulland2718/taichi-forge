"""Static default selection preserves explicit and deployed execution contracts."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


def range_tasks(kernel, *args):
    return [task for task in kernel.task_manifest(*args) if task.task_type == "range_for"]


@pytest.mark.parametrize("count", [17, 1024])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False,
                 advanced_optimization=False)
def test_static_default_jit_graph_and_aot(tmp_path, count):
    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(-3, count - 3):
            out[i + 3] = i * 7

    out = ti.ndarray(ti.i32, count)
    expected = np.arange(-3, count - 3, dtype=np.int32) * 7
    fill(out)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    if ti.lang.impl.current_cfg().arch != ti.cpu:
        task, = range_tasks(fill, out)
        assert task.selected_block_size in (32, 64, 128)
        if count == 17:
            subgroup = ti.lang.impl.current_cfg().gpu_subgroup_size or 32
            assert task.selected_block_size == max(32, min(128, subgroup))
        assert not task.source_block_size_explicit

    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1))
    graph = builder.compile()
    out.fill(-1)
    graph.run(dict(out=out))
    np.testing.assert_array_equal(out.to_numpy(), expected)

    module = ti.aot.Module()
    module.add_execution_plan("fill", graph)
    archive = tmp_path / "static-block.tcm"
    module.archive(str(archive))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(archive) as loaded:
        target = runtime.ndarray("i32", (count,))
        try:
            with loaded.prepare_execution("fill", dict(out=target)) as execute:
                execute()
            np.testing.assert_array_equal(np.frombuffer(target.read(), np.int32), expected)
        finally:
            target.close()
    graph.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False,
                 advanced_optimization=False)
def test_static_default_preserves_explicit_dynamic_and_atomic_geometry():
    @ti.kernel
    def explicit(out: ti.types.ndarray(ti.i32, 1)):
        ti.loop_config(block_dim=128)
        for i in range(17):
            out[i] = i

    @ti.kernel
    def dynamic(out: ti.types.ndarray(ti.i32, 1), n: ti.i32):
        for i in range(n):
            out[i] = i

    @ti.kernel
    def reduce(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(17):
            ti.atomic_add(out[0], i)

    out = ti.ndarray(ti.i32, 129)
    for kernel, args in ((explicit, (out,)), (dynamic, (out, 17)), (reduce, (out,))):
        out.fill(0)
        kernel(*args)
        assert range_tasks(kernel, *args)[0].selected_block_size == 128
    assert out.to_numpy()[0] == sum(range(17))
    dynamic(out, 129)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(129))


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False,
                 default_gpu_block_dim=256)
def test_static_default_preserves_custom_global_default():
    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(17):
            out[i] = i
    out = ti.ndarray(ti.i32, 17)
    fill(out)
    assert range_tasks(fill, out)[0].selected_block_size == 256
    np.testing.assert_array_equal(out.to_numpy(), np.arange(17))


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_static_default_spreads_multiple_blocks_and_preserves_policy():
    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(1024):
            out[i] = i * 3 + 1

    out = ti.ndarray(ti.i32, 1024)
    fill(out)
    cfg = ti.lang.impl.current_cfg()
    block = range_tasks(fill, out)[0].selected_block_size
    if cfg.gpu_compute_unit_count >= 32 and cfg.gpu_subgroup_size == 32:
        assert block == 32
    elif cfg.gpu_compute_unit_count >= 16 and cfg.gpu_subgroup_size <= 64:
        assert block <= 64
    else:
        assert block in (32, 64, 128)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(1024) * 3 + 1)
    explicit = fill.with_launch_policy(ti.TaskLaunchPolicy.block(256))
    explicit(out)
    assert [t for t in explicit.report(out).tasks if t.task_type == "range_for"][0].selected_block_size == 256
    np.testing.assert_array_equal(out.to_numpy(), np.arange(1024) * 3 + 1)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_static_default_primal_and_reverse():
    x = ti.field(ti.f32, 17, needs_grad=True)
    y = ti.field(ti.f32, 17, needs_grad=True)

    @ti.kernel
    def square():
        for i in range(17):
            y[i] = x[i] * x[i]

    values = np.arange(17, dtype=np.float32)
    x.from_numpy(values)
    y.grad.fill(1)
    square()
    square.grad()
    np.testing.assert_array_equal(y.to_numpy(), values ** 2)
    np.testing.assert_array_equal(x.grad.to_numpy(), 2 * values)


@pytest.mark.parametrize("tier", ["fast", "full"])
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_static_default_composes_with_non_block_optimization(tier):
    from taichi_forge.lang._kernel_optimization import (
        _IrOptimizationOptions, _KernelOptimizationSpec,
        _bind_kernel_optimization_spec,
    )

    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, 2)):
        for I in ti.grouped(ti.ndrange(3, 5)):
            out[I] = I[0] * 5 + I[1]

    out = ti.ndarray(ti.i32, (3, 5))
    ordinary_block = range_tasks(fill, out)[0].selected_block_size
    # A tier override owns no block axis. CUDA also permits a TLS control;
    # Vulkan intentionally rejects that backend-specific option.
    thread_local = "off" if ti.lang.impl.current_cfg().arch == ti.cuda else "auto"
    selected = _bind_kernel_optimization_spec(fill, _KernelOptimizationSpec(
        ir=_IrOptimizationOptions(thread_local=thread_local, compile_tier=tier)))
    selected(out)
    assert selected.report(out).tasks[0].selected_block_size == ordinary_block
    np.testing.assert_array_equal(out.to_numpy(), np.arange(15).reshape(3, 5))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_static_default_offload_plan_owns_each_block_separately():
    from taichi_forge.lang._offload_execution_plan import (
        _OffloadExecutionPlan, _bind_offload_execution_plan,
    )

    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(17):
            out[i] = i * 3
        for i in range(17):
            out[i] += 1

    out = ti.ndarray(ti.i32, 17)
    manifests = tuple(fill.task_manifest(out))
    assert [task.task_type for task in manifests] == ["range_for", "range_for"]
    base = _OffloadExecutionPlan.from_task_manifests(manifests)
    # Both directions exercise the LLVM per-task codegen ordinal: the
    # unspecified task inherits the compiler default, never its neighbor.
    for explicit_index in (0, 1):
        plan = base.replace_task(explicit_index, workgroup_size=256)
        selected = _bind_offload_execution_plan(fill, plan)
        selected(out)
        actual = selected.task_manifest(out)
        assert actual[explicit_index].selected_block_size == 256
        assert actual[1 - explicit_index].selected_block_size == manifests[1 - explicit_index].selected_block_size
        np.testing.assert_array_equal(out.to_numpy(), np.arange(17) * 3 + 1)
