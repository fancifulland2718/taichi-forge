import numpy as np

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._kernel_optimization import (
    _BackendCodegenOptions,
    _IrOptimizationOptions,
    _KernelOptimizationSpec,
    _LaunchOptions,
    _bind_kernel_optimization_spec,
)
from tests import test_utils


def _spec(thread_local):
    return _KernelOptimizationSpec(
        ir=_IrOptimizationOptions(thread_local=thread_local),
        backend=_BackendCodegenOptions(workgroup_size=256),
        launch=_LaunchOptions(block_mode="require"),
    )


def _range_task(report):
    tasks = tuple(task for task in report.tasks if task.task_type == "range_for")
    assert len(tasks) == 1
    return tasks[0]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_thread_local_variants_are_physical_and_observable():
    count = 1 << 16
    values = ti.ndarray(ti.i32, shape=count)
    result = ti.field(ti.i32, shape=())

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i % 17

    @ti.kernel
    def reduce(inp: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            result[None] += inp[i]

    fill(values)
    expected = int(np.sum(np.arange(count, dtype=np.int64) % 17))
    automatic = _bind_kernel_optimization_spec(reduce, _spec("auto"))
    disabled = _bind_kernel_optimization_spec(reduce, _spec("off"))

    automatic_report = automatic.report(values)
    disabled_report = disabled.report(values)
    automatic_task = _range_task(automatic_report)
    disabled_task = _range_task(disabled_report)
    assert automatic_task.thread_local_bytes > 0
    assert disabled_task.thread_local_bytes == 0
    assert automatic_task.logical_task_id == disabled_task.logical_task_id
    assert automatic_task.task_id != disabled_task.task_id
    assert automatic_task.optimization_spec_id != disabled_task.optimization_spec_id
    compilation_keys = {
        automatic._optimization_spec.compilation_specialization_key,
        disabled._optimization_spec.compilation_specialization_key,
    }
    compiled_variants = {
        key[3]: kernel_cpp
        for key, kernel_cpp in automatic._kernel.compiled_kernels.items()
        if len(key) == 5 and key[3] in compilation_keys
    }
    assert set(compiled_variants) == compilation_keys
    assert len({id(kernel_cpp) for kernel_cpp in compiled_variants.values()}) == 2

    result[None] = 0
    automatic(values)
    ti.sync()
    assert result[None] == expected
    result[None] = 0
    disabled(values)
    ti.sync()
    assert result[None] == expected

    program = impl.get_runtime().prog
    runtime_memory = program._runtime_statistics_snapshot()["memory"]
    host_memory = dict(ti_core.get_host_memory_pool_stats())
    device_memory = dict(ti_core.get_device_memory_pool_stats())
    for _ in range(32):
        result[None] = 0
        automatic(values)
        result[None] = 0
        disabled(values)
    ti.sync()
    assert program._runtime_statistics_snapshot()["memory"] == runtime_memory
    assert dict(ti_core.get_host_memory_pool_stats()) == host_memory
    assert dict(ti_core.get_device_memory_pool_stats()) == device_memory
