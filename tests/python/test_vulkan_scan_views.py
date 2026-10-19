"""Scan descriptors respect physical alignment at input and every scratch level."""

import os

import numpy as np
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_scan_offset_views_and_multilevel_workspace(tmp_path):
    configs = [(ti.u32, np.uint32, 31), (ti.u64, np.uint64, 277), (ti.i32, np.int32, 65539)]
    module = ti.aot.Module()
    for dtype, scalar, size in configs:
        source = (np.arange(size) % 5).astype(scalar)
        values = ti.ndarray(dtype, size)
        values.from_numpy(source)
        ti.algorithms.PrefixSumExecutor(size).run(values)
        np.testing.assert_array_equal(values.to_numpy(), np.cumsum(source, dtype=scalar))
        plan = ti.aot.ExecutionPlan(
            [dict(kind="native", provider="builtin", operation="scan", version=1,
                  parameters=dict(size=size, dtype=str(dtype)), bindings=dict(values="values"))],
            resources=[dict(kind="ndarray_view", name="values", array="parent", offset=1, shape=[size])],
        )
        module.add_execution_plan(str(dtype), plan)
    artifact = tmp_path / "scan-views.tcm"
    module.archive(str(artifact))
    for borrowed in (False, True):
        runtime = ti.aot.borrow_runtime() if borrowed else Runtime(
            "vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))
        )
        with runtime, runtime.load_module(artifact) as loaded:
            for dtype, scalar, size in configs:
                parent = runtime.ndarray(str(dtype), size + 3)
                with loaded.prepare_execution(str(dtype), dict(parent=parent)) as call:
                    for factor in (1, 0, 2):
                        initial = np.full(size + 3, 79, scalar)
                        initial[1:size + 1] = (np.arange(size) % 5) * factor
                        parent.write(initial)
                        expected = initial.copy()
                        expected[1:size + 1] = np.cumsum(initial[1:size + 1], dtype=scalar)
                        call()
                        np.testing.assert_array_equal(np.frombuffer(parent.read(), scalar), expected)
                parent.close()
