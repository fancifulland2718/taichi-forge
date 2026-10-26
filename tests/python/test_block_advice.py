"""Compile-time suggestions preserve explicit geometry and do not replay logs."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_explicit_block_advice_and_multi_task_contract():
    # Windows native logger handles must be captured from process startup.
    bootstrap = f"""
import importlib.abc, importlib.util, runpy, sys
sys.path[:] = {sys.path!r}
class NativePair(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'taichi_forge._lib.core.taichi_python':
            return importlib.util.spec_from_file_location(fullname, {core.__file__!r})
sys.meta_path.insert(0, NativePair())
runpy.run_path(sys.argv[1], run_name='__main__')
"""
    arch = core.arch_name(ti.lang.impl.current_cfg().arch)
    result = subprocess.run(
        [sys.executable, "-c", bootstrap, str(Path(__file__).resolve()), arch],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=60,
    )
    logs = result.stdout + result.stderr
    assert result.returncode == 0, logs
    assert "For multi-task kernels" in logs
    assert "not a measured speedup" in logs
    assert "keeping block_dim=256" in logs
    assert "global default_gpu_block_dim" in logs
    assert "[block-advice]" not in logs.split("WARM_BEGIN")[1].split("WARM_END")[0]
    assert "[block-advice]" not in logs.split("UNKNOWN_BEGIN")[1].split("UNKNOWN_END")[0]
    assert "[block-advice]" not in logs.split("MILD_BEGIN")[1].split("MILD_END")[0]


def exercise(arch):
    ti.init(arch=getattr(ti, arch), enable_fallback=False, offline_cache=False)
    out = ti.ndarray(ti.i32, 17)
    state = ti.field(ti.i32, ())
    field = ti.field(ti.i32, 17)

    @ti.kernel
    def multiple(a: ti.types.ndarray(ti.i32, 1)):
        state[None] = 2
        for i in range(17):
            a[i] = i + state[None]
        state[None] = 5
        for i in range(17):
            a[i] += state[None]

    uniform = multiple.with_launch_policy(ti.TaskLaunchPolicy.block(256, mode="require"))
    tasks = uniform.report(out).tasks
    assert [t.selected_block_size for t in tasks if t.task_type == "range_for"] == [256, 256]
    uniform(out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(17) + 7)

    @ti.kernel
    def mixed(a: ti.types.ndarray(ti.i32, 1)):
        for i in range(17):
            a[i] = i
        ti.loop_config(block_dim=64)
        for i in range(17):
            a[i] += 1

    hinted = mixed.with_launch_policy(ti.TaskLaunchPolicy.block(256))
    report = hinted.report(out)
    assert report.status == "hint_not_applied"
    assert [t.selected_block_size for t in report.tasks if t.task_type == "range_for"] == [256, 64]
    resources = [r for r in report.resources if r.task_type == "range_for"]
    assert not resources[0].rejected_candidates
    assert resources[1].rejected_candidates[0].block_dim == 256
    hinted(out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(17) + 1)
    with pytest.raises(RuntimeError, match="conflicts with.*loop_config"):
        mixed.with_launch_policy(ti.TaskLaunchPolicy.block(256, mode="require")).report(out)

    @ti.kernel
    def dense():
        ti.loop_config(block_dim=256)
        for i in field:
            field[i] = i + 3

    dense()
    assert [t.selected_block_size for t in dense.task_manifest() if t.task_type == "range_for"] == [256]
    np.testing.assert_array_equal(field.to_numpy(), np.arange(17) + 3)

    @ti.kernel
    def dense_default():
        for i in field:
            field[i] = i + 6

    dense_policy = dense_default.with_launch_policy(ti.TaskLaunchPolicy.block(256, mode="require"))
    dense_policy()
    assert [t.selected_block_size for t in dense_policy.report().tasks if t.task_type == "range_for"] == [256]
    np.testing.assert_array_equal(field.to_numpy(), np.arange(17) + 6)

    @ti.kernel
    def array(a: ti.types.ndarray(ti.i32, 1)):
        for i in a:
            a[i] = i + 4

    array_policy = array.with_launch_policy(ti.TaskLaunchPolicy.block(256, mode="require"))
    array_policy(out)
    assert [t.selected_block_size for t in array_policy.report(out).tasks if t.task_type == "range_for"] == [256]
    np.testing.assert_array_equal(out.to_numpy(), np.arange(17) + 4)
    print("WARM_BEGIN", flush=True)
    for _ in range(3):
        uniform(out)
        hinted(out)
        dense()
        dense_policy()
        array_policy(out)
    ti.sync()
    print("WARM_END", flush=True)

    @ti.kernel
    def unknown(a: ti.types.ndarray(ti.i32, 1)):
        ti.loop_config(block_dim=256)
        for i in range(a.shape[0]):
            a[i] = i

    print("UNKNOWN_BEGIN", flush=True)
    unknown(out)
    ti.sync()
    print("UNKNOWN_END", flush=True)

    @ti.kernel
    def mild(a: ti.types.ndarray(ti.i32, 1)):
        ti.loop_config(block_dim=128)
        for i in range(1024):
            a[i] = i

    larger = ti.ndarray(ti.i32, 1024)
    print("MILD_BEGIN", flush=True)
    mild(larger)
    ti.sync()
    print("MILD_END", flush=True)
    ti.reset()
    ti.init(arch=getattr(ti, arch), enable_fallback=False, offline_cache=False, default_gpu_block_dim=256)
    out = ti.ndarray(ti.i32, 17)

    @ti.kernel
    def global_setting(a: ti.types.ndarray(ti.i32, 1)):
        for i in range(17):
            a[i] = i

    global_setting(out)
    assert [t.selected_block_size for t in global_setting.task_manifest(out) if t.task_type == "range_for"] == [256]
    np.testing.assert_array_equal(out.to_numpy(), np.arange(17))


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_dense_source_block_survives_graph_and_aot(tmp_path):
    values = ti.field(ti.i32, (3, 5))

    @ti.kernel
    def dense(out: ti.types.ndarray(ti.i32, 2)):
        ti.loop_config(block_dim=256)
        for I in ti.grouped(values):
            values[I] = I[0] * 5 + I[1]
            out[I] = values[I] + 7

    out = ti.ndarray(ti.i32, (3, 5))
    builder = ti.graph.GraphBuilder()
    builder.dispatch(dense, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=2))
    graph = builder.compile()
    graph.run(dict(out=out))
    assert [t.selected_block_size for t in graph.task_manifest() if t.task_type == "range_for"] == [256]
    expected = np.arange(15).reshape(3, 5)
    np.testing.assert_array_equal(values.to_numpy(), expected)
    module = ti.aot.Module()
    module.add_field("values", values)
    module.add_execution_plan("dense", graph, snapshot_state=True)
    archive = tmp_path / "explicit-dense.tcm"
    module.archive(str(archive))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(archive) as loaded:
        target = runtime.ndarray("i32", (3, 5))
        try:
            with loaded.prepare_execution("dense", dict(out=target)) as call:
                call()
            np.testing.assert_array_equal(np.frombuffer(target.read(), np.int32).reshape(3, 5), expected + 7)
        finally:
            target.close()
    graph.close()


if __name__ == "__main__":
    exercise(sys.argv[2])
