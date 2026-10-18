"""Cold admission while compact storage replay is being qualified."""

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime, AotError
from tests import test_utils
from tests.python.aot_graph_recording_probe import read, write


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_unwired_ring_rejected_without_executing_or_allocating(tmp_path):
    @ti.kernel
    def step(state: ti.types.ndarray(ti.f32, ndim=1), out: ti.types.ndarray(ti.f32, ndim=1)):
        out[0] = state[0] + 1

    module = ti.aot.Module()
    module.add_kernel(step)
    operations = [dict(kind="kernel", entry="step", arguments=["state", "out"])]
    module.add_execution_plan("ring", ti.aot.ExecutionPlan(
        operations, resources=[dict(kind="ndarray", name="state", dtype="f32", shape=[1], zero_initialize=True)],
        storage_recipe=dict(kind="ordered_temporary_ring", version=1, capacity=65, arrays=["state"])))
    module.add_execution_plan("ordinary", ti.aot.ExecutionPlan(operations))
    artifact = tmp_path / "cold.tcm"
    module.archive(str(artifact))
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"
    for factory in (lambda: Runtime(arch), ti.aot.borrow_runtime):
        with factory() as runtime:
            loaded = runtime.load_module(artifact)
            state = runtime.ndarray("f32", 1)
            output = runtime.ndarray("f32", 1)
            write(state, [4])
            write(output, [-77])
            with pytest.raises(AotError, match="Ordered storage ring replay is not available"):
                loaded.prepare_execution("ring", dict(out=output))
            assert read(output) == [-77]
            ordinary = loaded.prepare_execution("ordinary", dict(state=state, out=output))
            ordinary()
            assert read(output) == [5]
            loaded.close()
            ordinary.close()
