import ast
import gc
import inspect
from pathlib import Path
import textwrap

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.kernel_impl import Kernel, _TaskLaunchBinding
from tests import test_utils


def _qualification_counters(program):
    stats = program._debug_gpu_artifact_qualification_stats()
    return {
        name: stats[name]
        for name in (
            "qualification_calls",
            "registration_materializations",
            "function_attribute_queries",
            "occupancy_queries",
        )
    }


def test_gpu_semantics_are_absent_from_python_launch_hot_paths():
    for callable_ in (Kernel.__call__, _TaskLaunchBinding.__call__):
        tree = ast.parse(textwrap.dedent(inspect.getsource(callable_)))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert not any("gpu_semantics" in name for name in names | attributes)

    root = Path(__file__).resolve().parents[2]
    program_source = (root / "taichi" / "program" / "program.cpp").read_text(encoding="utf-8")
    begin = program_source.index("void Program::RegisteredKernelExecutionPlan::launch(")
    end = program_source.index("Program::register_kernel_execution_plan(", begin)
    hot_path = program_source[begin:end]
    assert "gpu_semantics" not in hot_path
    assert "qualification" not in hot_path


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_ordinary_direct_and_graph_paths_do_not_touch_semantics_providers():
    count = 256
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def increment(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] += 1

    graph_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment, graph_arg)
    graph = builder.compile()
    program = impl.get_runtime().prog

    increment(values)
    graph.run({"out": values})
    ti.sync()
    before = _qualification_counters(program)
    for _ in range(100):
        increment(values)
    for _ in range(100):
        graph.run({"out": values})
    ti.sync()
    after = _qualification_counters(program)

    assert after == before
    assert values.to_numpy().sum() == count * 202


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_resident_and_qualified_query_loops_have_bounded_native_lifecycle():
    values = ti.ndarray(ti.i32, shape=4097)

    @ti.kernel
    def transform(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(4097):
            out[i] = (i * 17 + 3) % 101

    program = impl.get_runtime().prog
    resident = transform._primal._gpu_semantics_snapshot(values)
    gc.collect()
    runtime_before = program._runtime_statistics_snapshot()
    registrations_before = program._debug_kernel_registration_count()
    qualification_before = _qualification_counters(program)
    for _ in range(1000):
        assert transform._primal._gpu_semantics_snapshot(values) == resident
    gc.collect()
    runtime_after_resident = program._runtime_statistics_snapshot()
    assert program._debug_kernel_registration_count() == registrations_before
    assert _qualification_counters(program) == qualification_before
    assert runtime_after_resident["submission"] == runtime_before["submission"]
    assert runtime_after_resident["transfer"] == runtime_before["transfer"]
    assert runtime_after_resident["memory"] == runtime_before["memory"]

    first = transform._primal._gpu_semantics_qualification(values)
    registrations_qualified = program._debug_kernel_registration_count()
    runtime_before_repeat = program._runtime_statistics_snapshot()
    for _ in range(100):
        current = transform._primal._gpu_semantics_qualification(values)
        assert current.semantics == first.semantics
        assert current.registration_materialization_count == 0
    gc.collect()
    runtime_after_repeat = program._runtime_statistics_snapshot()
    assert program._debug_kernel_registration_count() == registrations_qualified
    assert runtime_after_repeat["submission"] == runtime_before_repeat["submission"]
    assert runtime_after_repeat["transfer"] == runtime_before_repeat["transfer"]
    assert runtime_after_repeat["memory"] == runtime_before_repeat["memory"]
    assert values.to_numpy().sum() == 0


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_qualified_artifacts_are_released_by_runtime_reset():
    @ti.kernel
    def no_op():
        pass

    arch = impl.current_cfg().arch
    program = impl.get_runtime().prog
    no_op._primal._gpu_semantics_qualification()
    if ti_core.arch_name(arch) == "cuda":
        assert program._debug_kernel_registration_count() > 0
        loaded_bytes = int(ti_core.query_int64("cuda_artifact_cubin_current_bytes"))
    else:
        assert program._debug_kernel_registration_count() == 0
        loaded_bytes = 0

    ti.reset()
    if ti_core.arch_name(arch) == "cuda":
        assert int(ti_core.query_int64("cuda_artifact_cubin_current_bytes")) == 0
    ti.init(arch=arch, enable_fallback=False, offline_cache=False)
    replacement = impl.get_runtime().prog
    assert replacement._debug_kernel_registration_count() == 0
    if ti_core.arch_name(arch) == "cuda":
        assert loaded_bytes >= 0
