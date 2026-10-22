"""Cold native code loans keep selected JIT code and live Field storage."""

import gc
import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import ExecutionPlan
from taichi_forge_runtime.aot._runtime import Module
from tests import test_utils


def _graph(kernel, *args):
    builder = ti.graph.GraphBuilder()
    builder.dispatch(kernel, *args)
    graph = builder.compile()
    assert len(graph._spec.nodes) == 1
    native = graph._spec.nodes[0].compiled_graph
    return graph, native


def _loan(runtime, native):
    handle = runtime._borrower.import_jit_kernels([("stage", native, 0, False)])
    return Module(runtime, handle)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_jit_code_loan_shares_dense_state_and_retires_before_tree():
    state = ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 8).place(state)
    tree = fields.finalize()
    state.fill(5)

    @ti.kernel
    def stage(out: ti.types.ndarray(ti.i32, 1), delta: ti.i32):
        for i in range(8):
            state[i] += delta
            out[i] = state[i]

    graph, native = _graph(
        stage,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "delta", ti.i32),
    )
    out = ti.ndarray(ti.i32, shape=8)
    with ti.aot.borrow_runtime() as runtime:
        module = _loan(runtime, native)
        output = runtime.import_ndarray(out)
        left = runtime.execution_domain(instance_capacity=4)
        right = runtime.execution_domain(instance_capacity=4)
        plan = ExecutionPlan(
            [dict(kind="kernel", entry="stage", arguments=["out", "delta"])]
        )
        first = module.prepare_execution(
            plan, dict(out=output, delta=3), execution_domain=left
        )
        second = module.prepare_execution(
            plan, dict(out=output, delta=7), execution_domain=right
        )
        first.initialize()
        second.initialize()
        ready = first.submit()
        done = second.submit(wait_for=(ready,))
        done.wait()
        np.testing.assert_array_equal(state.to_numpy(), np.full(8, 15))
        np.testing.assert_array_equal(out.to_numpy(), np.full(8, 15))
        # The ordinary JIT launcher sees the same root and stays usable.
        stage(out, 2)
        ti.sync()
        del graph
        gc.collect()
        first.submit().wait()
        np.testing.assert_array_equal(out.to_numpy(), np.full(8, 20))
        old_kernel = module.kernel("stage")
        tree_id, generation = tree.id, tree.generation
        tree.destroy()
        assert left.info["closed"] and right.info["closed"]
        with pytest.raises(RuntimeError, match="retired|live|closed"):
            first.submit()
        with pytest.raises(RuntimeError, match="retired"):
            module.kernel("stage")
        with pytest.raises(RuntimeError, match="live|retired"):
            old_kernel.bind(output, 1).run()
        replacement = ti.field(ti.i32)
        fields = ti.FieldsBuilder()
        fields.dense(ti.i, 8).place(replacement)
        replacement_tree = fields.finalize()
        assert (
            replacement_tree.id == tree_id and replacement_tree.generation != generation
        )
        with pytest.raises((ValueError, RuntimeError), match="artifact|retired|stale"):
            _loan(runtime, native)
        replacement_tree.destroy()
        first.close()
        second.close()
        module.close()
        left.close()
        right.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_jit_code_loan_rejects_old_program_without_frontend_dereference():
    arch = ti.lang.impl.current_cfg().arch
    @ti.kernel
    def stage(out: ti.types.ndarray(ti.i32, 1)):
        out[0] = 73

    graph, native = _graph(
        stage, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    )
    old = ti.aot.borrow_runtime()
    module = _loan(old, native)
    ti.reset()
    ti.init(arch=arch, offline_cache=False)
    with ti.aot.borrow_runtime() as current:
        with pytest.raises((ValueError, RuntimeError), match="Program"):
            _loan(current, native)
    module.close()
    old.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_jit_code_loan_rejects_shared_state_and_invalid_descriptors():
    @ti.kernel
    def random_stage(out: ti.types.ndarray(ti.i32, 1)):
        out[0] = ti.random(ti.i32)

    graph, native = _graph(
        random_stage, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    )
    with ti.aot.borrow_runtime() as runtime:
        domain = runtime.execution_domain()
        with runtime._lock:
            with pytest.raises(RuntimeError, match="submission scope"):
                _loan(runtime, native)
        with pytest.raises((ValueError, RuntimeError), match="random"):
            _loan(runtime, native)
        with pytest.raises((ValueError, RuntimeError), match="Invalid"):
            runtime._borrower.import_jit_kernels([("stage", native, 100, False)])
        assert domain.latest_completion().info["sequence"] == 0
        assert domain.info["accepting"]
        domain.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_jit_code_loan_preserves_frozen_task_launch_plan():
    from taichi_forge._lib import core
    from taichi_forge.lang._offload_execution_plan import (
        _OffloadExecutionPlan,
        _bind_offload_execution_plan,
    )

    count = 1 << 16
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(count):
            out[i] = i * 5 + 11

    baseline = _OffloadExecutionPlan.from_task_manifests(fill.task_manifest(values))
    indices = tuple(
        task.task_index for task in baseline.tasks if task.task_kind == "range_for"
    )
    assert len(indices) == 1
    graphs, natives = [], []
    for target in (2, 4):
        selected = baseline.replace_task(
            indices[0], range_work_per_thread_target=target
        )
        graph, native = _graph(
            _bind_offload_execution_plan(fill, selected),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1),
        )
        graphs.append(graph)
        natives.append(native)
    with ti.aot.borrow_runtime() as runtime:
        module = Module(
            runtime,
            runtime._borrower.import_jit_kernels(
                [
                    (f"fill{index}", native, 0, False)
                    for index, native in enumerate(natives)
                ]
            ),
        )
        output = runtime.import_ndarray(values)
        domain = runtime.execution_domain(instance_capacity=4)
        calls = []
        for index, target in enumerate((2, 4)):
            plan = ExecutionPlan(
                [dict(kind="kernel", entry=f"fill{index}", arguments=["out"])]
            )
            call = module.prepare_execution(
                plan, dict(out=output), execution_domain=domain
            )
            call.initialize()
            assert (
                core.query_int64("cuda_range_coarsening_last_work_per_thread_target")
                == target
            )
            calls.append(call)
        resolved = core.query_int64("cuda_range_coarsening_resolution_calls")
        for call in calls:
            call.submit().wait()
            np.testing.assert_array_equal(
                values.to_numpy(), np.arange(count, dtype=np.int32) * 5 + 11
            )
        assert core.query_int64("cuda_range_coarsening_resolution_calls") == resolved
        for call in calls:
            call.close()
        domain.close()
        module.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_code_loan_reuses_pipelines_and_revokes_field_generation():
    state = ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 8).place(state)
    tree = fields.finalize()
    state.fill(5)
    out = ti.ndarray(ti.i32, 8)

    @ti.kernel
    def advance(result: ti.types.ndarray(ti.i32, 1), delta: ti.i32):
        for i in range(8):
            state[i] += delta
            result[i] = state[i]

    graph, native = _graph(
        advance,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "delta", ti.i32),
    )
    graph.run(dict(out=out, delta=0))
    ti.sync()
    program = ti.lang.impl.get_runtime().prog
    registered = program._debug_kernel_lifecycle_stats()["registered_executables"]
    with ti.aot.borrow_runtime() as runtime:
        modules = [_loan(runtime, native) for _ in range(2)]
        output = runtime.import_ndarray(out)
        plan = ExecutionPlan([dict(kind="kernel", entry="stage", arguments=["out", "delta"])])
        calls = [module.prepare_execution(plan, dict(out=output, delta=delta))
                 for module, delta in zip(modules, (3, 7))]
        assert program._debug_kernel_lifecycle_stats()["registered_executables"] == registered
        first, second = calls
        first.submit().wait()
        np.testing.assert_array_equal(out.to_numpy(), np.full(8, 8))
        first.close()
        modules[0].close()
        del graph
        gc.collect()
        second.submit().wait()
        np.testing.assert_array_equal(out.to_numpy(), np.full(8, 15))
        advance(out, 2)
        ti.sync()
        second.submit().wait()
        np.testing.assert_array_equal(out.to_numpy(), np.full(8, 24))
        old_kernel = modules[1].kernel("stage")
        tree.destroy()
        with pytest.raises(RuntimeError, match="retired|live|closed"):
            second.submit()
        with pytest.raises(RuntimeError, match="retired|live"):
            old_kernel.run(output, 1)
        with pytest.raises((ValueError, RuntimeError), match="artifact|retired|stale"):
            _loan(runtime, native)
        second.close()
        modules[1].close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_code_loan_validates_complete_closure_before_registration():
    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, 1)):
        out[0] = 73

    graph, native = _graph(fill, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1))
    program = ti.lang.impl.get_runtime().prog
    with ti.aot.borrow_runtime() as runtime:
        registered = program._debug_kernel_lifecycle_stats()["registered_executables"]
        for invalid, message in (
            (("invalid", native, 100, False), "Invalid"),
            (("stage", native, 0, False), "Duplicate"),
            (("gated", native, 0, True), "gated"),
        ):
            with pytest.raises((ValueError, RuntimeError), match=message):
                runtime._borrower.import_jit_kernels([("stage", native, 0, False), invalid])
            assert program._debug_kernel_lifecycle_stats()["registered_executables"] == registered
        module = _loan(runtime, native)
        output = runtime.ndarray("i32", 1)
        module.kernel("stage").run(output)
        runtime.wait()
        assert int.from_bytes(output.read(), "little", signed=True) == 73
        module.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_jit_code_loan_preserves_packed_abi_and_explicit_gradients():
    child = ti.types.argpack(values=ti.types.ndarray(ti.f32, 1, needs_grad=True))
    parameters = ti.types.argpack(offset=ti.u64, gain=ti.f32, child=child)

    @ti.kernel
    def energy(p: parameters, loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in p.child.values:
            x = p.child.values[i] + ti.cast(p.offset - ti.u64(1 << 40), ti.f32)
            loss[0] += p.gain * x * x

    arguments = (
        ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "parameters", parameters),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1),
    )
    forward, forward_native = _graph(energy, *arguments)
    reverse, reverse_native = _graph(energy.grad, *arguments)
    with ti.aot.borrow_runtime() as runtime:
        module = Module(runtime, runtime._borrower.import_jit_kernels([
            ("forward", forward_native, 0, False),
            ("reverse", reverse_native, 0, False),
        ]))
        arrays = {name: runtime.ndarray("f32", 1) for name in ("x", "dx", "loss", "dloss")}
        for name, value in dict(x=3, dx=0, loss=0, dloss=1).items():
            arrays[name].write(np.array([value], np.float32).tobytes())
        pack = module.kernel("forward").argpack(
            0, dict(offset=(1 << 40) + 1, gain=2, child=dict(values=arrays["x"])),
            gradients={("child", "values"): arrays["dx"]},
        )
        plan = ExecutionPlan([
            dict(kind="kernel", entry="forward", arguments=["pack", "loss"]),
            dict(kind="kernel", entry="reverse", arguments=["pack", "loss"]),
        ], gradients=dict(loss="dloss"))
        prepared = module.prepare_execution(plan, dict(pack=pack, loss=arrays["loss"], dloss=arrays["dloss"]))
        del forward, reverse
        gc.collect()
        prepared.submit().wait()
        np.testing.assert_allclose(np.frombuffer(arrays["loss"].read(), np.float32), [32])
        np.testing.assert_allclose(np.frombuffer(arrays["dx"].read(), np.float32), [16])
        prepared.close()
        module.close()
