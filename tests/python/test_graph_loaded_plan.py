"""Loaded composite plans preserve native control, private state and cold initialization."""

import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import kernel_impl
from taichi_forge_runtime.aot import ExecutionPlan
from tests import test_utils


def _restart():
    arch = ti.cfg.arch
    ti.reset()
    ti.init(arch=arch, offline_cache=False, enable_fallback=False)


def _export_loop(tmp_path):
    @ti.kernel
    def start(counter: ti.types.ndarray(ti.i32, 0), checks: ti.types.ndarray(ti.i32, 0)):
        counter[None] = 0
        checks[None] = 0

    @ti.kernel
    def condition(
        counter: ti.types.ndarray(ti.i32, 0),
        checks: ti.types.ndarray(ti.i32, 0),
        predicate: ti.types.ndarray(ti.i32, 0),
        target: ti.i32,
    ):
        checks[None] += 1
        predicate[None] = ti.cast(counter[None] < target, ti.i32)

    @ti.kernel
    def advance(counter: ti.types.ndarray(ti.i32, 0)):
        counter[None] += 1

    @ti.kernel
    def finish(
        counter: ti.types.ndarray(ti.i32, 0),
        checks: ti.types.ndarray(ti.i32, 0),
        result: ti.types.ndarray(ti.i32, 1),
    ):
        result[0] = counter[None]
        result[1] = checks[None]

    module = ti.aot.Module()
    for name, mode in (("stage", "auto" if ti.cfg.arch == ti.cpu else "native_required"), ("host", "portable")):
        builder = ti.graph.GraphBuilder()
        counter, checks, predicate = [
            builder.private_ndarray(n, ti.i32, ()) for n in ("counter", "checks", "predicate")
        ]
        target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)
        result = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.i32, ndim=1)
        cond, body = builder.create_sequential(), builder.create_sequential()
        cond.dispatch(condition, counter, checks, predicate, target)
        body.dispatch(advance, counter)
        builder.dispatch(start, counter, checks)
        builder.while_loop(
            cond,
            body,
            predicate=predicate,
            counter=counter,
            control_inputs=(counter, checks),
            max_iterations=9,
            lowering_mode=mode,
        )
        builder.dispatch(finish, counter, checks, result)
        graph = builder.compile()
        module.add_execution_plan(name, graph)
        graph.close()
    path = tmp_path / "loop.tcm"
    module.archive(str(path))
    _restart()
    return path


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_loaded_plan_control_and_device_admission(tmp_path, monkeypatch):
    path = _export_loop(tmp_path)
    with ti.aot.borrow_runtime() as runtime:
        module = runtime.load_module(path)
        result = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
        target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "limit", ti.i32)
        arguments = dict(result=result, target=target)
        mode = "device" if ti.cfg.arch == ti.cuda else "preserve"
        original = ti.graph.GraphBuilder().dispatch_loaded_plan(module, "stage", arguments, execution=mode).compile()
        graph = original.definition.compile()
        output = ti.ndarray(ti.i32, 2)
        output.fill(-1)
        calls = dict(prepare=0, launch=0, initialize=0)
        prepare, api_call = module.prepare_execution, runtime._api.call

        def counted(*args, **kwargs):
            calls["prepare"] += 1
            return prepare(*args, **kwargs)

        def native_call(function, *args):
            if function is runtime._api.ti_launch_prepared_ext:
                calls["launch"] += 1
            elif function is runtime._api.ti_initialize_prepared_ext:
                calls["initialize"] += 1
            return api_call(function, *args)

        def forbidden(*args, **kwargs):
            raise AssertionError("Loaded Plan must not compile numerical source or wait through Runtime.wait")

        monkeypatch.setattr(module, "prepare_execution", counted)
        monkeypatch.setattr(runtime._api, "call", native_call)
        with monkeypatch.context() as guard:
            guard.setattr(kernel_impl.Kernel, "ensure_compiled", forbidden)
            guard.setattr(runtime, "wait", forbidden)
            bindings = [graph.bind(dict(output=output, limit=n)) for n in (0, 1, 3, 4, 5, 9)]
            assert calls == dict(prepare=6, launch=0, initialize=0)
            for binding in bindings:
                (info,) = binding.native_execution_info()
                assert info["kind"] == "loaded_aot_plan" and not info["initialized"]
                assert info["execution_kind"] == ("cuda_graph" if ti.cfg.arch == ti.cuda else "cpu_batch")
                binding.initialize().initialize()
            assert calls["initialize"] == 6
            for binding in bindings:
                graph.submit(binding).wait()
                n = binding.snapshot()["limit"]
                np.testing.assert_array_equal(output.to_numpy(), [n, n + 1])
            graph.submit(bindings[-1]).wait()
        assert calls == dict(prepare=6, launch=7, initialize=6)
        # Actual host scheduling is preserved or explicitly rejected, never silently admitted as device-only.
        host = ti.graph.GraphBuilder().dispatch_loaded_plan(module, "host", arguments, execution="device").compile()
        registered = set(runtime._resources)
        for _ in range(2):
            with pytest.raises(ti.TaichiRuntimeError, match="device execution rejected actual"):
                host.bind(dict(output=output, limit=1))
        gc.collect()
        assert runtime._resources == registered
        np.testing.assert_array_equal(output.to_numpy(), [9, 10])
        host.close()
        ticket = graph.submit(bindings[1])
        module.close()
        ticket.wait()
        with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="closed"):
            graph.submit(bindings[0])
        graph.close()
        original.close()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_loaded_plan_namespaces_packs_and_initialization(tmp_path):
    config = ti.types.argpack(delta=ti.i32)
    vector = ti.types.vector(2, ti.i32)

    @ti.kernel
    def stage(p: config, history: ti.types.ndarray(ti.i32, 1), output: ti.types.ndarray(ti.i32, 1), offsets: vector):
        history[0] += p.delta + offsets[0] - offsets[1]
        output[0] = history[0]

    artifact = ti.aot.Module()
    artifact.add_kernel(stage)
    artifact.add_execution_plan(
        "stage",
        ExecutionPlan(
            [dict(kind="kernel", entry="stage", arguments=["pack", "history", "output", "offsets"])],
            resources=[dict(kind="ndarray", name="history", dtype="i32", shape=[1])],
            initialization=[dict(kind="fill32", destination="history", value=7)],
        ),
    )
    path = tmp_path / "state.tcm"
    artifact.archive(str(path))
    _restart()
    with ti.aot.borrow_runtime() as runtime:
        first, second = (runtime.load_module(path) for _ in range(2))
        builder = ti.graph.GraphBuilder()
        sequence = builder.create_sequential()
        pack1 = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "first", config)
        pack2 = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "second", config)
        a = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "a", ti.i32, ndim=1)
        b = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "b", ti.i32, ndim=1)
        offsets = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "offsets", vector)
        wrong = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "offsets", ti.types.vector(2, ti.f32))
        with pytest.raises(ValueError, match="value symbol"):
            builder.dispatch_loaded_plan(first, "stage", dict(pack=pack1, output=a, offsets=wrong))
        sequence.dispatch_loaded_plan(first, "stage", dict(pack=pack1, output=a, offsets=offsets))
        sequence.dispatch_loaded_plan(second, "stage", dict(pack=pack2, output=b, offsets=offsets))
        builder.append(sequence)
        original = builder.compile()
        graph = original.definition.compile()
        a_value, b_value = (ti.ndarray(ti.i32, 1) for _ in range(2))
        a_value.fill(-1)
        b_value.fill(-1)
        packet = config(delta=2)
        binding = graph.bind(dict(first=packet, second=config(delta=10), a=a_value, b=b_value, offsets=vector([3, 3])))
        np.testing.assert_array_equal(a_value.to_numpy(), [-1])
        binding.initialize()
        np.testing.assert_array_equal(a_value.to_numpy(), [-1])  # Only the declared private initializer ran.
        assert len(binding.native_execution_info()) == 2
        packet.delta = 100
        graph.submit(binding)
        graph.submit(binding).wait()
        np.testing.assert_array_equal(a_value.to_numpy(), [11])
        np.testing.assert_array_equal(b_value.to_numpy(), [27])
        independent = graph.bind(
            dict(first=config(delta=3), second=config(delta=4), a=a_value, b=b_value, offsets=vector([2, 2]))
        )
        graph.submit(independent).wait()  # Lazy initialization also preserves ordinary Runtime behavior.
        np.testing.assert_array_equal(a_value.to_numpy(), [10])
        np.testing.assert_array_equal(b_value.to_numpy(), [11])
        graph.submit(binding).wait()
        np.testing.assert_array_equal(a_value.to_numpy(), [13])
        np.testing.assert_array_equal(b_value.to_numpy(), [37])

        # A portable outer loop also reuses its nested packet, including private history.
        @ti.kernel
        def condition(value: ti.types.ndarray(ti.i32, 1), predicate: ti.types.ndarray(ti.i32, 0)):
            predicate[None] = ti.cast(value[0] < 16, ti.i32)

        outer = ti.graph.GraphBuilder()
        predicate = outer.private_ndarray("predicate", ti.i32, ())
        check, body = outer.create_sequential(), outer.create_sequential()
        check.dispatch(condition, a, predicate)
        body.dispatch_loaded_plan(first, "stage", dict(pack=pack1, output=a, offsets=offsets))
        outer.while_loop(check, body, predicate=predicate, max_iterations=5, lowering_mode="portable")
        controlled = outer.compile()
        loop_binding = controlled.bind(dict(first=config(delta=3), a=a_value, offsets=vector([2, 2])))
        loop_binding.initialize()
        assert len(loop_binding.native_execution_info()) == 1
        a_value.fill(0)
        controlled.run(loop_binding)
        np.testing.assert_array_equal(a_value.to_numpy(), [16])
        controlled.close()
        first.close()
        with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="closed"):
            graph.submit(binding)
        graph.close()
        original.close()
