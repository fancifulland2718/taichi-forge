"""Automatic pack lowering preserves selected control and explicit adjoints."""

import array
import json

import pytest
import taichi_forge as ti
from taichi_forge_runtime import aot
from tests import test_utils


def read(value, code="f"):
    values = array.array(code)
    values.frombytes(value.read())
    return list(values)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_automatic_argpack_while_export_and_adjoint(tmp_path):
    packet = ti.types.argpack(limit=ti.i32, step=ti.f32, values=ti.types.ndarray(ti.f32, 1, needs_grad=True))
    source = ti.ndarray(ti.f32, 1, needs_grad=True)
    pack = packet(limit=0, step=2, values=source)

    @ti.kernel
    def reset(count: ti.types.ndarray(ti.i32, 0)):
        count[None] = 0

    @ti.kernel
    def condition(p: packet, count: ti.types.ndarray(ti.i32, 0), predicate: ti.types.ndarray(ti.i32, 0)):
        predicate[None] = int(count[None] < p.limit)

    @ti.kernel
    def advance(p: packet, count: ti.types.ndarray(ti.i32, 0)):
        for i in p.values:
            p.values[i] += p.step
        count[None] += 1

    @ti.kernel
    def energy(p: packet, loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in p.values:
            loss[0] += 2 * p.values[i] ** 2

    @ti.kernel
    def finish(count: ti.types.ndarray(ti.i32, 0), result: ti.types.ndarray(ti.i32, 1)):
        result[0] = count[None]

    def graph(budget):
        builder = ti.graph.GraphBuilder()
        p = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", packet)
        loss = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
        result = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "count", ti.i32, ndim=1)
        count = builder.private_ndarray("counter", ti.i32, ())
        predicate = builder.private_ndarray("predicate", ti.i32, ())
        prefix, cond, body = (builder.create_sequential() for _ in range(3))
        prefix.dispatch(reset, count)
        cond.dispatch(condition, p, count, predicate)
        body.dispatch(advance, p, count)
        prefix.while_loop(
            cond,
            body,
            predicate=predicate,
            counter=count,
            control_inputs=(count,),
            carried_state=(),
            max_iterations=budget,
            lowering_mode="auto" if ti.cfg.arch == ti.cpu else "native_required",
        )
        prefix.dispatch(energy, p, loss)
        prefix.dispatch(energy.grad, p, loss)
        prefix.dispatch(finish, count, result)
        builder.append(prefix)
        return builder.compile()

    ordinary, zero = graph(3), graph(0)
    module = ti.aot.Module()
    for name, limit, selected in (
        ("none", 0, ordinary),
        ("early", 1, ordinary),
        ("budget", 9, ordinary),
        ("zero", 9, zero),
    ):
        pack.limit = limit
        module.add_execution_plan(
            name,
            selected,
            source_bindings=dict(pack=pack, values=source),
            gradients=dict(values="dx", loss="dloss"),
        )
    module.add_execution_plan("public", ordinary)
    artifact = tmp_path / "control.tcm"
    module.archive(str(artifact))
    if ti.cfg.arch == ti.cuda:
        pack.limit = 1
        source.fill(3)
        source.grad.fill(0)
        loss = ti.ndarray(ti.f32, 1, needs_grad=True)
        count = ti.ndarray(ti.i32, 1)
        loss.fill(0)
        loss.grad.fill(1)
        with ti.graph.execution_domain(instance_capacity=1) as domain:
            prepared = ordinary.prepare(dict(pack=pack, loss=loss, count=count), execution_domain=domain)
            prepared.initialize()
            prepared.submit().wait()
            assert source.to_numpy().tolist() == [5]
            assert loss.to_numpy().tolist() == [50]
            assert source.grad.to_numpy().tolist() == [20]
            assert count.to_numpy().tolist() == [1]
            prepared.close()
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        arrays = {name: runtime.ndarray("f32", 1) for name in ("values", "dx", "loss", "dloss")}
        arrays["count"] = runtime.ndarray("i32", 1)
        for name, rounds in (("none", 0), ("early", 1), ("budget", 3), ("zero", 0), ("public", 3)):
            for key, value in dict(values=3, dx=0, loss=0, dloss=1).items():
                arrays[key].write(array.array("f", [value]).tobytes())
            plan = loaded.execution_plan(name)
            bindings = {key: arrays[key] for key in plan.argument_names if key != "pack"}
            if name == "public":
                bindings["pack"] = plan.argpack(
                    loaded,
                    "pack",
                    dict(limit=9, step=2, values=arrays["values"]),
                    gradients={("values",): arrays["dx"]},
                )
            if ti.cfg.arch == ti.cuda and name == "budget":
                invalid = json.loads(plan.to_json())
                loop = next(op for op in invalid["operations"] if op["kind"] == "while")
                loop["lowering"] = dict(kind="cuda_host", version=1)
                invalid["version"] = 26
                # Initialization records before replay rejects a host loop.
                # Failed group preparation must retire that partial recording
                # without executing initialization or replay.
                with runtime.execution_domain(instance_capacity=1) as failed:
                    with pytest.raises(RuntimeError, match="[Hh]ost"):
                        aot.ExecutionPlan.from_json(json.dumps(invalid)).prepare(
                            loaded, bindings, execution_domain=failed
                        )
                    assert failed.info["accepting"] and not failed.info["closed"]
                    recovered = plan.prepare(loaded, bindings, execution_domain=failed)
                    recovered.initialize()
                    recovered.close()
                assert read(arrays["values"]) == [3]
                assert read(arrays["loss"]) == read(arrays["dx"]) == [0]
            owner = plan.prepare(loaded, bindings, gradients=dict(loss=arrays["dloss"]) if name == "public" else None)
            owner.run()
            runtime.wait()
            expected = 3 + 2 * rounds
            assert read(arrays["values"]) == [expected]
            assert read(arrays["loss"]) == [2 * expected**2]
            assert read(arrays["dx"]) == [4 * expected]
            assert read(arrays["count"], "i") == [rounds]
            owner.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_automatic_argpack_branch_export_and_adjoint(tmp_path):
    packet = ti.types.argpack(gain=ti.f32, values=ti.types.ndarray(ti.f32, 1, needs_grad=True))
    source = ti.ndarray(ti.f32, 1, needs_grad=True)
    pack = packet(gain=2, values=source)

    @ti.kernel
    def condition(choice: ti.types.ndarray(ti.i32, 1), selected: ti.types.ndarray(ti.i32, 0)):
        selected[None] = choice[0]

    @ti.kernel
    def square(p: packet, loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in p.values:
            loss[0] += p.gain * p.values[i] ** 2

    @ti.kernel
    def cube(p: packet, loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in p.values:
            loss[0] += p.gain * p.values[i] ** 3

    builder = ti.graph.GraphBuilder()
    p = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", packet)
    loss = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    choice = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "choice", ti.i32, ndim=1)
    selected = builder.private_ndarray("selected", ti.i32, ())
    cond, yes, no = (builder.create_sequential() for _ in range(3))
    cond.dispatch(condition, choice, selected)
    yes.dispatch(square, p, loss)
    yes.dispatch(square.grad, p, loss)
    no.dispatch(cube, p, loss)
    no.dispatch(cube.grad, p, loss)
    builder.if_then_else(
        cond,
        yes,
        predicate=selected,
        else_region=no,
        lowering_mode="native_required" if ti.cfg.arch == ti.cuda else "auto",
    )
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan(
        "frozen",
        graph,
        source_bindings=dict(pack=pack, values=source),
        gradients=dict(values="dx", loss="dloss"),
    )
    module.add_execution_plan("public", graph)
    artifact = tmp_path / "branch.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        arrays = {name: runtime.ndarray("f32", 1) for name in ("values", "dx", "loss", "dloss")}
        arrays["choice"] = runtime.ndarray("i32", 1)
        for name in ("frozen", "public"):
            plan = loaded.execution_plan(name)
            bindings = {key: arrays[key] for key in plan.argument_names if key != "pack"}
            if name == "public":
                bindings["pack"] = plan.argpack(
                    loaded,
                    "pack",
                    dict(gain=2, values=arrays["values"]),
                    gradients={("values",): arrays["dx"]},
                )
            owner = plan.prepare(loaded, bindings, gradients=dict(loss=arrays["dloss"]) if name == "public" else None)
            for choose, energy, derivative in ((1, 18, 12), (0, 54, 54), (1, 18, 12)):
                for key, value in dict(values=3, dx=0, loss=0, dloss=1).items():
                    arrays[key].write(array.array("f", [value]).tobytes())
                arrays["choice"].write(array.array("i", [choose]).tobytes())
                owner.run()
                runtime.wait()
                assert read(arrays["loss"]) == [energy]
                assert read(arrays["dx"]) == [derivative]
            owner.close()
