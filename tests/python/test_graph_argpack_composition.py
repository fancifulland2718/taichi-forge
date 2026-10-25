"""Pack contracts survive selected control and native recording composition."""

import array
import json

import taichi_forge as ti
from taichi_forge_runtime import aot
from tests import test_utils


def read(value):
    values = array.array("f")
    values.frombytes(value.read())
    return list(values)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_argpack_selected_fork_join_keeps_branch_gradients(tmp_path):
    packet = ti.types.argpack(gain=ti.f32, values=ti.types.ndarray(ti.f32, 1, needs_grad=True))
    x, z = (ti.ndarray(ti.f32, 1, needs_grad=True) for _ in range(2))
    left, right = packet(gain=2, values=x), packet(gain=3, values=z)

    @ti.kernel
    def square(p: packet, out: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in p.values:
            out[i] = p.gain * p.values[i] ** 2

    @ti.kernel
    def join(left: ti.types.ndarray(ti.f32, 1), right: ti.types.ndarray(ti.f32, 1), total: ti.types.ndarray(ti.f32, 1)):
        total[0] = left[0] + right[0]

    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    for name in ("left", "right"):
        p = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, name, packet)
        out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name + "_out", ti.f32, ndim=1)
        builder.dispatch(square, p, out)
        builder.dispatch(square.grad, p, out)
    builder.dispatch(
        join,
        *(ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("left_out", "right_out", "total")),
    )
    # Explicit internal physical selection, as in the existing fork/join gate;
    # branch arrays below are distinct. This does not qualify automatic hazard
    # inference or arbitrary aliased pack members for parallel scheduling.
    builder._set_parallel_dispatch_groups(((0, 1), (2, 3)))
    graph = builder.compile()
    module = ti.aot.Module()
    frozen = module.add_execution_plan(
        "frozen",
        graph,
        source_bindings=dict(left=left, right=right, x=x, z=z),
        gradients=dict(x="dx", z="dz", left_out="dleft", right_out="dright"),
    )
    parallel = next(op for op in json.loads(frozen.to_json())["operations"] if op["kind"] == "cuda_parallel")
    assert [len(branch) for branch in parallel["branches"]] == [2, 2]
    module.add_execution_plan("public", graph)
    artifact = tmp_path / "parallel.tcm"
    module.archive(str(artifact))
    graph.close()
    ti.reset()
    with aot.Runtime("cuda") as runtime:
        loaded = runtime.load_module(artifact)
        arrays = {
            name: runtime.ndarray("f32", 1)
            for name in ("x", "z", "dx", "dz", "left_out", "right_out", "dleft", "dright", "total")
        }
        for name in ("frozen", "public"):
            for key, value in dict(x=3, z=5, dx=0, dz=0, left_out=0, right_out=0, dleft=1, dright=1).items():
                arrays[key].write(array.array("f", [value]).tobytes())
            plan = loaded.execution_plan(name)
            if name == "frozen":
                owner = plan.prepare(loaded, arrays)
            else:
                packs = {
                    side: plan.argpack(
                        loaded, side, dict(gain=gain, values=arrays[value]), gradients=dict(values=arrays[grad])
                    )
                    for side, gain, value, grad in (("left", 2, "x", "dx"), ("right", 3, "z", "dz"))
                }
                owner = plan.prepare(
                    loaded,
                    dict(packs, left_out=arrays["left_out"], right_out=arrays["right_out"], total=arrays["total"]),
                    gradients=dict(left_out=arrays["dleft"], right_out=arrays["dright"]),
                )
            owner.run()
            runtime.wait()
            assert read(arrays["left_out"]) == [18] and read(arrays["right_out"]) == [75]
            assert read(arrays["dx"]) == [12] and read(arrays["dz"]) == [30]
            assert read(arrays["total"]) == [93]
            owner.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_argpack_native_sequence_and_adjoint_share_storage(tmp_path):
    packet = ti.types.argpack(gain=ti.f32, values=ti.types.ndarray(ti.f32, 1, needs_grad=True))
    source = ti.ndarray(ti.f32, 2)
    transformed = ti.ndarray(ti.f32, 2, needs_grad=True)
    pack = packet(gain=2, values=transformed)
    sequence = ti.algorithms.primitive_sequence()
    sequence.transform(source, transformed, scale=3, bias=1)

    @ti.kernel
    def energy(p: packet, loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in p.values:
            loss[0] += p.gain * p.values[i] ** 2

    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    p = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", packet)
    loss = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    builder.dispatch(energy, p, loss)
    builder.dispatch(energy.grad, p, loss)
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan(
        "step",
        graph,
        source_bindings=dict(pack=pack, source=source, values=transformed),
        gradients=dict(values="dx", loss="dloss"),
    )
    backend = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    artifact = tmp_path / "recording.tcm"
    module.archive(str(artifact))
    graph.close()
    sequence.clear()
    ti.reset()
    with aot.Runtime(backend) as runtime:
        loaded = runtime.load_module(artifact)
        arrays = {
            name: runtime.ndarray("f32", size)
            for name, size in (("source", 2), ("values", 2), ("dx", 2), ("loss", 1), ("dloss", 1))
        }
        owner = loaded.prepare_execution("step", arrays)
        for data, expected in (([1, 2], [4, 7]), ([3, 4], [10, 13])):
            arrays["source"].write(array.array("f", data).tobytes())
            for key, value in (("values", [-99, -99]), ("dx", [0, 0]), ("loss", [0]), ("dloss", [1])):
                arrays[key].write(array.array("f", value).tobytes())
            owner.run()
            runtime.wait()
            assert read(arrays["values"]) == expected
            assert read(arrays["loss"]) == [sum(2 * x * x for x in expected)]
            assert read(arrays["dx"]) == [4 * x for x in expected]
        owner.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_argpack_nested_control_keeps_selected_lowering(tmp_path):
    from taichi_forge.graph._recipes import GraphFamilySelection

    packet = ti.types.argpack(
        outer=ti.i32, inner=ti.i32, step=ti.f32, values=ti.types.ndarray(ti.f32, 1, needs_grad=True)
    )
    values = ti.ndarray(ti.f32, 1, needs_grad=True)
    pack = packet(outer=1, inner=1, step=2, values=values)

    @ti.kernel
    def reset(count: ti.types.ndarray(ti.i32, 0)):
        count[None] = 0

    @ti.kernel
    def outer_condition(p: packet, count: ti.types.ndarray(ti.i32, 0), predicate: ti.types.ndarray(ti.i32, 0)):
        predicate[None] = int(count[None] < p.outer)

    @ti.kernel
    def inner_condition(p: packet, count: ti.types.ndarray(ti.i32, 0), predicate: ti.types.ndarray(ti.i32, 0)):
        predicate[None] = int(count[None] < p.inner)

    @ti.kernel
    def increment(count: ti.types.ndarray(ti.i32, 0)):
        count[None] += 1

    @ti.kernel
    def advance(p: packet, count: ti.types.ndarray(ti.i32, 0)):
        count[None] += 1
        for i in p.values:
            p.values[i] += p.step

    @ti.kernel
    def energy(p: packet, loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in p.values:
            loss[0] += 2 * p.values[i] ** 2

    builder = ti.graph.GraphBuilder()
    p = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", packet)
    loss = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    outer, inner, op, ip = (builder.private_ndarray(name, ti.i32, ()) for name in ("outer", "inner", "op", "ip"))
    oc, ob, ic, ib = (builder.create_sequential() for _ in range(4))
    builder.dispatch(reset, outer)
    oc.dispatch(outer_condition, p, outer, op)
    ob.dispatch(reset, inner)
    ic.dispatch(inner_condition, p, inner, ip)
    ib.dispatch(advance, p, inner)
    ob.while_loop(ic, ib, predicate=ip, counter=inner, max_iterations=2, masked_execution=True, name="inner")
    ob.dispatch(increment, outer)
    parent = builder.create_sequential() if ti.cfg.arch == ti.vulkan else builder
    parent.while_loop(oc, ob, predicate=op, counter=outer, max_iterations=2, masked_execution=True, name="outer")
    if ti.cfg.arch == ti.vulkan:
        builder.append(parent)
    builder.dispatch(energy, p, loss)
    builder.dispatch(energy.grad, p, loss)
    definition = builder.freeze()
    recipe = None
    if ti.cfg.arch == ti.cuda:
        catalog = definition.recipe_catalog()
        fragment = next(
            f
            for f in catalog.fragments
            if f.provider_namespace == "taichi_forge.graph.structured_control"
            and GraphFamilySelection.from_fragment(f).choice_id == "control:cuda_nested_conditional:v1"
        )
        recipe = catalog.compose((fragment.fragment_id,), stage="argpack-nested").recipe
    module = ti.aot.Module()
    with definition.materialize(recipe) as materialized:
        graph = materialized.executor
        for name, a, b in (("zero", 0, 1), ("early", 1, 1), ("budget", 9, 9)):
            pack.outer, pack.inner = a, b
            plan = module.add_execution_plan(
                name, graph, source_bindings=dict(pack=pack, values=values), gradients=dict(values="dx", loss="dloss")
            )
            if ti.cfg.arch == ti.vulkan:
                loop = next(op for op in json.loads(plan.to_json())["operations"] if op["kind"] == "while")
                assert loop["lowering"]["kind"] == "vulkan_nested"
        module.add_execution_plan("public", graph)
    backend = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    artifact = tmp_path / "nested.tcm"
    module.archive(str(artifact))
    ti.reset()
    with aot.Runtime(backend) as runtime:
        loaded = runtime.load_module(artifact)
        arrays = {name: runtime.ndarray("f32", 1) for name in ("values", "dx", "loss", "dloss")}
        for name, rounds in (("zero", 0), ("early", 1), ("budget", 4), ("public", 2)):
            for key, value in dict(values=3, dx=0, loss=0, dloss=1).items():
                arrays[key].write(array.array("f", [value]).tobytes())
            plan = loaded.execution_plan(name)
            if name == "public":
                packet = plan.argpack(
                    loaded,
                    "pack",
                    dict(outer=1, inner=9, step=2, values=arrays["values"]),
                    gradients=dict(values=arrays["dx"]),
                )
                owner = plan.prepare(
                    loaded, dict(pack=packet, loss=arrays["loss"]), gradients=dict(loss=arrays["dloss"])
                )
            else:
                owner = plan.prepare(loaded, arrays)
            owner.run()
            runtime.wait()
            expected = 3 + 2 * rounds
            assert read(arrays["values"]) == [expected]
            assert read(arrays["loss"]) == [2 * expected**2]
            assert read(arrays["dx"]) == [4 * expected]
            owner.close()
