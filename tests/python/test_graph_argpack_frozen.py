"""Frozen packs consume selected layouts and retain independent execution state."""

import array
import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime import aot
from tests import test_utils


def _fixture():
    record = ti.types.struct(weights=ti.types.vector(2, ti.f32))
    child = ti.types.argpack(offset=ti.u64, values=ti.types.ndarray(ti.f32, 1))
    pack_type = ti.types.argpack(record=record, child=child)
    values, output = ti.ndarray(ti.f32, 2), ti.ndarray(ti.f32, 2)
    values.from_numpy(np.array([3, 7], np.float32))
    pack = pack_type(record=record(weights=[2, 5]), child=child(offset=(1 << 40) + 1, values=values))

    @ti.kernel
    def stage(p: pack_type, out: ti.types.ndarray(ti.f32, 1)):
        for i in out:
            out[i] = p.record.weights[0] * p.child.values[i] + ti.cast(p.child.offset - ti.u64(1 << 40), ti.f32)

    @ti.kernel
    def finish(out: ti.types.ndarray(ti.f32, 1), p: pack_type):
        for i in out:
            out[i] += p.record.weights[1]

    symbol = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", pack_type)
    out_symbol = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stage, symbol, out_symbol)
    builder.dispatch(finish, out_symbol, symbol)
    return builder.compile(), pack, values, output, record


def _read(value):
    result = array.array("f")
    result.frombytes(value.read())
    return list(result)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_argpack_automatic_export_preserves_device_values(tmp_path):
    graph, pack, values, output, record = _fixture()
    # Mutating a nested Python matrix does not write the enclosing managed pack.
    pack.record.weights[0] = 90
    graph.run(dict(pack=pack, out=output))
    np.testing.assert_array_equal(output.to_numpy(), [12, 20])
    backend = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    backend = "cpu" if backend == "x64" else backend
    module = ti.aot.Module(caps=["spirv_has_int64"] if backend == "vulkan" else None)
    module.add_execution_plan("step", graph, source_bindings=dict(pack=pack, values=values))
    artifact = tmp_path / "pack.tcm"
    module.archive(str(artifact))
    pack.record = record(weights=[10, 20])
    pack.child.offset = (1 << 40) + 3
    graph.run(dict(pack=pack, out=output))
    np.testing.assert_array_equal(output.to_numpy(), [53, 93])
    graph.close()
    ti.reset()
    with aot.Runtime(backend) as runtime:
        loaded = runtime.load_module(artifact)
        plan = loaded.execution_plan("step")
        assert set(plan.argument_names) == {"values", "out"}
        x, first, second = (runtime.ndarray("f32", 2) for _ in range(3))
        x.write(array.array("f", [3, 7]).tobytes())
        one = plan.prepare(loaded, dict(values=x, out=first))
        two = plan.prepare(loaded, dict(values=x, out=second))
        one.run()
        one.close()
        two.run()
        runtime.wait()
        assert _read(first) == _read(second) == [12, 20]
        two.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_argpack_domain_uses_private_values_and_selected_code():
    graph, pack, values, output, record = _fixture()
    other = ti.ndarray(ti.f32, 2)
    with (
        ti.graph.execution_domain(capacity=2, instance_capacity=1) as left,
        ti.graph.execution_domain(capacity=2, instance_capacity=1) as right,
    ):
        output.fill(-1)
        pack.child.values = ti.ndarray(ti.i32, 2)
        with pytest.raises(RuntimeError, match="ndarray|dtype|type"):
            graph.prepare(dict(pack=pack, out=output), execution_domain=left)
        np.testing.assert_array_equal(output.to_numpy(), [-1, -1])
        pack.child.values = values
        first = graph.prepare(dict(pack=pack, out=output), execution_domain=left)
        first.initialize()
        pack.record = record(weights=[10, 20])
        pack.child.offset = (1 << 40) + 3
        second = graph.prepare(dict(pack=pack, out=other), execution_domain=right)
        second.initialize()
        pack.child.values = ti.ndarray(ti.f32, 2)
        pack.child.offset = 0
        graph.close()
        del graph, pack, values
        gc.collect()
        a, b = first.submit(), second.submit()
        a.wait()
        b.wait()
        np.testing.assert_array_equal(output.to_numpy(), [12, 20])
        np.testing.assert_array_equal(other.to_numpy(), [53, 93])
        first.close()
        second.submit().wait()
        np.testing.assert_array_equal(other.to_numpy(), [53, 93])
        second.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_argpack_export_explicit_adjoint_and_resource_only_pack(tmp_path):
    child = ti.types.argpack(values=ti.types.ndarray(ti.f32, 1, needs_grad=True))
    pack_type = ti.types.argpack(child=child)
    x, loss = (ti.ndarray(ti.f32, 1, needs_grad=True) for _ in range(2))
    pack = pack_type(child=child(values=x))

    @ti.kernel
    def energy(p: pack_type, result: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in p.child.values:
            result[0] += p.child.values[i] ** 2

    builder = ti.graph.GraphBuilder()
    arguments = (
        ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", pack_type),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1),
    )
    builder.dispatch(energy, *arguments)
    builder.dispatch(energy.grad, *arguments)
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("step", graph, source_bindings=dict(pack=pack, x=x), gradients=dict(x="dx", loss="dloss"))
    artifact = tmp_path / "adjoint.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        bindings = {name: runtime.ndarray("f32", 1) for name in ("x", "dx", "loss", "dloss")}
        for name, value in dict(x=3, dx=0, loss=0, dloss=1).items():
            bindings[name].write(array.array("f", [value]).tobytes())
        prepared = loaded.prepare_execution("step", bindings)
        prepared.run()
        runtime.wait()
        assert _read(bindings["loss"]) == [9]
        assert _read(bindings["dx"]) == [6]
        prepared.close()
