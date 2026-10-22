"""CGraph logical packs share kernel binding, AD and prepared owner contracts."""

import array
import json

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime import aot
from tests import test_utils


def _write(value, data):
    value.write(array.array("f", data).tobytes())


def _read(value):
    data = array.array("f")
    data.frombytes(value.read())
    return list(data)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_cgraph_argpack_direct_prepared_gradient_and_retirement(tmp_path):
    record = ti.types.struct(weights=ti.types.vector(2, ti.f32))
    child = ti.types.argpack(offset=ti.u64, values=ti.types.ndarray(ti.f32, 1, needs_grad=True))
    pack_type = ti.types.argpack(record=record, child=child)

    @ti.kernel
    def stage(p: pack_type, out: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in out:
            out[i] = p.record.weights[0] * p.child.values[i] + ti.cast(p.child.offset - ti.u64(1 << 40), ti.f32)

    @ti.kernel
    def energy(
        out: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        config: pack_type,
    ):
        for i in out:
            loss[0] += out[i] ** 2 + config.record.weights[1]

    @ti.kernel
    def checksum(p: pack_type) -> ti.u64:
        return p.child.offset

    symbol = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", pack_type)
    out_symbol = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1)
    loss_symbol = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stage, symbol, out_symbol)
    builder.dispatch(energy, out_symbol, loss_symbol, symbol)
    builder.dispatch(energy.grad, out_symbol, loss_symbol, symbol)
    builder.dispatch(stage.grad, symbol, out_symbol)
    builder.dispatch(checksum, symbol)
    graph = builder.compile()
    backend = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    module = ti.aot.Module(caps=["spirv_has_int64"] if backend == "vulkan" else None)
    module.add_graph("step", graph)
    module.add_graph("again", graph)
    directory = tmp_path / "module"
    module.save(str(directory))
    metadata = json.loads((directory / "graphs.v3.json").read_text())
    assert metadata["version"] == 3
    schema_entries = next(item["value"] for item in metadata["arguments"] if item["key"] == "step")
    schemas = {item["key"]: item["value"] for item in schema_entries}
    assert set(schemas) == {"pack", "out", "loss"}
    assert not (directory / "graphs.v2.json").exists()
    archive = tmp_path / "module.tcm"
    module.archive(str(archive))
    x = ti.ndarray(ti.f32, 2, needs_grad=True)
    output = ti.ndarray(ti.f32, 2, needs_grad=True)
    loss = ti.ndarray(ti.f32, 1, needs_grad=True)
    x.from_numpy(np.array([3, 7], np.float32))
    packet = pack_type(record=record(weights=[2, 5]), child=child(offset=(1 << 40) + 1, values=x))
    loss.grad.fill(1)
    graph.run(dict(pack=packet, out=output, loss=loss))
    np.testing.assert_array_equal(output.to_numpy(), [7, 15])
    np.testing.assert_array_equal(x.grad.to_numpy(), [28, 60])
    graph.close()
    ti.reset()
    with aot.Runtime(backend) as runtime:
        loaded = runtime.load_module(archive)
        entry = loaded.graph("step")
        x, dx, output, dout = (runtime.ndarray("f32", 2) for _ in range(4))
        loss, dloss = (runtime.ndarray("f32", 1) for _ in range(2))
        _write(x, [3, 7])
        _write(dloss, [1])
        packet = entry.argpack(
            "pack",
            dict(record=dict(weights=[2, 5]), child=dict(offset=(1 << 40) + 1, values=x)),
            gradients={("child", "values"): dx},
        )
        arguments = dict(pack=packet, out=output, loss=loss)
        gradients = dict(out=dout, loss=dloss)
        # The first dispatch is valid. Missing a later binding still prevents
        # *every* dispatch, for both ordinary and prepared calls.
        _write(output, [-11, -11])
        for action in (entry.run, entry.prepare):
            with pytest.raises(RuntimeError, match="[Gg]radient"):
                action(arguments, gradients=dict(out=dout))
            runtime.wait()
            assert _read(output) == [-11, -11]
        for method in (entry.run, lambda args, **kwargs: entry.prepare(args, **kwargs)):
            for value, data in ((output, [0, 0]), (dout, [0, 0]), (dx, [0, 0]), (loss, [0])):
                _write(value, data)
            prepared = method(arguments, gradients=gradients)
            if prepared is not None:
                prepared.run()
                prepared.close()
            runtime.wait()
            assert _read(output) == [7, 15]
            assert _read(loss) == [284]
            assert _read(dx) == [28, 60]
        other_dx, other_output, other_dout = (runtime.ndarray("f32", 2) for _ in range(3))
        other_loss = runtime.ndarray("f32", 1)
        other = loaded.graph("again").argpack(
            "pack",
            dict(record=dict(weights=[3, 6]), child=dict(offset=(1 << 40) + 2, values=x)),
            gradients={("child", "values"): other_dx},
        )
        first = entry.prepare(arguments, gradients=gradients)
        second = entry.prepare(
            dict(pack=other, out=other_output, loss=other_loss), gradients=dict(out=other_dout, loss=dloss)
        )
        first.close()
        packet.close()
        second.run()
        runtime.wait()
        assert _read(other_output) == [11, 23]
        assert _read(other_loss) == [662]
        assert _read(other_dx) == [66, 138]
        other.close()
        with pytest.raises(RuntimeError, match="closed|retired|invalid|destroyed"):
            second.run()
        second.close()
        with pytest.raises(RuntimeError, match="logical ArgPack"):
            entry.argpack("out", {})
    # The reader must reject a logical schema that omits a lifted resource.
    members = schemas["pack"]["members"][1]["members"]
    members.pop()
    (directory / "graphs.v3.json").write_text(json.dumps(metadata))
    with aot.Runtime(backend) as runtime:
        with pytest.raises(RuntimeError, match="Graph ArgPack|origins"):
            runtime.load_module(directory).graph("step")
