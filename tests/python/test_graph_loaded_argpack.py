"""Loaded ArgPack calls use exported value ABI and borrowed storage leases."""

import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import kernel_impl
from taichi_forge_runtime.aot import AotError
from tests import test_utils


def _export(tmp_path):
    record = ti.types.struct(weights=ti.types.vector(2, ti.i32), bias=ti.i32)
    child = ti.types.argpack(offset=ti.u64, values=ti.types.ndarray(ti.i32, 1))
    config = ti.types.argpack(record=record, child=child)

    @ti.kernel
    def advance(p: config, out: ti.types.ndarray(ti.i32, 1)):
        for i in out:
            out[i] = (
                p.child.values[i] * p.record.weights[0]
                + p.record.bias
                + ti.cast(p.child.offset - ti.u64(1 << 40), ti.i32)
            )

    artifact = ti.aot.Module()
    artifact.add_kernel(advance)
    path = tmp_path / "pack.tcm"
    artifact.archive(str(path))
    arch = ti.cfg.arch
    ti.reset()
    ti.init(arch=arch, offline_cache=False, enable_fallback=False)
    return path, record, child, config


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_loaded_argpack_snapshot_aliases_and_retirement(tmp_path, monkeypatch):
    path, record, child, config = _export(tmp_path)
    with ti.aot.borrow_runtime() as runtime:
        module = runtime.load_module(path)
        entry = module.kernel("advance")
        pack = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", config)
        out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
        original = ti.graph.GraphBuilder().dispatch_loaded(entry, pack, out).compile()
        graph = original.definition.compile()
        values = ti.ndarray(ti.i32, 19)
        values.fill(0)
        packet = config(record=record(weights=[2, 3], bias=1), child=child(offset=(1 << 40) + 3, values=values))
        count, imported = [], []
        prepare, import_array = entry.prepare, runtime.import_ndarray
        api_call = runtime._api.call
        observed = dict(pack=0, launch=0)

        def native_call(function, *args):
            if function is runtime._api.ti_create_kernel_argpack_ext:
                observed["pack"] += 1
            elif function is runtime._api.ti_launch_prepared_ext:
                observed["launch"] += 1
            return api_call(function, *args)

        def counted(*args):
            count.append(1)
            return prepare(*args)

        def import_count(value):
            imported.append(value)
            return import_array(value)

        def forbidden(*args, **kwargs):
            raise AssertionError("Loaded entry must not compile numerical source")

        monkeypatch.setattr(entry, "prepare", counted)
        monkeypatch.setattr(runtime, "import_ndarray", import_count)
        monkeypatch.setattr(runtime._api, "call", native_call)
        with monkeypatch.context() as guard:
            guard.setattr(kernel_impl.Kernel, "ensure_compiled", forbidden)
            binding = graph.bind(dict(pack=packet, out=values))
            assert imported == [values]  # Aliases across pack and root bind once.
            graph.submit(binding)
            graph.submit(binding).wait()
            packet.child.offset = (1 << 40) + 10
            graph.submit(binding).wait()  # Existing packet retains its snapshot.
        np.testing.assert_array_equal(values.to_numpy(), np.full(19, 28))
        assert len(count) == 1
        assert observed == dict(pack=1, launch=3)
        rebound = graph.bind(dict(pack=packet, out=values))
        graph.submit(rebound).wait()
        np.testing.assert_array_equal(values.to_numpy(), np.full(19, 67))
        assert len(count) == 2
        assert observed == dict(pack=2, launch=4)
        # Transient pack/import handles belong to the prepared binding only.
        gc.collect()
        registered = len(runtime._resources)
        for _ in range(3):
            ticket = graph.submit(dict(pack=packet, out=values))
            ticket.wait()
            del ticket
        gc.collect()
        assert len(runtime._resources) == registered
        ticket = graph.submit(binding)
        module.close()
        ticket.wait()
        with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="closed"):
            graph.submit(rebound)
        graph.close()
        original.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_loaded_argpack_rejects_schema_and_releases_failed_bindings(tmp_path):
    path, record, child, config = _export(tmp_path)
    with ti.aot.borrow_runtime() as runtime:
        entry = runtime.load_module(path).kernel("advance")
        out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
        wrong = ti.types.argpack(
            record=record, child=ti.types.argpack(offset=ti.i32, values=ti.types.ndarray(ti.i32, 1))
        )
        with pytest.raises(ValueError, match="value type"):
            ti.graph.GraphBuilder().dispatch_loaded(entry, ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", wrong), out)
        missing = ti.types.argpack(record=record)
        with pytest.raises(ValueError, match="members"):
            ti.graph.GraphBuilder().dispatch_loaded(entry, ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", missing), out)
        # A matching pure-value layout cannot disguise a false resource type.
        fake_child = ti.types.argpack(offset=ti.u64, values=ti.types.ndarray(ti.f32, 1))
        fake = ti.types.argpack(record=record, child=fake_child)
        graph = (
            ti.graph.GraphBuilder()
            .dispatch_loaded(entry, ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", fake), out)
            .compile()
        )
        values, output = ti.ndarray(ti.f32, 7), ti.ndarray(ti.i32, 7)
        packet = fake(record=record(weights=[2, 3], bias=1), child=fake_child(offset=1 << 40, values=values))
        registered = set(runtime._resources)
        for _ in range(2):
            with pytest.raises((ValueError, AotError), match="(?i)type|dtype|argument"):
                graph.bind(dict(pack=packet, out=output))
        gc.collect()
        assert runtime._resources == registered
        graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_loaded_sequence_uses_graph_owned_workspace(tmp_path):
    path, record, child, config = _export(tmp_path)

    @ti.kernel
    def publish(work: ti.types.ndarray(ti.i32, 1), out: ti.types.ndarray(ti.i32, 1)):
        for i in out:
            out[i] = work[i] + 1

    @ti.kernel
    def condition(predicate: ti.types.ndarray(ti.i32, 0)):
        predicate[None] = 1

    with ti.aot.borrow_runtime() as runtime:
        module = runtime.load_module(path)
        entry = module.kernel("advance")
        values, output = ti.ndarray(ti.i32, 19), ti.ndarray(ti.i32, 19)
        values.fill(5)
        packet = config(record=record(weights=[2, 3], bias=1), child=child(offset=(1 << 40) + 3, values=values))
        pack = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", config)
        out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
        for sequential in (False, True):
            builder = ti.graph.GraphBuilder()
            scratch = builder.private_ndarray("scratch", ti.i32, (19,))
            target = builder.create_sequential() if sequential else builder
            target.dispatch_loaded(entry, pack, scratch)
            target.dispatch(publish, scratch, out)
            if sequential:
                builder.append(target)
            original = builder.compile()
            graph = original.definition.compile()
            binding = graph.bind(dict(pack=packet, out=output))
            graph.submit(binding).wait()
            np.testing.assert_array_equal(output.to_numpy(), np.full(19, 15))
            if sequential and ti.cfg.arch == ti.cuda:
                predicate = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0)
                check = builder.create_sequential()
                check.dispatch(condition, predicate)
                with pytest.raises(ti.TaichiRuntimeError, match="host native actions"):
                    ti.graph.GraphBuilder().while_loop(
                        check, target, predicate=predicate, max_iterations=3, lowering_mode="native_required"
                    )
            graph.close()
            original.close()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_loaded_pack_and_root_gradients_use_exported_abi(tmp_path):
    config = ti.types.argpack(scale=ti.f32, values=ti.types.ndarray(ti.f32, 1, needs_grad=True))

    @ti.kernel
    def stage(p: config, out: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in out:
            out[i] = p.scale * p.values[i]

    module = ti.aot.Module()
    module.add_kernel(stage)
    module.add_kernel(stage.grad, name="backward")
    path = tmp_path / "grad.tcm"
    module.archive(str(path))
    arch = ti.cfg.arch
    ti.reset()
    ti.init(arch=arch, offline_cache=False, enable_fallback=False)
    values, out = (ti.ndarray(ti.f32, 3, needs_grad=True) for _ in range(2))
    values.from_numpy(np.array([1, 2, 3], np.float32))
    values.grad.fill(0)
    out.grad.fill(1)
    packet = config(scale=2, values=values)
    pack_symbol = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "p", config)
    out_symbol = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1)
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(path)
        builder = ti.graph.GraphBuilder()
        builder.dispatch_loaded(loaded.kernel("stage"), pack_symbol, out_symbol)
        builder.dispatch_loaded(loaded.kernel("backward"), pack_symbol, out_symbol)
        graph = builder.compile()
        graph.submit(dict(p=packet, out=out)).wait()
        np.testing.assert_array_equal(out.to_numpy(), [2, 4, 6])
        np.testing.assert_array_equal(values.grad.to_numpy(), [2, 2, 2])
        graph.close()
