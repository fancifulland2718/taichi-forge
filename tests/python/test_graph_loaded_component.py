"""Loaded component composition resolves hidden resources from immutable instances."""

import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import kernel_impl
from taichi_forge_runtime.aot import Runtime
from tests import test_utils
from tests.python.test_aot_component_control import _ControlModel
from tests.python.test_aot_component_dense import _environment
from tests.python.test_aot_component_helpers import _HelperModel


def _export(tmp_path, record):
    root = _environment().compile_graph(record)
    artifact = ti.aot.Module()
    root.export(artifact, "stage")
    path = tmp_path / "component.tcm"
    artifact.archive(str(path))
    arch = ti.cfg.arch
    ti.reset()
    ti.init(arch=arch, offline_cache=False, enable_fallback=False)
    return path


def _count(name):
    return ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, ti.i32)


def _vector(name):
    return ti.graph.Arg(ti.graph.ArgKind.MATRIX, name, ti.types.vector(3, ti.f32))


def _read(instance, name):
    return np.frombuffer(instance.resource(name).read(), np.float32)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_loaded_components_share_code_with_safe_instance_bindings(tmp_path, monkeypatch):
    model = _ControlModel(7, 5)
    path = _export(tmp_path, model.record_control)
    with ti.aot.borrow_runtime() as runtime:
        module = runtime.load_module(path)
        component = module.component("stage")
        first = component.instantiate(dict(capacity=19, active=17))
        second = component.instantiate(dict(capacity=37, active=35))
        for instance in (first, second):
            instance.initialize(zero=component.contract["resources"])
        mode = "device" if ti.cfg.arch == ti.cuda else "preserve"
        builder = ti.graph.GraphBuilder()
        sequence = builder.create_sequential()
        for instance, suffix in ((first, "a"), (second, "b")):
            sequence.dispatch_loaded_component(
                instance,
                values=dict(gravity=_vector("g" + suffix)),
                dimensions=dict(active=_count("n" + suffix)),
                execution=mode,
            )
        builder.append(sequence)
        original = builder.compile()
        graph = original.definition.compile()
        peer = (
            ti.graph.GraphBuilder()
            .dispatch_loaded_component(
                second,
                values=dict(gravity=_vector("g")),
                dimensions=dict(active=_count("n")),
                execution=mode,
            )
            .compile()
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("Loaded component composition must not compile numerical source")

        gravity = ti.Vector([2.0, 3.0, 4.0])
        with monkeypatch.context() as guard:
            guard.setattr(kernel_impl.Kernel, "ensure_compiled", forbidden)
            binding = graph.bind(dict(ga=gravity, gb=ti.Vector([4.0, 5.0, 6.0]), na=3, nb=5))
            changed = graph.bind(dict(ga=ti.Vector([7.0, 8.0, 9.0]), gb=ti.Vector([6.0, 7.0, 8.0]), na=1, nb=0))
            peer_binding = peer.bind(dict(g=ti.Vector([1.0, 2.0, 3.0]), n=1))
            infos = binding.native_execution_info()
            assert len(infos) == 2
            assert all(info["kind"] == "loaded_aot_component" for info in infos)
            assert {info["execution_kind"] for info in infos} == {
                "cuda_graph" if ti.cfg.arch == ti.cuda else "cpu_batch"
            }
            binding.initialize()
            changed.initialize()
            peer_binding.initialize()
            gravity[0] = 1000  # Bind-time values are immutable snapshots.
            graph.submit(binding)
            graph.submit(binding)
            graph.submit(changed).wait()
            np.testing.assert_array_equal(_read(first, "x"), [19, 12, 12] + [0] * 16)
            np.testing.assert_array_equal(_read(second, "x"), [40] * 5 + [0] * 32)
            np.testing.assert_array_equal(_read(first, "y")[:1], [1])
            np.testing.assert_array_equal(_read(second, "y")[:1], [0])
            ticket = graph.submit(binding)
            first.close()  # Cold retirement must preserve accepted work and the peer.
            ticket.wait()
            with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="closed"):
                graph.submit(binding)
            peer.submit(peer_binding).wait()
            np.testing.assert_array_equal(_read(second, "x"), [61, 60, 60, 60, 60] + [0] * 32)
            module.close()
            with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="closed"):
                peer.submit(peer_binding)
        graph.close()
        original.close()
        peer.close()
        second.close()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_loaded_component_public_arrays_and_failed_preparation(tmp_path, monkeypatch):
    model = _HelperModel(7, 5)
    path = _export(tmp_path, model.record)
    with ti.aot.borrow_runtime() as runtime:
        module = runtime.load_module(path)
        component = module.component("stage")
        with Runtime(str(runtime.arch)) as foreign_runtime:
            foreign = foreign_runtime.load_module(path).component("stage")
            with foreign.instantiate(dict(capacity=19, active=17)) as instance:
                with pytest.raises(ti.TaichiRuntimeError, match="borrowed from the current Program"):
                    ti.graph.GraphBuilder().dispatch_loaded_component(instance)
        with component.instantiate(dict(capacity=19, active=17)) as instance:
            instance.initialize(zero=component.contract["resources"])
            counter = ti.ndarray(ti.i32, 1)
            counter.fill(0)
            symbols = dict(counter=ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "counter", ti.i32, ndim=1))
            with pytest.raises(ValueError, match="public value/argument"):
                ti.graph.GraphBuilder().dispatch_loaded_component(instance)
            with pytest.raises(ValueError, match="active counts"):
                ti.graph.GraphBuilder().dispatch_loaded_component(
                    instance,
                    values=dict(gravity=_vector("g")),
                    dimensions=dict(capacity=_count("n")),
                    arguments=symbols,
                )
            graph = (
                ti.graph.GraphBuilder()
                .dispatch_loaded_component(
                    instance,
                    values=dict(gravity=_vector("g")),
                    dimensions=dict(active=_count("n")),
                    arguments=symbols,
                )
                .compile()
            )
            registered = set(runtime._resources)
            for _ in range(2):
                with pytest.raises(ValueError, match="exceeds capacity"):
                    graph.bind(dict(g=ti.Vector([2.0, 3.0, 4.0]), n=20, counter=counter))
            gc.collect()
            assert runtime._resources == registered
            # Failure did not publish a prepared binding or freeze initialization.
            instance.initialize(zero=component.contract["resources"])
            if ti.cfg.arch == ti.cpu:
                rejected = (
                    ti.graph.GraphBuilder()
                    .dispatch_loaded_component(
                        instance,
                        values=dict(gravity=_vector("g")),
                        arguments=symbols,
                        execution="device",
                    )
                    .compile()
                )
                with pytest.raises(ti.TaichiRuntimeError, match="device execution rejected actual"):
                    rejected.bind(dict(g=ti.Vector([2.0, 3.0, 4.0]), counter=counter))
                rejected.close()
                gc.collect()
                assert runtime._resources == registered
                instance.initialize(zero=component.contract["resources"])
            with monkeypatch.context() as guard:
                guard.setattr(
                    kernel_impl.Kernel, "ensure_compiled", lambda *a, **kw: pytest.fail("numerical recompile")
                )
                bound = graph.bind(dict(g=ti.Vector([2.0, 3.0, 4.0]), n=5, counter=counter))
                bound.initialize()
                graph.submit(bound).wait()
                np.testing.assert_array_equal(_read(instance, "x"), [-1] * 5 + [0] * 14)
                imported = runtime.import_ndarray(counter)
                try:
                    np.testing.assert_array_equal(np.frombuffer(imported.read(), np.int32), [1])
                finally:
                    imported.close()
            graph.close()
