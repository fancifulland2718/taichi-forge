"""Independent component bindings reuse native domain ownership/retirement."""

import gc
import struct
import threading
import weakref

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils
from tests.python.test_aot_component_dense import _environment
from tests.python.test_aot_component_graph import _RootModel


def _export_external_inputs(tmp_path):
    from taichi_forge.aot._component import Component
    from tests.python.test_aot_component_borrowed import _InputSource, _InputReader

    source = _InputSource(1, 2)
    reader = _InputReader(source)
    module = ti.aot.Module(ti.lang.impl.current_cfg().arch)
    Component.from_owner(source).compile_graph(
        source.record, host_resources=(source.neighbor, source.padding_state)
    ).export(module, "source")
    Component.from_owner(reader).compile_graph(
        reader.record, initializer=reader.record
    ).export(module, "reader")
    path = tmp_path / "external-lifetime.tcm"
    module.archive(str(path))
    return path


def _external_source(loaded, n):
    source = loaded.component("source").instantiate(dict(n=n, padding=3))
    source.initialize(dict(
        payload=struct.pack(f"<{n}f", *range(n)),
        neighbor=struct.pack(f"<{n}f", *([-7] * n)),
        padding_state=struct.pack("<3i", 17, 17, 17),
    ))
    return source


def _external_reader(loaded, source, n):
    return loaded.component("reader").instantiate(
        dict(count=n), inputs={"state": source.borrow("payload")}
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_external_field_cold_initializer_rechecks_source_generation(tmp_path, monkeypatch):
    path = _export_external_inputs(tmp_path)
    with Runtime("cpu") as runtime, runtime.load_module(path) as loaded:
        with _external_source(loaded, 7) as source, _external_reader(loaded, source, 7) as reader:
            prepare = loaded.prepare_execution
            injected = []

            def reset_before_retention(*args, **kwargs):
                # The reader has validated and frozen its bindings, but has not
                # yet registered a source use. Reproduce that unlocked window
                # deterministically without a device submission or a race timer.
                source.initialize(zero=("payload",))
                injected.append(True)
                return prepare(*args, **kwargs)

            monkeypatch.setattr(loaded, "prepare_execution", reset_before_retention)
            with pytest.raises(RuntimeError, match="stale"):
                reader.initialize_state()
            assert injected == [True]
            assert not reader._model_initialized and not reader._transfer_active
            assert not source._has_live_bindings()
            # Rejection must precede initializer execution, even though native
            # cold preparation itself was allowed to finish.
            assert not any(reader._resources["output"].read())
            monkeypatch.setattr(loaded, "prepare_execution", prepare)
            with _external_reader(loaded, source, 7) as fresh:
                fresh.initialize_state()
                draw = fresh.prepare()
                source.retire_bindings()
                assert draw._handle is None
                source.initialize(dict(payload=struct.pack("<7f", *range(7))))
                assert struct.unpack("<7f", source.resource("payload").read()) == tuple(range(7))
                with pytest.raises(RuntimeError, match="stale"):
                    fresh.prepare()
                draw.close()


@pytest.mark.parametrize("independent_domain", [False, True], ids=["ordinary", "domain"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_external_field_initializer_preserves_source_ordering(tmp_path, independent_domain):
    from contextlib import ExitStack, closing

    path = _export_external_inputs(tmp_path)
    with ExitStack() as scope:
        runtime = scope.enter_context(Runtime("cuda"))
        loaded = scope.enter_context(runtime.load_module(path))
        domain = (scope.enter_context(runtime.execution_domain(capacity=8, instance_capacity=6))
                  if independent_domain else None)
        sources = [scope.enter_context(_external_source(loaded, n)) for n in (7, 13)]
        steps = [scope.enter_context(source.prepare(execution_domain=domain)) for source in sources]
        for call in steps:
            call.initialize()
        readers = [scope.enter_context(_external_reader(loaded, source, n))
                   for source, n in zip(sources, (7, 13))]
        # Source calls already exist. Cold initialization must reuse their
        # ordering lane and must not close the caller's execution domain.
        for reader, n in zip(readers, (7, 13)):
            reader.initialize_state()
            assert struct.unpack(f"<{n}f", reader.resource("output").read()) == tuple(2 * i for i in range(n))
        if domain is not None:
            assert domain.info["accepting"]
        with runtime.execution_domain() as wrong:
            with pytest.raises(ValueError, match="same ordered|one ordered"):
                readers[0].prepare(execution_domain=wrong)
            with pytest.raises(ValueError, match="same ordered|one ordered"):
                sources[0].prepare(execution_domain=wrong)
        if domain is not None:
            with pytest.raises(ValueError, match="same ordered|one ordered"):
                readers[0].prepare()
        draws = [scope.enter_context(reader.prepare(execution_domain=domain)) for reader in readers]
        for call in draws:
            call.initialize()

        def replay(call):
            if domain is None:
                call.run()
                runtime.wait()
            else:
                ticket = call.submit()
                assert ticket is not None
                with closing(ticket):
                    ticket.wait()

        for _ in range(2):
            for step, draw in zip(steps, draws):
                replay(step)
                replay(draw)
        for reader, n in zip(readers, (7, 13)):
            assert struct.unpack(f"<{n}f", reader.resource("output").read()) == tuple(2 * (i + 2) for i in range(n))
        steps[0].close()
        # Only the reader's prepared call now keeps the source state live.
        with pytest.raises(RuntimeError, match="cold boundary"):
            sources[0].initialize(zero=("payload",))
        sources[0].retire_bindings()
        assert draws[0]._handle is None
        sources[0].initialize(zero=("payload",))
        readers[0].close()
        with sources[0].prepare(execution_domain=domain) as step:
            step.initialize()
            replay(step)
        assert struct.unpack("<7f", sources[0].resource("payload").read()) == (1,) * 7
        assert struct.unpack("<7f", sources[0].resource("neighbor").read()) == (-7,) * 7


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_external_field_source_retirement_and_released_call_owners(tmp_path):
    path = _export_external_inputs(tmp_path)
    with Runtime("cuda") as runtime, runtime.load_module(path) as loaded:
        with runtime.execution_domain(capacity=8, instance_capacity=4) as domain:
            source, peer = _external_source(loaded, 7), _external_source(loaded, 13)
            reader = _external_reader(loaded, source, 7)
            reader.initialize_state()
            draw = reader.prepare(execution_domain=domain)
            peer_call = peer.prepare(execution_domain=domain)
            draw.initialize()
            peer_call.initialize()
            tickets = [draw.submit()]
            assert tickets[0] is not None
            entering, proceed = threading.Event(), threading.Event()
            errors = []
            original = runtime._api.call

            def observe(function, *args):
                if function is domain._functions["retire_instance"] and not entering.is_set():
                    entering.set()
                    if not proceed.wait(10):
                        raise RuntimeError("test did not release borrowed retirement boundary")
                return original(function, *args)

            def close_source():
                try:
                    source.close()
                except BaseException as error:
                    errors.append(error)

            runtime._api.call = observe
            worker = threading.Thread(target=close_source)
            worker.start()
            try:
                assert entering.wait(10)
                acquired = runtime._lock.acquire(timeout=2)
                assert acquired, "Source retirement held the Runtime submission lock"
                runtime._lock.release()
                with pytest.raises(RuntimeError, match="closed|retiring"):
                    draw.submit()
                tickets.append(peer_call.submit())
                assert tickets[-1] is not None
            finally:
                proceed.set()
                worker.join(10)
                runtime._api.call = original
            assert not worker.is_alive() and not errors
            assert tickets[0].done()
            tickets[-1].wait()
            assert struct.unpack("<13f", peer.resource("payload").read()) == tuple(i + 1 for i in range(13))
            reader.close()
            draw.close()
            peer_call.close()
            peer.close()
            for ticket in tickets:
                ticket.close()

            # Host-handle release must retain both component owners until the
            # existing domain retirement boundary, not merely the ndarray.
            retained_source = _external_source(loaded, 7)
            retained_reader = _external_reader(loaded, retained_source, 7)
            retained_reader.initialize_state()
            released = retained_reader.prepare(execution_domain=domain)
            released.initialize()
            ticket = released.submit()
            assert ticket is not None
            released.release()
            with pytest.raises(RuntimeError, match="cold boundary"):
                retained_source.initialize(zero=("payload",))
            source_ref, reader_ref = weakref.ref(retained_source), weakref.ref(retained_reader)
            del retained_source, retained_reader, released
            gc.collect()
            assert source_ref() is not None and reader_ref() is not None
            ticket.wait()
        gc.collect()
        assert source_ref() is None and reader_ref() is None
        ticket.close()


def _export(tmp_path):
    model = _RootModel(7, 7)
    root = _environment().compile_graph(model.record)
    artifact = ti.aot.Module(ti.lang.impl.current_cfg().arch)
    root.export(artifact, "step")
    path = tmp_path / "component.tcm"
    artifact.archive(str(path))
    return path


def _instance(component, capacity, active):
    instance = component.instantiate(dict(capacity=capacity, active=active))
    instance.initialize(zero=component.contract["resources"])
    return instance


def _x(instance):
    return struct.unpack(f"<{instance.dimensions['capacity']}f", instance.resource("x").read())


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_component_domain_bindings_retire_without_holding_runtime_lock(tmp_path):
    path = _export(tmp_path)
    with Runtime("cuda") as runtime:
        module = runtime.load_module(path)
        component = module.component("step")
        with runtime.execution_domain(capacity=8, instance_capacity=3) as domain:
            first, peer = _instance(component, 41, 37), _instance(component, 257, 251)
            gravity = [2, 3, 4]
            first_call = first.prepare(values=dict(gravity=gravity), execution_domain=domain)
            peer_call = peer.prepare(values=dict(gravity=[5, 7, 9]), execution_domain=domain)
            gravity[0] = 99  # Already prepared argument snapshots cannot change.
            assert first_call._owners[0] is peer_call._owners[0] is module
            assert first.resource("x")._memory._handle != peer.resource("x")._memory._handle
            for call in (first_call, peer_call):
                call.initialize()
                assert call.info["execution_kind"] == "cuda_graph"
            tickets = [first_call.submit(), peer_call.submit(), first_call.submit()]
            assert all(ticket is not None for ticket in tickets)
            with runtime.execution_domain() as another_lane:
                with pytest.raises(ValueError, match="one ordered execution domain"):
                    peer.prepare(values=dict(gravity=[1, 2, 3]), execution_domain=another_lane)
            with pytest.raises(ValueError, match="one ordered execution domain"):
                peer.prepare(values=dict(gravity=[1, 2, 3]))
            # Native preparation failure must not consume the remaining slot.
            unbound = _instance(component, 7, 3)
            with Runtime("cuda") as other:
                with other.execution_domain() as wrong_domain:
                    with pytest.raises(RuntimeError, match="runtime|owner"):
                        unbound.prepare(values=dict(gravity=[1, 2, 3]), execution_domain=wrong_domain)
            unbound.close()
            changed = first.prepare(
                values=dict(gravity=[11, 13, 17]), dimensions=dict(active=3), execution_domain=domain
            )
            changed.initialize()
            tickets.append(changed.submit(wait_for=(tickets[-1],)))
            resources_before_failure = set(runtime._resources)
            with pytest.raises(RuntimeError, match="instance capacity"):
                peer.prepare(values=dict(gravity=[1, 2, 3]), execution_domain=domain)
            assert runtime._resources == resources_before_failure
            tickets[-1].wait()
            assert _x(first) == tuple([15] * 3 + [4] * 34 + [0] * 4)
            assert _x(peer) == tuple([5] * 251 + [0] * 6)
            tickets.append(first_call.submit())

            # Pause at the real native retirement boundary. While close waits,
            # another thread must be able to submit the peer; every first-instance
            # prepared handle must already reject new submissions.
            entering, proceed = threading.Event(), threading.Event()
            errors = []
            original = runtime._api.call

            def observe(function, *args):
                if function is domain._functions["retire_instance"] and not entering.is_set():
                    entering.set()
                    if not proceed.wait(10):
                        raise RuntimeError("test did not release retirement boundary")
                return original(function, *args)

            def close_first():
                try:
                    first.close()
                except BaseException as error:
                    errors.append(error)

            runtime._api.call = observe
            worker = threading.Thread(target=close_first)
            worker.start()
            try:
                assert entering.wait(10)
                acquired = runtime._lock.acquire(timeout=2)
                assert acquired, "Component close kept Runtime lock during domain retirement"
                runtime._lock.release()
                with pytest.raises(RuntimeError, match="closed|retiring"):
                    changed.submit()
                tickets.append(peer_call.submit())
                assert tickets[-1] is not None
            finally:
                proceed.set()
                worker.join(10)
                runtime._api.call = original
            assert not worker.is_alive() and not errors
            assert tickets[-2].done()
            tickets[-1].wait()
            assert _x(peer) == tuple([10] * 251 + [0] * 6)
            # Cold reset requires retired preparations; replacement bindings
            # retain new scalar/count values without changing the peer's code.
            peer_call.close()
            peer.initialize(zero=component.contract["resources"])
            replacement = peer.prepare(
                values=dict(gravity=[19, 0, 0]), dimensions=dict(active=1), execution_domain=domain
            )
            replacement.initialize()
            tickets.append(replacement.submit())
            tickets[-1].wait()
            assert _x(peer) == tuple([19] + [0] * 256)
            peer.close()
            assert domain.info["accepting"]
            for ticket in tickets:
                ticket.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_component_released_call_keeps_instance_until_domain_retirement(tmp_path):
    path = _export(tmp_path)
    with Runtime("cuda") as runtime:
        module = runtime.load_module(path)
        component = module.component("step")
        resources_before = set(runtime._resources)
        with runtime.execution_domain(capacity=2, instance_capacity=1) as domain:
            instance = _instance(component, 41, 37)
            call = instance.prepare(values=dict(gravity=[2, 3, 4]), execution_domain=domain)
            call.initialize()
            ticket = call.submit()
            call.release()
            with pytest.raises(RuntimeError, match="cold boundary"):
                instance.initialize(zero=("x",))
            owner = weakref.ref(instance)
            del instance, call
            gc.collect()
            assert owner() is not None
            ticket.wait()
            assert _x(owner()) == tuple([2] * 37 + [0] * 4)
        gc.collect()
        assert owner() is None
        assert runtime._resources == resources_before
        ticket.close()
