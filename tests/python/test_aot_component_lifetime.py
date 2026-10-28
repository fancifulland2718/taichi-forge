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
