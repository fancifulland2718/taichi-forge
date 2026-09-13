"""Ray plans keep physical identity and shared ownership distinct from handles."""

from contextlib import ExitStack
import json

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _factory(stack):
    ray = ti.hardware.ray
    if impl.current_cfg().arch == ti.vulkan:
        if not ray.is_opacity_micromap_available():
            pytest.skip("Vulkan micromaps unavailable")
        asset_type = ray.VulkanOpacityMicromap
        make_blas = ray.TriangleBLAS
        make_scene = ray.InstanceTLAS
        instance = ray.RayInstance
    else:
        from tests.python.test_hardware_optix_micromap import _provider

        provider = stack.enter_context(_provider())
        asset_type = ray.OptixOpacityMicromap
        make_blas = provider.triangle_gas
        make_scene = provider.instance_scene
        instance = ray.OptixRayInstance
    vertices = ti.ndarray(ti.f32, (3, 3))
    indices = ti.ndarray(ti.i32, (1, 3))
    vertices.from_numpy(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))

    def blas(bits):
        asset = None if bits is None else asset_type(bytes([bits]), [(0, 1, 2)])
        return stack.enter_context(make_blas(vertices, indices, opacity_micromap=asset))

    def scene(*blases):
        return stack.enter_context(make_scene([instance(b) for b in blases]))

    return blas, scene


def _definition(*scenes):
    builder = ti.graph.GraphBuilder()
    for scene in scenes:
        builder.append_native(scene.record_typed(4), admission="explicit")
    return builder.freeze()


@test_utils.test(arch=[ti.vulkan, ti.cuda], offline_cache=False)
def test_ray_recipe_identity_separates_aliases_plans_and_immutable_assets():
    with ExitStack() as stack:
        blas, scene = _factory(stack)
        scenes = [scene(blas(bits)) for bits in (None, None, 0xAA, 0xAA, 0x55)]
        definitions = [_definition(s) for s in scenes]
        recipes = [d.baseline_recipe for d in definitions]
        assert len({r.semantic_graph_id for r in recipes}) == 1
        assert recipes[0].recipe_id == recipes[1].recipe_id
        assert recipes[2].recipe_id == recipes[3].recipe_id
        assert len({r.planned_physical_id for r in recipes}) == 3
        for d in definitions:
            payload = json.dumps(d.planned_physical_manifest)
            assert "static_slot" in payload and "ray-plan:" in payload
            assert "vulkan-ray-" not in payload and "optix-ray-" not in payload
        # Distinct but identical scenes must not become aliases. Sharing one
        # scene twice must, and its alias slots must survive an equivalent rebuild.
        assert (
            _definition(scenes[0], scenes[0]).semantic_graph_id
            == _definition(scenes[1], scenes[1]).semantic_graph_id
        )
        assert (
            _definition(scenes[0], scenes[0]).semantic_graph_id
            != _definition(scenes[0], scenes[1]).semantic_graph_id
        )
        # Exercise the real materializer, not only descriptive hashes.
        materialized = [stack.enter_context(d.materialize()) for d in definitions[:4]]
        ids = [m.materialized_physical_id for m in materialized]
        assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]


@test_utils.test(arch=[ti.vulkan, ti.cuda], offline_cache=False)
def test_ray_memory_counts_shared_dependencies_after_owner_close(monkeypatch):
    from taichi_forge.graph._native import collect_provider_memory_reports

    with ExitStack() as stack:
        blas, scene = _factory(stack)
        geometry = blas(0xAA)
        first = scene(geometry, geometry)
        second = scene(geometry)
        graphs = []
        for scenes in ((first,), (first, second)):
            builder = ti.graph.GraphBuilder()
            for s in scenes:
                builder.append_native(s.record_typed(4), admission="explicit")
            graph = builder.compile()
            stack.callback(graph.close)
            graphs.append(graph)

        def reports(graph):
            return graph._spec.provider_memory_reports()

        def total(items):
            return sum(item.known_resident_requested_bytes for item in items)

        one, both = reports(graphs[0]), reports(graphs[1])
        rays, hits, ids = (
            ti.ndarray(ti.f32, (4, 8)),
            ti.ndarray(ti.f32, (4, 4)),
            ti.ndarray(ti.u32, (4, 4)),
        )
        rays.from_numpy(
            np.tile([0.2, 0.2, 1, 0, 0, 0, -1, 10], (4, 1)).astype(np.float32)
        )
        bound = graphs[1].bind(dict(rays=rays, hits=hits, hit_indices=ids))
        from taichi_forge.hardware import _ray_identity, _ray_memory
        from taichi_forge.graph import _native

        def not_execution_work(*args, **kwargs):
            raise AssertionError(
                "identity hashing or memory reporting entered execution"
            )

        with monkeypatch.context() as patch:
            patch.setattr(_ray_identity, "_recording_identity", not_execution_work)
            patch.setattr(
                _ray_memory, "collect_provider_memory_reports", not_execution_work
            )
            patch.setattr(
                _native, "collect_provider_memory_reports", not_execution_work
            )
            for _ in range(2):
                graphs[1].run(bound)
            second.trace_typed(rays, hits, ids)
            ti.sync()
        np.testing.assert_array_equal(ids.to_numpy()[:, 3], 1)
        # The Graph and standalone scene must expose the same reachable bytes.
        assert total(one) == first.memory_report().known_resident_requested_bytes
        assert (
            total(both) - total(one)
            == second._graph_provider_memory_report().known_resident_requested_bytes
        )
        omm = [
            c
            for r in both
            for c in r.components
            if c.name.startswith("opacity_micromap")
        ]
        assert sum(c.resident and c.requested_bytes > 0 for c in omm) == 1
        geometry_bytes = geometry.memory_report().known_resident_requested_bytes
        geometry.close()
        assert geometry.closed
        assert geometry.memory_report().known_resident_requested_bytes == geometry_bytes
        assert total(reports(graphs[1])) == total(both)
        graphs[1].run(bound)
        ti.sync()
        np.testing.assert_array_equal(ids.to_numpy()[:, 3], 1)
        first.close()
        assert geometry.memory_report().known_resident_requested_bytes == geometry_bytes
        second.close()
        assert geometry.memory_report().known_resident_requested_bytes == 0
        # Dependency accounting is pure observation, including reset: no probe,
        # replay, synchronize or native memory-status call is needed.
        ti.reset()
        assert total(collect_provider_memory_reports((geometry, first, second))) == 0


def test_provider_memory_dependency_walk_deduplicates_aliases_and_cycles():
    from taichi_forge.graph._native import collect_provider_memory_reports

    class Owner:
        def __init__(self, key):
            self.key, self.dependencies = key, ()

        def _graph_provider_memory_identity(self):
            return self.key

        def _graph_provider_memory_report(self):
            return self.key

        def _graph_provider_memory_dependencies(self):
            return self.dependencies

    parent, child, alias = Owner("parent"), Owner("child"), Owner("child")
    parent.dependencies = (child, alias)
    child.dependencies = (parent,)
    assert collect_provider_memory_reports((parent, child)) == ("parent", "child")
