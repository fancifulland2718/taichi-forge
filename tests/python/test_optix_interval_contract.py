"""JIT and reconstructed AOT preserve the same OptiX ray interval inputs."""

from contextlib import ExitStack
import os

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("transform_kind", ["flat", "scaled", "mirrored"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_interval_clipping_and_endpoint_route_equivalence(tmp_path, transform_kind):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit common OptiX adapter required")

    # The plane is at world z=1.5 in every case. Keep exact endpoints in the
    # comparison, without prescribing their acceptance across hardware/drivers.
    vertices = np.array([[-1, -1, 1], [0, 1, 1], [1, -1, 1]], np.float32)
    indices = np.array([[0, 1, 2]], np.int32)
    transform = np.array([[2, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1.5, 0]], np.float32)
    if transform_kind == "flat":
        vertices[:, 2] = 1.5
    elif transform_kind == "mirrored":
        transform[0, 0] = -2
    bounds = (np.nextafter(np.float32(1.5), np.float32(0)), np.float32(1.5),
              np.nextafter(np.float32(1.5), np.float32(2)))
    rays = np.array(
        [[0, 0, 0, 0.01, 0, 0, 1, bound] for bound in bounds]
        + [[0, 0, 0, bound, 0, 0, 1, 3] for bound in bounds]
        + [[0, 0, 0, 0.01, 0, 0, 1, 3],
           [0, 0, 0, 0.01, 0, 0, 1, 1.25],
           [0, 0, 0, 1.75, 0, 0, 1, 3],
           [0, 0, 3, 0.01, 0, 0, -1, 3]], np.float32)

    def array(value, dtype):
        result = ti.ndarray(dtype, value.shape)
        result.from_numpy(value)
        return result

    with ExitStack() as stack:
        provider = stack.enter_context(ti.hardware.ray.load_optix_provider(provider_path=adapter))
        vertex_array, index_array = array(vertices, ti.f32), array(indices, ti.i32)
        if transform_kind == "flat":
            scene = stack.enter_context(provider.triangle_scene(vertex_array, index_array))
            resources = {scene: dict(vertices="vertices", indices="indices")}
        else:
            gas = stack.enter_context(provider.triangle_gas(vertex_array, index_array))
            instance = ti.hardware.ray.OptixRayInstance(gas, transform=tuple(transform[0]), custom_index=31)
            scene = stack.enter_context(provider.instance_scene((instance,)))
            resources = {gas: dict(vertices="vertices", indices="indices"), scene: dict(transforms="transforms")}

        # A two-sided device table exercises the filtering pipeline independently
        # of its acceptance rule and of the opaque fast path.
        faces = array(np.zeros(1, np.uint32), ti.u32)
        producer = ti.aot.Module()
        recordings = {}
        for policy, rules in (("opaque", None), ("filtered", (ti.hardware.ray.OptixFaceRuleTable("faces"),))):
            typed = scene.record_typed(len(rays), face_rules=rules)
            shadow = scene.record_occlusion(len(rays), face_rules=rules)
            builder = ti.graph.GraphBuilder()
            builder.append_native(typed, admission="explicit")
            builder.append_native(shadow, admission="explicit")
            graph = builder.compile()
            stack.callback(graph.close)
            producer.add_execution_plan(policy, graph, resource_bindings=resources)
            recordings[policy] = (typed, shadow)
        artifact = tmp_path / "intervals.tcm"
        producer.archive(str(artifact))

        runtime = stack.enter_context(ti.aot.borrow_runtime())
        runtime.configure_provider("optix", adapter_path=adapter)
        module = runtime.load_module(str(artifact))
        stack.callback(module.close)
        common = dict(vertices=vertices, indices=indices, rays=rays,
                      hits=np.zeros((len(rays), 4), np.float32),
                      hit_indices=np.zeros((len(rays), 4), np.int32),
                      occluded=np.zeros(len(rays), np.int32), faces=np.zeros(1, np.uint32),
                      transforms=transform)
        expected = None
        for policy, (typed, shadow) in recordings.items():
            frame = {}
            for name in module.execution_plan(policy).argument_names:
                data = common[name]
                frame[name] = runtime.ndarray({"f": "f32", "i": "i32", "u": "u32"}[data.dtype.kind], data.shape)
                stack.callback(frame[name].close)
                frame[name].write(data.tobytes())
            call = module.prepare_execution(policy, frame)
            stack.callback(call.close)
            jit_rays = array(rays, ti.f32)
            hits = ti.ndarray(ti.f32, (len(rays), 4))
            ids = ti.ndarray(ti.i32, (len(rays), 4))
            flags = ti.ndarray(ti.i32, len(rays))
            extra = dict(faces=faces) if policy == "filtered" else {}
            main = typed.prepare_graph_execute(dict(rays=jit_rays, hits=hits, hit_indices=ids, **extra))
            compact = shadow.prepare_graph_execute(dict(rays=jit_rays, occluded=flags, **extra))
            for miss in (False, True):
                values = rays.copy()
                if miss:
                    values[:, 0] = 20
                jit_rays.from_numpy(values)
                frame["rays"].write(values.tobytes())
                main()
                compact()
                call()
                runtime.wait()
                actual = (hits.to_numpy(), ids.to_numpy(), flags.to_numpy())
                for name, data in zip(("hits", "hit_indices", "occluded"), actual):
                    np.testing.assert_array_equal(np.frombuffer(frame[name].read(), data.dtype).reshape(data.shape), data)
                np.testing.assert_array_equal(actual[1][:, 3], actual[2])
                if miss:
                    np.testing.assert_array_equal(actual[0], np.tile([-1, 0, 0, 0], (len(rays), 1)))
                    np.testing.assert_array_equal(actual[1], np.tile([-1, -1, -1, 0], (len(rays), 1)))
                else:
                    np.testing.assert_array_equal(actual[2][6:], [1, 0, 0, 1])
                    np.testing.assert_allclose(actual[0][[6, 9], :3], [[1.5, .5, .25]] * 2)
                    if expected is None:
                        expected = actual
                    else:
                        for reference, data in zip(expected, actual):
                            np.testing.assert_array_equal(reference, data)
