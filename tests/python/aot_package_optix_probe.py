"""Source-free package OptiX factories, native Graphs and live instance refits."""
import sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np

# Only the standalone Runtime package is available; reject frontend imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"))
class NoFrontend:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("taichi", "taichi_forge") or fullname.startswith(("taichi.", "taichi_forge.")):
            raise ImportError("Numerical frontend is forbidden in this consumer")
sys.meta_path.insert(0, NoFrontend())
from taichi_forge_runtime.aot import Runtime, load_package


@contextmanager
def rejects(error, message):
    try:
        yield
    except error as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError("Expected " + error.__name__)


output, adapter, library = sys.argv[1:]
with Runtime("cuda") as runtime, load_package(runtime, output) as code:
    model = code.host.Model()
    from taichi_forge_runtime.aot._package_graph import Arg, ArgKind
    with model.provider(adapter, library) as provider:
        vertices = runtime.ndarray("f32", (3, 3))
        vertices.write(np.array([[-1,-1,0],[1,-1,0],[0,1,0]], np.float32).tobytes())
        indices = runtime.ndarray("i32", (1, 3))
        indices.write(np.array([[0,1,2]], np.int32).tobytes())
        with provider.triangle_gas(vertices, indices) as gas:
            pose = np.eye(3, 4, dtype=np.float32)
            with model.scene(provider, gas, pose) as scene, model.scene(provider, gas, pose) as other:
                rays = runtime.ndarray("f32", (1, 8))
                rays.write(np.array([[0,0,2,0,0,0,-1,100]], np.float32).tobytes())
                hits, ids = runtime.ndarray("f32", (1, 4)), runtime.ndarray("i32", (1, 4))
                transforms = runtime.ndarray("f32", (1, 3, 4))
                query = scene.record_typed(1, rays="rays", hits="hits", hit_indices="ids",
                                           scene="scene", graph_fixed_query=False)
                refit = scene.record_refit_transforms(transforms="transforms", scene="scene")
                builder = code.graph_builder()
                with rejects(ValueError, "reserved package binding name"):
                    builder.append_native(scene.record_typed(1, rays="_forge_package_graph_0_0"))
                region = code.graph_builder()
                region.append_native(refit, admission="auto")
                region.append_native(query, admission="auto")
                builder.append(region)
                unused = builder.create_sequential()
                unused.append_native(scene.record_typed(2, rays="unused_rays"), admission="auto")
                conflicting = code.graph_builder()
                conflicting.append_native(scene.record_typed(2), admission="auto")
                with rejects(ValueError, "incompatible array extents"):
                    builder.append(conflicting)
                builder.dispatch(model.publish, Arg(ArgKind.NDARRAY, "hits", "f32", ndim=2))
                definition = builder.freeze()
                with rejects(ValueError, "preserved execution queue"):
                    definition.select_execution_recipe(queue="graphics")
                with rejects(ValueError, "another definition"):
                    definition.materialize(builder.freeze().select_execution_recipe())
                with definition.materialize(definition.select_execution_recipe()) as materialized:
                    graph = materialized.executor
                    with rejects(ValueError, "explicit writable resources"):
                        graph.written_resources()
                    values = dict(scene=scene, rays=rays, hits=hits, ids=ids, transforms=transforms)
                    wrong_rays = runtime.ndarray("f32", (2, 8))
                    try:
                        with rejects(ValueError, "extent differs"):
                            graph.prepare(dict(values, rays=wrong_rays))
                    finally:
                        wrong_rays.close()
                    first = None
                    for z in (0., .5, -.5):
                        pose[2, 3] = z
                        transforms.write(pose.tobytes())
                        graph.run(values)
                        runtime.wait()
                        np.testing.assert_allclose(model.out.to_numpy(), [2-z], atol=1e-6)
                        if first is None:
                            first = graph._cached_call
                        assert graph._cached_call is first
                    query.execute(dict(scene=other, rays=rays, hits=hits, ids=ids))
                    runtime.wait()
                    np.testing.assert_allclose(np.frombuffer(hits.read(), np.float32)[0], 2, atol=1e-6)
                    query.close()
                with rejects(RuntimeError, "Graph is closed"):
                    graph.run(values)
                # A live per-primitive table and compact occlusion use the same
                # native executor. Updating either ray/table data keeps binding.
                from taichi_forge_runtime.aot._package_ray import OptixFaceRuleTable
                faces = runtime.ndarray("i32", (1,))
                occluded = runtime.ndarray("i32", (1,))
                occlusion = other.record_occlusion(1, rays="rays", occluded="out",
                    face_rules=OptixFaceRuleTable("faces"), graph_fixed_query=False)
                for face, origin, direction, expected in ((0, -2., 1., 1), (1, -2., 1., 0), (1, 2., -1., 1)):
                    faces.write(np.array([face], np.int32).tobytes())
                    rays.write(np.array([[0,0,origin,0,0,0,direction,100]], np.float32).tobytes())
                    occlusion.execute(dict(rays=rays, out=occluded, faces=faces))
                    runtime.wait()
                    assert np.frombuffer(occluded.read(), np.int32)[0] == expected
                with rejects(ValueError, "explicit writable resources"):
                    occlusion._graph.written_resources()
                occlusion.close()
                faces.close()
                occluded.close()
                for array in (rays, hits, ids, transforms):
                    array.close()
        vertices.close()
        indices.close()
print("SOURCE_FREE_OPTIX_PACKAGE_OK", flush=True)
