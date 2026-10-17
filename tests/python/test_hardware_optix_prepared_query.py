"""Optional fixed-query ABI: immutable parameters, ordered launch and scene leases."""

from contextlib import ExitStack
import ctypes as c
import json
import os
import statistics
import time

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.hardware import _optix as optix
from taichi_forge.lang._storage_view import describe_storage
from tests import test_utils
from tests.python.test_hardware_optix_faces import _TRIANGLES, _reference
from tests.python.aot_optix_probe import RAYS


class QueryDesc(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("scene_kind", c.c_uint32),
        ("scene", c.c_void_p),
        ("ray_count", c.c_uint32),
        ("occlusion", c.c_uint32),
        ("any_hit", c.c_uint32),
        ("face_count", c.c_uint32),
        ("rays", c.c_uint64),
        ("hits", c.c_uint64),
        ("hit_indices", c.c_uint64),
        ("faces", c.c_uint64),
    ]


class QueryMemory(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("parameter_bytes", c.c_uint32),
        ("device_bytes", c.c_uint64),
    ]


class QueryApi(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("revision", c.c_uint32),
        (
            "prepare",
            c.CFUNCTYPE(
                c.c_int, c.c_void_p, c.POINTER(QueryDesc), c.POINTER(c.c_void_p)
            ),
        ),
        ("launch", c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_uint64)),
        ("destroy", c.CFUNCTYPE(c.c_int, c.c_void_p)),
        ("memory", c.CFUNCTYPE(c.c_int, c.c_void_p, c.POINTER(QueryMemory))),
    ]


def driver_function(library, name, args):
    function = getattr(library, name)
    function.argtypes, function.restype = args, c.c_int
    return function


@pytest.mark.parametrize("instanced", (False, True))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fixed_queries_ordered_refit_and_retirement(instanced, tmp_path):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit prepared-query adapter required")
    driver = c.WinDLL("nvcuda.dll") if os.name == "nt" else c.CDLL("libcuda.so.1")
    p = c.c_void_p
    create_stream = driver_function(driver, "cuStreamCreate", [c.POINTER(p), c.c_uint])
    destroy_stream = driver_function(driver, "cuStreamDestroy_v2", [p])
    synchronize = driver_function(driver, "cuStreamSynchronize", [p])
    check = lambda status: status == 0 or pytest.fail(f"CUDA status {status}")
    with ExitStack() as stack:
        provider = stack.enter_context(optix.OptixProvider(provider_path=adapter))
        api = QueryApi()
        get_api = provider._loaded.library.taichi_forge_optix_fixed_query_api
        get_api.argtypes, get_api.restype = [
            c.c_uint32,
            c.c_size_t,
            c.POINTER(QueryApi),
        ], c.c_int
        optix._invoke_checked(
            provider._loaded.api, get_api, 1, c.sizeof(api), c.byref(api)
        )
        assert provider._loaded.api.info.features & (1 << 21)

        def array(values, dtype):
            values = np.asarray(
                values,
                (
                    np.float32
                    if dtype == ti.f32
                    else np.uint64 if dtype == ti.u64 else np.int32
                ),
            )
            result = ti.ndarray(dtype, values.shape)
            result.from_numpy(values)
            return result

        vertices = array(_TRIANGLES.reshape(6, 3), ti.f32)
        indices = array(np.arange(6).reshape(2, 3), ti.i32)
        transforms = [np.eye(4, dtype=np.float32)[:3]]
        customs = [0]
        if instanced:
            gas = stack.enter_context(provider.triangle_gas(vertices, indices))
            transforms.append(transforms[0].copy())
            transforms[1][0, 3] = 4
            customs = [17, 29]
            scene = stack.enter_context(
                provider.instance_scene(
                    tuple(
                        optix.OptixRayInstance(
                            gas,
                            transform=tuple(t.ravel()),
                            custom_index=custom,
                            opaque=bool(i),
                            sbt_record_offset=2 * i,
                        )
                        for i, (t, custom) in enumerate(zip(transforms, customs))
                    )
                )
            )
        else:
            scene = stack.enter_context(provider.triangle_scene(vertices, indices))
        rays_np = np.array(RAYS, np.float32)
        first_np = rays_np.copy()
        first_np[:, 7] = (
            1.5  # At most one candidate per ray; first-hit has a unique oracle.
        )
        rays, first_rays = array(rays_np, ti.f32), array(first_np, ti.f32)
        face_table = array([1, 1], ti.i32)
        values = [rays, first_rays, face_table]
        table_storage = provider._runtime_prog._prepare_external_cuda_storage(
            (describe_storage(face_table).descriptor,), (False,)
        )
        face_records = array(
            [[table_storage.pointers[0], 2]] + ([[0, 1]] if instanced else []), ti.u64
        )
        values.append(face_records)
        outputs = []
        for kind in ("typed", "occlusion", "faces", "face_occlusion", "first"):
            occlusion = "occlusion" in kind
            hit = ti.ndarray(
                ti.i32 if occlusion else ti.f32,
                (len(RAYS),) if occlusion else (len(RAYS), 4),
            )
            ids = None if occlusion else ti.ndarray(ti.i32, (len(RAYS), 4))
            hit.fill(-99)
            offsets = (len(values), len(values) + 1 if ids is not None else None)
            values += [hit] + ([] if ids is None else [ids])
            outputs.append((kind, hit, ids, offsets))
        storage = provider._runtime_prog._prepare_external_cuda_storage(
            tuple(describe_storage(value).descriptor for value in values),
            (False,) * 4 + (True,) * (len(values) - 4),
        )
        invoke = lambda fn: provider._runtime_prog._invoke_external_cuda_prepared(
            storage, fn
        )
        queries, stream = [], p()
        costs = {}

        def create():
            start = time.perf_counter()
            check(create_stream(c.byref(stream), 1))
            for kind, _, ids, (hit_slot, id_slot) in outputs:
                faces = kind.startswith("face")
                occlusion = "occlusion" in kind
                desc = QueryDesc(
                    c.sizeof(QueryDesc),
                    int(instanced),
                    scene._scene,
                    len(RAYS),
                    int(occlusion),
                    int(occlusion or kind == "first"),
                    len(transforms) if faces else 0,
                    storage.pointers[int(kind == "first")],
                    storage.pointers[hit_slot],
                    0 if ids is None else storage.pointers[id_slot],
                    storage.pointers[3] if faces else 0,
                )
                handle = p()
                if faces:
                    # Failure after scene retention must release that lease.
                    invalid = QueryDesc.from_buffer_copy(desc)
                    invalid.face_count += 1
                    rejected = p(1)
                    assert api.prepare(provider._context, c.byref(invalid), c.byref(rejected)) == 1
                    assert rejected.value is None
                optix._invoke_checked(
                    provider._loaded.api,
                    api.prepare,
                    provider._context,
                    c.byref(desc),
                    c.byref(handle),
                )
                queries.append(handle)
                memory = QueryMemory(c.sizeof(QueryMemory))
                optix._invoke_checked(
                    provider._loaded.api, api.memory, handle, c.byref(memory)
                )
                assert (
                    memory.parameter_bytes
                    == memory.device_bytes
                    == (64 if faces else 48 if occlusion or kind == "first" else 32)
                )
            costs["prepare_ms"] = (time.perf_counter() - start) * 1000

        def launch():
            for handle in queries:
                optix._invoke_checked(
                    provider._loaded.api, api.launch, handle, stream.value
                )

        try:
            ti.sync()
            invoke(create)
            assert all(np.all(hit.to_numpy() == -99) for _, hit, _, _ in outputs)
            destroy_scene = (
                provider._loaded.api.destroy_instance_scene
                if instanced
                else provider._loaded.api.destroy_triangle_scene
            )

            def check_retained_scene():
                assert destroy_scene(scene._scene) == 8  # TI_FORGE_OPTIX_ERROR_LIFETIME

            invoke(check_retained_scene)
            for shift, rules in (
                (0, [1, 1]),
                (0, [0, 1]),
                (4, [0, 1]),
                (0.25, [1, 1]),
                (0, [1, 1]),
            ):
                moved = _TRIANGLES.copy()
                moved[:, :, 2] += shift
                vertices.from_numpy(moved.reshape(6, 3))
                if instanced:
                    gas.refit(vertices)
                    scene.refit()
                else:
                    scene.refit(vertices)
                face_table.from_numpy(np.array(rules, np.int32))
                ti.sync()
                invoke(lambda: (launch(), check(synchronize(stream))))
                for kind, hit, ids, _ in outputs:
                    face = kind.startswith("face")
                    reference = _reference(
                        first_np if kind == "first" else rays_np,
                        [
                            (
                                moved,
                                (
                                    rules
                                    if i == 0 and face
                                    else [1, 1] if face else [0, 0]
                                ),
                                t,
                                custom,
                            )
                            for i, (t, custom) in enumerate(zip(transforms, customs))
                        ],
                    )
                    expected_hits, expected_ids = reference
                    if ids is None:
                        np.testing.assert_array_equal(
                            hit.to_numpy(), expected_ids[:, 3]
                        )
                    else:
                        np.testing.assert_allclose(
                            hit.to_numpy(), expected_hits, atol=1e-5
                        )
                        np.testing.assert_array_equal(ids.to_numpy(), expected_ids)

            # Five ordered queries, measured inside the same context callback.
            # This reports adapter-call overhead, not application speedup.
            def measure():
                submits, completes = [], []
                for sample in range(24):
                    start = time.perf_counter_ns()
                    launch()
                    submitted = time.perf_counter_ns()
                    check(synchronize(stream))
                    if sample >= 8:
                        submits.append((submitted - start) / 1000)
                        completes.append((time.perf_counter_ns() - start) / 1000)
                costs.update(
                    submit_us=statistics.median(submits),
                    complete_us=statistics.median(completes),
                )

            invoke(measure)
        finally:

            def retire():
                if stream.value:
                    check(synchronize(stream))
                for handle in reversed(queries):
                    optix._invoke_checked(provider._loaded.api, api.destroy, handle)
                if stream.value:
                    check(destroy_stream(stream))

            invoke(retire)
        (tmp_path / "costs.json").write_text(json.dumps(costs, indent=2))
