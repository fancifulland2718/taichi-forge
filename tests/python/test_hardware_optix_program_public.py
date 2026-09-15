"""Public programs: managed binding, real shaders, reuse and owner retirement."""

import gc
import os
from pathlib import Path
import weakref

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.hardware import ray
from tests import test_utils

F, L, E, R = ray.OptixParameterField, ray.OptixParameterLayout, ray.OptixShaderEntry, ray.OptixSbtRecord
PARAMS = L(
    24,
    (
        F("scene", 0, kind="scene"),
        F("output", 8, kind="buffer", access="write"),
        F("bias", 16, "u32"),
        F("count", 20, "u32"),
    ),
)
HIT = L(8, (F("value", 0, "u32"), F("accept", 4, "u32")))
MISS = L(4, (F("value", 0, "u32"),), alignment=4)


def _provider():
    path = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    try:
        provider = ray.OptixProvider(provider_path=path)
    except RuntimeError as error:
        if path:
            raise
        pytest.skip(f"OptiX provider unavailable: {error}")
    if not provider.identity["feature_bits"] & (1 << 14):
        provider.close()
        pytest.skip("installed adapter predates programmable pipelines")
    return provider


def _module():
    return ray.PtxModule.from_file(Path(__file__).parent / "assets/hardware_optix_program.ptx")


def _program(provider):
    return provider.program(
        (_module(),),
        raygen={"render": E(0, "__raygen__render")},
        miss={"miss": E(0, "__miss__value")},
        hit_groups={"hit": ray.OptixHitGroup(E(0, "__closesthit__value"), E(0, "__anyhit__mask"))},
        parameters=PARAMS,
        payload_count=1,
    )


def _scene(provider):
    vertices = ti.ndarray(ti.f32, (3, 3))
    indices = ti.ndarray(ti.i32, (1, 3))
    vertices.from_numpy(np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    gas = provider.triangle_gas(vertices, indices)
    scene = provider.instance_scene((ray.OptixRayInstance(gas, custom_index=17),))
    return gas, scene


def _record(program, scene):
    miss = R("miss", layout=MISS, values={"value": 7})
    accepted = R("hit", layout=HIT, values={"value": 100, "accept": 1})
    rejected = R("hit", layout=HIT, values={"value": 200, "accept": 0})
    return program.record(
        8,
        raygen="render",
        miss=(miss, miss),
        hit=(accepted, rejected),
        parameters={"scene": "world", "output": "out", "bias": 5, "count": 8},
        scenes={"world": scene},
    )


@pytest.mark.parametrize("native_bridge", [True, False])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_public_program_reuses_owned_packet_and_rebinds_without_recompilation(native_bridge, monkeypatch):
    from taichi_forge._lib import core

    if not native_bridge:
        monkeypatch.delattr(core, "_bind_external_cuda_call")
    provider = _provider()
    gas, scene = _scene(provider)
    program = _program(provider)
    recording = _record(program, scene)
    output = ti.ndarray(ti.u32, 8)
    output.fill(999)
    launch = recording.prepare({"out": output})
    try:
        # Allocation/packing is not initialization and cannot write user data.
        np.testing.assert_array_equal(output.to_numpy(), [999] * 8)
        with pytest.raises(RuntimeError, match="initialize"):
            launch.run()
        with pytest.raises(RuntimeError, match="programs"):
            provider.close()
        with pytest.raises(RuntimeError, match="prepared"):
            scene.close()
        launch.initialize().initialize()
        assert (launch._native_call is not None) == native_bridge
        assert launch.graph_recording()._graph_source_contract()["execution"]["submission_bridge"] == (
            "native_c_abi" if native_bridge else "python_callback"
        )
        for _ in range(3):
            launch.run()
        expected = [122, 12, 12, 12, 122, 12, 12, 12]
        np.testing.assert_array_equal(output.to_numpy(), expected)
        assert launch.memory_report().known_resident_requested_bytes == 256
        assert launch.preparation_info()["pinned_host_bytes"] == 256
        if native_bridge:
            # A provider failure must escape the same native submission scope,
            # without retiring a healthy packet or leaking its storage lease.
            import ctypes as c

            message = b"injected prepared launch failure\0"

            @c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_uint64)
            def fail(handle, stream):
                return -7

            @c.CFUNCTYPE(c.c_size_t, c.c_void_p, c.c_size_t)
            def error(buffer, size):
                if buffer:
                    c.memmove(buffer, message, min(size, len(message)))
                return len(message)

            failure_call = core._bind_external_cuda_call(
                launch._runtime_prog,
                launch._storage,
                c.cast(fail, c.c_void_p).value,
                launch._handle.value,
                c.cast(error, c.c_void_p).value,
                (fail, error),
            )
            with pytest.raises(RuntimeError, match="injected prepared launch failure"):
                failure_call()
            failure_call.invalidate()
            with pytest.raises(RuntimeError, match="retired"):
                failure_call()
            launch.run()
            np.testing.assert_array_equal(output.to_numpy(), expected)
        rebound = ti.ndarray(ti.u32, 8)
        with recording.prepare({"out": rebound}) as second:
            second.initialize().run()
            np.testing.assert_array_equal(rebound.to_numpy(), expected)
        program.close()
        assert launch.closed
        assert launch.memory_report().known_resident_requested_bytes == 0
        with pytest.raises(RuntimeError, match="closed"):
            launch.run()
    finally:
        launch.close()
        program.close()
        scene.close()
        gas.close()
        provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_sbt_only_buffer_texture_owners_and_inplace_updates():
    provider = _provider()
    # No AS and no launch parameters are needed for this raygen program.
    # All entries in a module are compiled, including the trace test entries.
    program = provider.program((_module(),), raygen={"indirect": E(0, "__raygen__indirect")}, payload_count=1)
    layout = L(
        24, (F("out", 0, kind="buffer", access="write"), F("in", 8, kind="buffer"), F("tex", 16, kind="texture"))
    )
    record = program.record(8, raygen=R("indirect", layout=layout, values={"out": "out", "in": "in", "tex": "tex"}))
    output, inputs = ti.ndarray(ti.u32, 8), ti.ndarray(ti.u32, 8)
    texture = ti.Texture(ti.Format.r32f, (2, 2))
    pixels = ti.ndarray(ti.f32, (2, 2))
    pixels.fill(9)
    texture.from_ndarray(pixels)
    inputs.from_numpy(np.arange(8, dtype=np.uint32))
    input_ref, texture_ref = weakref.ref(inputs), weakref.ref(texture)
    launch = record.prepare({"out": output, "in": inputs, "tex": texture})
    del inputs, texture
    gc.collect()
    try:
        assert input_ref() is not None and texture_ref() is not None
        launch.initialize().run()
        np.testing.assert_array_equal(output.to_numpy(), np.arange(8) + 9)
        input_ref().fill(40)
        pixels.fill(2)
        texture_ref().from_ndarray(pixels)
        launch.run()
        np.testing.assert_array_equal(output.to_numpy(), [42] * 8)
        # Same object may deliberately occupy two symbolic bindings. Native
        # writable-range validation sees one merged owner, not fake overlap.
        with record.prepare({"out": output, "in": output, "tex": texture_ref()}) as alias:
            alias.initialize().run()
            np.testing.assert_array_equal(output.to_numpy(), [44] * 8)
        launch.close()
        gc.collect()
        assert input_ref() is None and texture_ref() is None
    finally:
        launch.close()
        program.close()
        provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_public_program_failed_prepare_and_runtime_reset_retire_in_owner_order():
    provider = _provider()
    gas, scene = _scene(provider)
    program = _program(provider)
    recording = _record(program, scene)
    output = ti.ndarray(ti.u32, 8)
    with pytest.raises(ValueError, match="exactly"):
        recording.prepare({})
    # Native SBT coverage failure must roll back its program and AS leases.
    bad = program.record(
        8,
        raygen="render",
        miss=recording.miss,
        hit=(),
        parameters={"scene": "world", "output": "out", "bias": 0, "count": 8},
        scenes={"world": scene},
    )
    with pytest.raises(RuntimeError, match="SBT"):
        bad.prepare({"out": output})
    launch = recording.prepare({"out": output}).initialize()
    launch.run()
    ti.reset()
    assert all(item.closed for item in (launch, program, scene, gas, provider))
    with pytest.raises(RuntimeError, match="closed|invalidated"):
        launch.run()
    launch.close()
    program.close()
    provider.close()


def test_sbt_payload_is_a_c_byte_snapshot_not_a_mutable_python_container():
    value = np.array(3, dtype=np.uint32)
    record = R("miss", layout=MISS, values={"value": value})
    value[...] = 99
    assert record.to_dict()["scalar_bytes"] == "03000000"
    with pytest.raises(ValueError, match="16-byte"):
        R("hit", layout=L(32, alignment=32))
    with pytest.raises(ValueError, match="binding"):
        R("hit", layout=L(8, (F("buffer", 0, kind="buffer"),)), values={"buffer": 12345})


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_prepared_program_graph_composes_refit_and_consumer_without_repreparation(monkeypatch):
    from taichi_forge.hardware import _optix_program_graph as graph_adapter

    provider = _provider()
    gas, scene = _scene(provider)
    program = _program(provider)
    output, result = ti.ndarray(ti.u32, 8), ti.ndarray(ti.u32, 8)
    transforms = ti.ndarray(ti.f32, (1, 12))
    launch = _record(program, scene).prepare({"out": output})
    graph = None
    try:
        with pytest.raises(RuntimeError, match="initialize"):
            launch.graph_recording()
        launch.initialize()
        command = launch.graph_recording()
        assert command.replay_mode == "rerecord"
        diagnostic = command._as_graph_native_node().compile().debug_info
        assert diagnostic["capture"] == "unavailable"
        assert diagnostic["capture_reason"] == "optix_program_launch_not_capture_supported"
        assert diagnostic["submission_bridge"] == "native_c_abi"

        @ti.kernel
        def move(t: ti.types.ndarray(ti.f32, ndim=2), shift: ti.f32):
            for i in range(12):
                t[0, i] = ti.cast(i == 0 or i == 5 or i == 10, ti.f32)
                if i == 3:
                    t[0, i] = shift

        @ti.kernel
        def consume(source: ti.types.ndarray(ti.u32, ndim=1), target: ti.types.ndarray(ti.u32, ndim=1)):
            for i in target:
                target[i] = source[i] + 1

        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            move,
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "transforms", ti.f32, ndim=2),
            ti.graph.Arg(ti.graph.ArgKind.SCALAR, "shift", ti.f32),
        )
        builder.append_native(scene.record_refit_transforms(), admission="explicit")
        builder.append_native(launch, admission="explicit")
        builder.dispatch(
            consume,
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.u32, ndim=1),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.u32, ndim=1),
        )
        graph = builder.compile()
        bound = graph.bind({"transforms": transforms, "shift": 0.0, "out": output, "result": result})
        assert bound.fast_path_qualified
        with pytest.raises(RuntimeError, match="fixed bindings"):
            bound.update(out=ti.ndarray(ti.u32, 8))

        def no_repeat(*args, **kwargs):
            raise AssertionError("Graph replay reinitialized or revalidated its fixed OptiX packet")

        for shift in (0.0, 4.0, 0.0):
            bound.update(shift=shift)
            with monkeypatch.context() as patched:
                patched.setattr(launch, "initialize", no_repeat)
                patched.setattr(launch, "_launch", no_repeat)
                patched.setattr(launch, "_run", no_repeat)
                patched.setattr(graph_adapter._PreparedProgramRecording, "validate_graph_bindings", no_repeat)
                patched.setattr(graph_adapter._PreparedProgramRecording, "prepare_graph_execute", no_repeat)
                graph.submit(bound).wait()
            expected = [123, 13, 13, 13, 123, 13, 13, 13] if shift == 0 else [13] * 8
            np.testing.assert_array_equal(result.to_numpy(), expected)
        launch.close()
        with pytest.raises(RuntimeError, match="closed|invalidated"):
            graph.run(bound)
    finally:
        if graph is not None:
            graph.close()
        launch.close()
        program.close()
        scene.close()
        gas.close()
        provider.close()


@pytest.mark.parametrize("retire", ["close", "reset"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_program_graph_feedback_inflight_and_retirement(retire, monkeypatch):
    provider = _provider()
    gas, scene = _scene(provider)
    program = _program(provider)
    output, result = ti.ndarray(ti.u32, 8), ti.ndarray(ti.u32, 8)
    output.fill(999)
    result.fill(0)
    launch = _record(program, scene).prepare({"out": output}).initialize()
    retained_call = launch._native_call
    assert retained_call is not None
    graph = None
    try:
        command = launch.graph_recording()
        assert command.replay_mode == "rerecord"

        @ti.kernel
        def consume(source: ti.types.ndarray(ti.u32, ndim=1), target: ti.types.ndarray(ti.u32, ndim=1)):
            for i in target:
                target[i] += source[i]

        builder = ti.graph.GraphBuilder()
        builder.append_native(command, admission="explicit")
        builder.dispatch(
            consume,
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.u32, ndim=1),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.u32, ndim=1),
        )
        graph = builder.compile()
        bound = graph.bind({"out": output, "result": result})
        np.testing.assert_array_equal(output.to_numpy(), [999] * 8)
        np.testing.assert_array_equal(result.to_numpy(), [0] * 8)
        with pytest.raises(RuntimeError, match="fixed bindings"):
            bound.update(out=ti.ndarray(ti.u32, 8))

        def no_reprepare(*args, **kwargs):
            raise AssertionError("Graph replay reinitialized its prepared program")

        with monkeypatch.context() as patched:
            patched.setattr(launch, "initialize", no_reprepare)
            patched.setattr(command, "prepare_graph_execute", no_reprepare)
            patched.setattr(command, "validate_graph_bindings", no_reprepare)
            submissions = [graph.submit(bound) for _ in range(4)]
            for submission in submissions:
                submission.wait()
        expected = np.array([122, 12, 12, 12, 122, 12, 12, 12], np.uint32)
        np.testing.assert_array_equal(result.to_numpy(), expected * 4)
        # A consumer's CUDA replay must not imply that the OptiX action was
        # captured. The report keeps the two execution segments separate.
        segments = graph.execution_stats().segments
        assert len(segments) == 2
        assert not segments[0].backend_replay_path
        assert segments[1].backend_replay_path
        # Close/reset handles outstanding native work and retires the Graph
        # before freeing the prepared SBT or pipeline addresses.
        graph.submit(bound)
        if retire == "close":
            program.close()
            assert program.closed and launch.closed
        else:
            ti.reset()
            assert all(item.closed for item in (launch, program, scene, gas, provider))
        with pytest.raises(RuntimeError, match="closed|reset|reinitialization"):
            graph.run(bound)
        with pytest.raises(RuntimeError, match="retired"):
            retained_call()
    finally:
        if graph is not None:
            graph.close()
        launch.close()
        program.close()
        scene.close()
        gas.close()
        provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_program_graph_sbt_field_texture_inputs_and_explicit_graph_close():
    provider = _provider()
    program = provider.program((_module(),), raygen={"indirect": E(0, "__raygen__indirect")}, payload_count=1)
    layout = L(
        24, (F("out", 0, kind="buffer", access="write"), F("in", 8, kind="buffer"), F("tex", 16, kind="texture"))
    )
    recording = program.record(8, raygen=R("indirect", layout=layout, values={"out": "out", "in": "in", "tex": "tex"}))
    inputs, output = ti.field(ti.u32, 8), ti.ndarray(ti.u32, 8)
    inputs.fill(3)
    pixels = ti.ndarray(ti.f32, (2, 2))
    pixels.fill(9)
    texture = ti.Texture(ti.Format.r32f, (2, 2))
    texture.from_ndarray(pixels)
    bindings = {"out": output, "in": inputs, "tex": texture}
    launch = recording.prepare(bindings).initialize()
    graph = None
    try:
        builder = ti.graph.GraphBuilder()
        builder.append_native(launch, admission="explicit")
        graph = builder.compile()
        bound = graph.bind(bindings)
        for value in (3, 7):
            inputs.fill(value)
            graph.run(bound)
            np.testing.assert_array_equal(output.to_numpy(), [value + 9] * 8)
        pixels.fill(2)
        texture.from_ndarray(pixels)
        graph.run(bound)
        np.testing.assert_array_equal(output.to_numpy(), [9] * 8)
        graph.close()
        assert not launch.closed
        launch.run()  # the caller's original direct packet is still usable
        np.testing.assert_array_equal(output.to_numpy(), [9] * 8)
    finally:
        if graph is not None:
            graph.close()
        launch.close()
        program.close()
        provider.close()
