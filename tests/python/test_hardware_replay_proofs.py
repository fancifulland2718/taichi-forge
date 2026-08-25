import gc

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from tests.python.hardware_process_memory import ProcessMemoryPlateau


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_cufft_fixed_plan_mixed_command_replay_proof():
    if not ti.hardware.fft.is_available():
        pytest.skip("compatible cuFFT library is required")

    length = 64
    batch_count = 2
    shape = (batch_count, length, 2)
    plan = ti.hardware.fft.CufftPlan1D(length, batch_count=batch_count, transform="c2c")

    @ti.kernel
    def prepare(
        source: ti.types.ndarray(dtype=ti.f32, ndim=3),
        work: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for i, j, component in source:
            work[i, j, component] = source[i, j, component]

    @ti.kernel
    def finish(
        source: ti.types.ndarray(dtype=ti.f32, ndim=3),
        result: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for i, j, component in source:
            result[i, j, component] = source[i, j, component]

    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=3)
        for name in ("source", "work", "fft_output", "result")
    }
    recording = plan.record(input="work", output="fft_output")
    builder = ti.graph.GraphBuilder()
    builder.dispatch(prepare, args["source"], args["work"])
    builder.append_native(recording, admission="explicit")
    builder.dispatch(finish, args["fft_output"], args["result"])
    graph = builder.compile()
    assert recording.replay_mode == "stream_capture"
    assert graph._instance_debug_info["kind"] == "mixed_backend_region"
    assert len(graph._graph_stats) == 1
    assert graph._graph_stats[0]["diagnostics_counters_complete"]

    source_host = np.zeros(shape, dtype=np.float32)
    source_host[..., 0] = np.linspace(
        0.25, 2.0, batch_count * length, dtype=np.float32
    ).reshape(batch_count, length)
    source_host[..., 1] = np.linspace(
        -0.75, 0.5, batch_count * length, dtype=np.float32
    ).reshape(batch_count, length)
    expected_complex = np.fft.fft(
        source_host[..., 0] + 1j * source_host[..., 1], axis=-1
    )
    expected = np.stack((expected_complex.real, expected_complex.imag), axis=-1).astype(
        np.float32
    )

    def make_bindings():
        values = {name: ti.ndarray(ti.f32, shape=shape) for name in args}
        values["source"].from_numpy(source_host)
        return values

    bindings = make_bindings()
    process_memory = ProcessMemoryPlateau(
        "cuda-cufft-fixed-plan-mixed-command-replay", ("cuda-cufft",)
    )
    process_memory.capture("before")
    graph.run(bindings)
    for replay_index in range(999):
        graph.run(bindings)
        if replay_index == 498:
            ti.sync()
            process_memory.capture("midpoint")
    ti.sync()
    process_memory.capture("after")
    process_memory.finish(1_000)

    stats = graph._graph_stats[0]
    assert stats["captures"] == 1
    assert stats["exact_replays"] >= 999
    assert stats["patched_replays"] == 0
    assert stats["last_path"] == "cuda_exact_replay"
    np.testing.assert_allclose(
        bindings["result"].to_numpy(), expected, rtol=1e-5, atol=1e-6
    )

    cache = graph._instance._backend_executable._jit_cache
    cache._set_stable_replay_optimization(False)
    graph.run(bindings)
    ti.sync()
    assert graph._graph_stats[0]["last_path"] == "ordinary_fallback"
    assert graph._graph_stats[0]["last_fallback_reason"] == "runtime_mode"
    cache._set_stable_replay_optimization(True)

    rebound_generations = []
    for _ in range(100):
        rebound = make_bindings()
        rebound_generations.append(rebound)
        graph.run(rebound)
        graph.run(rebound)
    ti.sync()
    rebound_stats = graph._graph_stats[0]
    assert rebound_stats["patched_replays"] == 0
    assert rebound_stats["last_path"] == "cuda_exact_replay"
    assert rebound_stats["backend_replay_signature_slots"] == 2
    assert rebound_stats["backend_replay_signature_slot_capacity"] == 2
    np.testing.assert_allclose(
        rebound["result"].to_numpy(), expected, rtol=1e-5, atol=1e-6
    )

    plan.close()
    with pytest.raises(RuntimeError, match="plan has been closed"):
        graph.run(bindings)
    ti.reset()
    with pytest.raises(RuntimeError, match="compiled before ti.reset"):
        graph.run(bindings)


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_cusparse_mixed_command_replay_proof():
    if not ti.hardware.linalg.cusparse_is_available():
        pytest.skip("compatible cuSPARSE library is required")

    size = 16
    row_offsets = ti.ndarray(ti.i32, shape=size + 1)
    column_indices = ti.ndarray(ti.i32, shape=size)
    values = ti.ndarray(ti.f32, shape=size)
    row_offsets.from_numpy(np.arange(size + 1, dtype=np.int32))
    column_indices.from_numpy(np.arange(size, dtype=np.int32))
    diagonal = np.arange(1, size + 1, dtype=np.float32)
    values.from_numpy(diagonal)
    pattern = ti.linalg.SparsePattern.csr(size, size, row_offsets, column_indices)
    matrix = pattern.matrix(values)

    @ti.kernel
    def prepare(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        work: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(size):
            work[index] = 2.0 * source[index]

    @ti.kernel
    def finish(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        result: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(size):
            result[index] = source[index] + 1.0

    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("source", "work", "spmv_output", "result")
    }
    recording = ti.hardware.linalg.CusparseSpmvRecording(
        matrix, input="work", output="spmv_output"
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(prepare, args["source"], args["work"])
    builder.append_native(recording, admission="auto")
    builder.dispatch(finish, args["spmv_output"], args["result"])
    graph = builder.compile()
    assert recording.replay_mode == "stream_capture"
    assert graph._instance_debug_info["kind"] == "mixed_backend_region"
    assert len(graph._graph_stats) == 1

    source = ti.ndarray(ti.f32, shape=size)
    work = ti.ndarray(ti.f32, shape=size)
    spmv_output = ti.ndarray(ti.f32, shape=size)
    result = ti.ndarray(ti.f32, shape=size)
    source_host = np.linspace(0.25, 2.0, size, dtype=np.float32)
    source.from_numpy(source_host)
    bindings = {
        "source": source,
        "work": work,
        "spmv_output": spmv_output,
        "result": result,
    }

    # Enable private counters before the first submission so capture/replay
    # evidence is complete rather than inferred from timing.
    assert graph._graph_stats[0]["diagnostics_counters_complete"]
    process_memory = ProcessMemoryPlateau(
        "cuda-cusparse-mixed-command-replay", ("cuda-cusparse",)
    )
    process_memory.capture("before")
    graph.run(bindings)
    for replay_index in range(999):
        graph.run(bindings)
        if replay_index == 498:
            ti.sync()
            process_memory.capture("midpoint")
    ti.sync()
    process_memory.capture("after")
    process_memory.finish(1_000)

    stats = graph._graph_stats[0]
    assert stats["captures"] == 1
    assert stats["exact_replays"] >= 999
    assert stats["patched_replays"] == 0
    assert stats["last_path"] == "cuda_exact_replay"
    np.testing.assert_allclose(
        result.to_numpy(), 2.0 * diagonal * source_host + 1.0, rtol=1e-6
    )

    provider_stats = matrix._debug_runtime_stats()["operations"]
    # Provider work is issued only during warm-up and capture, never from the
    # Python replay loop or once per exact replay.
    assert provider_stats["spmv_calls"] == 2

    cache = graph._instance._backend_executable._jit_cache
    cache._set_stable_replay_optimization(False)
    graph.run(bindings)
    ti.sync()
    fallback_stats = graph._graph_stats[0]
    assert fallback_stats["last_path"] == "ordinary_fallback"
    assert fallback_stats["last_fallback_reason"] == "runtime_mode"
    assert matrix._debug_runtime_stats()["operations"]["spmv_calls"] == 3
    cache._set_stable_replay_optimization(True)

    # A changed allocation identity must never patch cuSPARSE descriptors.
    rebound_generations = []
    for _ in range(100):
        rebound = {name: ti.ndarray(ti.f32, shape=size) for name in bindings}
        rebound_generations.append(rebound)
        rebound["source"].from_numpy(source_host)
        graph.run(rebound)
        graph.run(rebound)
    ti.sync()
    rebound_stats = graph._graph_stats[0]
    assert rebound_stats["patched_replays"] == 0
    assert rebound_stats["last_path"] == "cuda_exact_replay"
    assert rebound_stats["backend_replay_signature_slots"] == 2
    assert rebound_stats["backend_replay_signature_slot_capacity"] == 2
    # The debug snapshot exposes the active slot's counters, not a lifetime
    # total across recycled slots.  Provider calls remain the exact churn
    # oracle: two calls (prewarm + capture) per new fixed binding, and no call
    # for its following replay.
    assert matrix._debug_runtime_stats()["operations"]["spmv_calls"] == 203
    np.testing.assert_allclose(
        rebound["result"].to_numpy(),
        2.0 * diagonal * source_host + 1.0,
        rtol=1e-6,
    )

    ti.reset()
    with pytest.raises(RuntimeError, match="compiled before ti.reset"):
        graph.run(bindings)


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_indirect_retained_replay_keys_command_generation(monkeypatch):
    monkeypatch.setenv("TI_VULKAN_GRAPHICS_RETAINED_REPLAY_PROOF", "1")
    if not ti.hardware.graphics.is_indirect_available():
        pytest.skip("Vulkan indirect graphics commands are unavailable")

    from tests.python.test_hardware_graphics import (
        _texture_rgb,
        _triangle_pipeline,
        _two_triangle_vertices,
    )

    pipeline = _triangle_pipeline()
    vertices = _two_triangle_vertices()
    target = ti.Texture(ti.Format.rgba8, (64, 64))
    draw = pipeline.pass_draw(
        ti.hardware.graphics.IndirectDraw(1, vertex_record_limit=6),
        vertex_buffers={0: "vertices"},
        indirect_buffer="commands",
    )
    recording = pipeline.record_pass((draw,), color="target")
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="explicit")
    graph = builder.compile()
    program = ti.lang.impl.get_runtime().prog
    before = dict(program._debug_vulkan_graphics_resource_stats())

    first_command = ti.ndarray(ti.u32, shape=4)
    first_command.from_numpy(np.array([3, 1, 0, 0], dtype=np.uint32))
    first_bindings = {
        "target": target,
        "vertices": vertices,
        "commands": first_command,
    }
    process_memory = ProcessMemoryPlateau(
        "vulkan-indirect-retained-replay", ("vulkan-graphics",)
    )
    replay_iterations = 1_000 if process_memory.enabled else 3
    process_memory.capture("before")
    for replay_index in range(replay_iterations):
        graph.run(first_bindings)
        ti.sync()
        if replay_index == replay_iterations // 2 - 1:
            process_memory.capture("midpoint")
    process_memory.capture("after")
    process_memory.finish(replay_iterations)
    first = dict(program._debug_vulkan_graphics_resource_stats())
    assert first["retained_replay_prewarms"] - before["retained_replay_prewarms"] == 1
    assert first["retained_replay_records"] - before["retained_replay_records"] == 1
    assert (
        first["retained_replay_replays"] - before["retained_replay_replays"]
        == replay_iterations - 2
    )

    second_command = ti.ndarray(ti.u32, shape=4)
    second_command.from_numpy(np.array([3, 1, 3, 0], dtype=np.uint32))
    second_bindings = {**first_bindings, "commands": second_command}
    for _ in range(3):
        graph.run(second_bindings)
        ti.sync()
    second = dict(program._debug_vulkan_graphics_resource_stats())
    assert (
        second["retained_replay_binding_misses"]
        > first["retained_replay_binding_misses"]
    )
    assert second["retained_replay_prewarms"] - first["retained_replay_prewarms"] == 1
    assert second["retained_replay_records"] - first["retained_replay_records"] == 1
    assert second["retained_replay_replays"] - first["retained_replay_replays"] == 1
    image = _texture_rgb(target)
    assert image[32, 20, 1] > 32
    assert image[32, 52].max() == 0

    pipeline.close()
    ti.sync()
    assert program._debug_vulkan_graphics_resource_stats()["retained_replay_slots"] == 0


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_bindless_retained_replay_owns_table_generations(monkeypatch):
    monkeypatch.setenv("TI_VULKAN_GRAPHICS_RETAINED_REPLAY_PROOF", "1")
    if not ti.hardware.graphics.is_bindless_buffer_available():
        pytest.skip("Vulkan bindless graphics buffers are unavailable")

    from tests.python.test_hardware_graphics import (
        _bindless_triangle_pipeline,
        _texture_rgb,
        _triangle_vertices,
    )

    def buffer(value):
        result = ti.ndarray(ti.f32, shape=4)
        result.from_numpy(np.array(value, dtype=np.float32))
        return result

    pipeline = _bindless_triangle_pipeline()
    vertices = _triangle_vertices()
    selector = ti.ndarray(ti.u32, shape=4)
    selector.from_numpy(np.array([2, 0, 0, 0], dtype=np.uint32))
    fixed_colors = (
        buffer((1.0, 0.0, 0.0, 1.0)),
        buffer((1.0, 1.0, 0.0, 1.0)),
        buffer((1.0, 0.0, 1.0, 1.0)),
    )
    draw = pipeline.pass_draw(
        ti.hardware.graphics.Draw(3),
        vertex_buffers={0: "vertices"},
        shader_buffers={
            (0, 0): ("color0", "color1", "color2", "color3"),
            (0, 1): "selector",
        },
    )
    recording = pipeline.record_pass((draw,), color="target")
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="explicit")
    graph = builder.compile()

    def bindings(target, selected):
        return {
            "target": target,
            "vertices": vertices,
            "color0": fixed_colors[0],
            "color1": fixed_colors[1],
            "color2": selected,
            "color3": fixed_colors[2],
            "selector": selector,
        }

    program = ti.lang.impl.get_runtime().prog
    blue_target = ti.Texture(ti.Format.rgba8, (64, 64))
    blue = buffer((0.0, 0.0, 1.0, 1.0))
    blue_bindings = bindings(blue_target, blue)
    before = dict(program._debug_vulkan_graphics_resource_stats())
    for _ in range(3):
        graph.run(blue_bindings)
        ti.sync()
    fixed = dict(program._debug_vulkan_graphics_resource_stats())
    assert fixed["retained_replay_prewarms"] - before["retained_replay_prewarms"] == 1
    assert fixed["retained_replay_records"] - before["retained_replay_records"] == 1
    assert fixed["retained_replay_replays"] - before["retained_replay_replays"] == 1

    # Submit two allocation generations without an intervening host wait. Each
    # retained command owns its immutable descriptor table through completion.
    green_target = ti.Texture(ti.Format.rgba8, (64, 64))
    green = buffer((0.0, 1.0, 0.0, 1.0))
    blue_ticket = graph.submit(blue_bindings)
    green_ticket = graph.submit(bindings(green_target, green))
    green_ticket.wait()
    blue_ticket.wait()
    concurrent = dict(program._debug_vulkan_graphics_resource_stats())
    assert (
        concurrent["retained_replay_binding_misses"]
        > fixed["retained_replay_binding_misses"]
    )
    assert concurrent["retained_replay_slots"] <= 2
    assert concurrent["retained_replay_submit_failures"] == 0
    blue_pixel = _texture_rgb(blue_target)[32, 32]
    green_pixel = _texture_rgb(green_target)[32, 32]
    assert blue_pixel[0] < 8 and blue_pixel[1] < 8 and blue_pixel[2] > 32
    assert green_pixel[0] < 8 and green_pixel[1] > 32 and green_pixel[2] < 8

    process_memory = ProcessMemoryPlateau(
        "vulkan-bindless-retained-generations", ("vulkan-graphics",)
    )
    generation_count = 1_000 if process_memory.enabled else 10
    process_memory.capture("before")
    churn_before = dict(program._debug_vulkan_graphics_resource_stats())
    last_bindings = None
    for generation in range(generation_count):
        selected = buffer(
            (0.0, 1.0, 0.0, 1.0) if generation % 2 else (0.0, 0.0, 1.0, 1.0)
        )
        last_bindings = bindings(blue_target, selected)
        for _ in range(3):
            graph.run(last_bindings)
            ti.sync()
        if generation == generation_count // 2 - 1:
            gc.collect()
            process_memory.capture("midpoint")
    gc.collect()
    process_memory.capture("after")
    process_memory.finish(generation_count)
    churn = dict(program._debug_vulkan_graphics_resource_stats())
    assert (
        churn["retained_replay_prewarms"] - churn_before["retained_replay_prewarms"]
        == generation_count
    )
    assert (
        churn["retained_replay_records"] - churn_before["retained_replay_records"]
        == generation_count
    )
    assert (
        churn["retained_replay_replays"] - churn_before["retained_replay_replays"]
        == generation_count
    )
    assert 1 <= churn["retained_replay_slots"] <= 2
    assert churn["retained_replay_submit_failures"] == 0

    pipeline.close()
    ti.sync()
    closed = dict(program._debug_vulkan_graphics_resource_stats())
    assert closed["retained_replay_slots"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        graph.run(last_bindings)


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fixed_binding_retained_graphics_replay_proof(monkeypatch):
    monkeypatch.setenv("TI_VULKAN_GRAPHICS_RETAINED_REPLAY_PROOF", "1")
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    from tests.python.test_hardware_graphics import (
        _texture_rgb,
        _triangle_pipeline,
        _triangle_vertices,
    )

    pipeline = _triangle_pipeline()
    source = _triangle_vertices()
    source_host = source.to_numpy()
    vertices = ti.ndarray(ti.f32, shape=15)
    marker = ti.ndarray(ti.i32, shape=1)
    target = ti.Texture(ti.Format.rgba8, (256, 256))

    @ti.kernel
    def prepare(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        vertices: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in vertices:
            vertices[index] = source[index]

    @ti.kernel
    def finish(marker: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        marker[0] += 1

    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    vertices_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "vertices", ti.f32, ndim=1)
    marker_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "marker", ti.i32, ndim=1)
    draw = pipeline.pass_draw(
        ti.hardware.graphics.Draw(3), vertex_buffers={0: "vertices"}
    )
    recording = pipeline.record_pass((draw,), color="target")
    # This is an internal proof only: the public recording/manifest contract
    # remains rerecord until both formal performance and lifecycle gates pass.
    assert recording.replay_mode == "rerecord"
    assert recording._experimental_retained_replay

    builder = ti.graph.GraphBuilder()
    builder.dispatch(prepare, source_arg, vertices_arg)
    builder.append_native(recording, admission="explicit")
    builder.dispatch(finish, marker_arg)
    graph = builder.compile()
    bindings = {
        "source": source,
        "vertices": vertices,
        "marker": marker,
        "target": target,
    }
    program = ti.lang.impl.get_runtime().prog
    before = dict(program._debug_vulkan_graphics_resource_stats())

    process_memory = ProcessMemoryPlateau(
        "vulkan-fixed-binding-retained-replay", ("vulkan-graphics",)
    )
    process_memory.capture("before")
    for replay_index in range(1_000):
        graph.run(bindings)
        ti.sync()
        if replay_index == 499:
            process_memory.capture("midpoint")
    process_memory.capture("after")
    process_memory.finish(1_000)

    fixed = dict(program._debug_vulkan_graphics_resource_stats())
    assert (
        fixed["retained_replay_attempts"] - before["retained_replay_attempts"] == 1_000
    )
    assert fixed["retained_replay_prewarms"] - before["retained_replay_prewarms"] == 1
    assert fixed["retained_replay_records"] - before["retained_replay_records"] == 1
    assert fixed["retained_replay_replays"] - before["retained_replay_replays"] == 998
    assert (
        fixed["retained_replay_busy_fallbacks"]
        == before["retained_replay_busy_fallbacks"]
    )
    assert (
        fixed["retained_replay_submit_failures"]
        == before["retained_replay_submit_failures"]
    )
    assert fixed["retained_replay_slots"] == 1
    assert fixed["retained_replay_slot_capacity"] == 2
    assert fixed["retained_replay_peak_slots"] == 1
    assert fixed["retained_replay_inflight_slots"] == 0
    assert fixed["retained_replay_last_path"] == 3
    assert marker.to_numpy()[0] == 1_000

    # Two fixed-binding packets may overlap without rerecord fallback. The
    # second bounded slot is populated on demand and both slots are ready
    # after the single terminal wait.
    burst_before = dict(program._debug_vulkan_graphics_resource_stats())
    tickets = [graph.submit(bindings) for _ in range(2)]
    tickets[-1].wait()
    burst = dict(program._debug_vulkan_graphics_resource_stats())
    new_records = (
        burst["retained_replay_records"] - burst_before["retained_replay_records"]
    )
    new_replays = (
        burst["retained_replay_replays"] - burst_before["retained_replay_replays"]
    )
    assert new_records in (0, 1)
    assert new_records + new_replays == 2
    assert (
        burst["retained_replay_busy_fallbacks"]
        == burst_before["retained_replay_busy_fallbacks"]
    )
    assert burst["retained_replay_slots"] == 1 + new_records
    assert burst["retained_replay_slot_capacity"] == 2
    assert burst["retained_replay_peak_slots"] == 1 + new_records
    assert burst["retained_replay_inflight_slots"] == 0
    assert marker.to_numpy()[0] == 1_002
    image = _texture_rgb(target)
    assert image[..., 0].max() > 32
    assert image[..., 1].max() > 32
    assert image[..., 2].max() > 32

    # Every allocation generation gets prewarm, record, and one exact replay;
    # the bounded retained set is recycled instead of growing with churn.
    rebound_generations = []
    churn_before = dict(program._debug_vulkan_graphics_resource_stats())
    for _ in range(100):
        rebound_source = ti.ndarray(ti.f32, shape=15)
        rebound_source.from_numpy(source_host)
        rebound = {
            "source": rebound_source,
            "vertices": ti.ndarray(ti.f32, shape=15),
            "marker": ti.ndarray(ti.i32, shape=1),
            "target": ti.Texture(ti.Format.rgba8, (64, 64)),
        }
        rebound_generations.append(rebound)
        for _ in range(3):
            graph.run(rebound)
            ti.sync()
    churn = dict(program._debug_vulkan_graphics_resource_stats())
    assert (
        churn["retained_replay_attempts"] - churn_before["retained_replay_attempts"]
        == 300
    )
    assert (
        churn["retained_replay_prewarms"] - churn_before["retained_replay_prewarms"]
        == 100
    )
    assert (
        churn["retained_replay_records"] - churn_before["retained_replay_records"]
        == 100
    )
    assert (
        churn["retained_replay_replays"] - churn_before["retained_replay_replays"]
        == 100
    )
    assert (
        churn["retained_replay_binding_misses"]
        - churn_before["retained_replay_binding_misses"]
        == 100
    )
    assert churn["retained_replay_slots"] == 1
    assert 1 <= churn["retained_replay_peak_slots"] <= 2
    assert churn["retained_replay_last_path"] == 3
    np.testing.assert_array_equal(rebound["vertices"].to_numpy(), source_host)
    assert rebound["marker"].to_numpy()[0] == 3

    # Hot flag removal fails back to the existing rerecord path and retires a
    # retained slot only when the ordinary pass touches its attachment.
    monkeypatch.delenv("TI_VULKAN_GRAPHICS_RETAINED_REPLAY_PROOF")
    graph.run(rebound)
    ti.sync()
    fallback = dict(program._debug_vulkan_graphics_resource_stats())
    assert fallback["retained_replay_attempts"] == churn["retained_replay_attempts"]
    assert fallback["retained_replay_slots"] == 0
    assert (
        fallback["retained_replay_invalidations"]
        == churn["retained_replay_invalidations"] + 1
    )
    assert fallback["retained_replay_last_path"] == 0

    # Close while a replay submission may still be in flight. Vulkan stream
    # ownership carries native references to completion; the retained slot is
    # synchronously detached and the pipeline retires on the compute bridge.
    monkeypatch.setenv("TI_VULKAN_GRAPHICS_RETAINED_REPLAY_PROOF", "1")
    for _ in range(3):
        graph.run(bindings)
        ti.sync()
    graph.run(bindings)
    pipeline.close()
    closing = dict(program._debug_vulkan_graphics_resource_stats())
    assert closing["retained_replay_slots"] == 0
    assert closing["live"] == 0
    ti.sync()
    closed = dict(program._debug_vulkan_graphics_resource_stats())
    assert closed["live"] == 0
    assert closed["retiring"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        graph.run(bindings)

    # Pipeline/recording/slot create-destroy churn is a separate lifecycle
    # axis from binding churn above.
    for _ in range(100):
        churn_pipeline = _triangle_pipeline()
        churn_vertices = _triangle_vertices()
        churn_target = ti.Texture(ti.Format.rgba8, (32, 32))
        churn_draw = churn_pipeline.pass_draw(
            ti.hardware.graphics.Draw(3), vertex_buffers={0: "vertices"}
        )
        churn_recording = churn_pipeline.record_pass((churn_draw,), color="target")
        churn_bindings = {"vertices": churn_vertices, "target": churn_target}
        for _ in range(3):
            churn_recording.execute(churn_bindings)
            ti.sync()
        churn_pipeline.close()
    ti.sync()
    after_pipeline_churn = dict(program._debug_vulkan_graphics_resource_stats())
    assert after_pipeline_churn["retained_replay_slots"] == 0
    assert 1 <= after_pipeline_churn["retained_replay_peak_slots"] <= 2
    assert after_pipeline_churn["live"] == 0
    assert after_pipeline_churn["retiring"] == 0
    assert after_pipeline_churn["retained_replay_submit_failures"] == 0

    reset_pipeline = _triangle_pipeline()
    reset_vertices = _triangle_vertices()
    reset_target = ti.Texture(ti.Format.rgba8, (32, 32))
    reset_recording = reset_pipeline.record_pass(
        (
            reset_pipeline.pass_draw(
                ti.hardware.graphics.Draw(3), vertex_buffers={0: "vertices"}
            ),
        ),
        color="target",
    )
    reset_builder = ti.graph.GraphBuilder()
    reset_builder.append_native(reset_recording, admission="explicit")
    reset_graph = reset_builder.compile()
    reset_bindings = {"vertices": reset_vertices, "target": reset_target}
    for _ in range(3):
        reset_graph.run(reset_bindings)
        ti.sync()
    program._debug_inject_runtime_fault(
        -4, "injected_retained_replay_failure", "injected Vulkan device loss"
    )
    with pytest.raises(RuntimeError, match="injected_retained_replay_failure"):
        reset_graph.run(reset_bindings)
    reset_pipeline.close()
    ti.reset()
    with pytest.raises(RuntimeError, match="compiled before ti.reset|previous ti.init"):
        reset_graph.run(reset_bindings)


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.vulkan, offline_cache=False, kernel_profiler=True)
def test_vulkan_retained_graphics_replay_profiler_falls_back(monkeypatch):
    monkeypatch.setenv("TI_VULKAN_GRAPHICS_RETAINED_REPLAY_PROOF", "1")
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    from tests.python.test_hardware_graphics import (
        _triangle_pipeline,
        _triangle_vertices,
    )

    pipeline = _triangle_pipeline()
    vertices = _triangle_vertices()
    target = ti.Texture(ti.Format.rgba8, (32, 32))
    recording = pipeline.record_pass(
        (
            pipeline.pass_draw(
                ti.hardware.graphics.Draw(3), vertex_buffers={0: "vertices"}
            ),
        ),
        color="target",
    )
    assert recording._experimental_retained_replay
    for _ in range(3):
        recording.execute({"vertices": vertices, "target": target})
        ti.sync()
    stats = dict(
        ti.lang.impl.get_runtime().prog._debug_vulkan_graphics_resource_stats()
    )
    assert stats["retained_replay_attempts"] == 0
    assert stats["retained_replay_slots"] == 0
    assert stats["retained_replay_last_path"] == 0
    pipeline.close()
