import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from tests.python.hardware_process_memory import ProcessMemoryPlateau


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_cusparse_mixed_command_replay_proof(monkeypatch):
    monkeypatch.setenv("TI_CUDA_MIXED_COMMAND_REPLAY_PROOF", "1")
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

    monkeypatch.delenv("TI_CUDA_MIXED_COMMAND_REPLAY_PROOF")
    graph.run(bindings)
    ti.sync()
    fallback_stats = graph._graph_stats[0]
    assert fallback_stats["last_path"] == "ordinary_fallback"
    assert fallback_stats["last_fallback_reason"] == "runtime_mode"
    assert matrix._debug_runtime_stats()["operations"]["spmv_calls"] == 3
    monkeypatch.setenv("TI_CUDA_MIXED_COMMAND_REPLAY_PROOF", "1")

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
