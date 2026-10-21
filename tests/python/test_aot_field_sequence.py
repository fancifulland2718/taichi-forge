"""Field candidate compaction/sort/unique uses the automatic Graph exporter."""

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("prepared", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_field_candidate_sequence_export_and_native_only_snapshot(tmp_path, prepared):
    if prepared and ti.lang.impl.current_cfg().arch == ti.cpu:
        pytest.skip("Public prepared compact/unique currently requires CUDA/Vulkan")
    n = 17
    raw, compact, unique = (ti.field(ti.u64, shape=n) for _ in range(3))
    flags, payload, compact_payload, unique_payload = (ti.field(ti.i32, shape=n) for _ in range(4))
    count, payload_count, unique_count = (ti.field(ti.i32, shape=()) for _ in range(3))
    x = ti.field(ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    sentinel = 0xFFFFFFFFFFFFFFFF

    @ti.kernel
    def seed(
        source_keys: ti.types.ndarray(ti.u64, 1),
        source_flags: ti.types.ndarray(ti.i32, 1),
        source: ti.types.ndarray(ti.f32, 1),
    ):
        for i in range(n):
            raw[i], flags[i], payload[i], x[i] = source_keys[i], source_flags[i], i, source[i]
            compact[i], unique[i] = ti.u64(sentinel), ti.u64(sentinel)
            compact_payload[i], unique_payload[i], x.grad[i] = -1, -1, 0
        count[None], payload_count[None], unique_count[None] = -1, -1, -1
        loss[None], loss.grad[None] = 0, 1

    @ti.kernel
    def energy():
        for i in range(n):
            if i < unique_count[None] and unique[i] != ti.u64(sentinel):
                loss[None] += (i + 1) * x[unique_payload[i]] ** 2

    @ti.kernel
    def report(
        out_keys: ti.types.ndarray(ti.u64, 1),
        out_ids: ti.types.ndarray(ti.i32, 1),
        stats: ti.types.ndarray(ti.i32, 1),
        out_loss: ti.types.ndarray(ti.f32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
    ):
        for i in range(n):
            out_keys[i], out_ids[i], dx[i] = unique[i], unique_payload[i], x.grad[i]
        stats[0], stats[1], stats[2], out_loss[0] = count[None], payload_count[None], unique_count[None], loss[None]

    sequence = (
        ti.algorithms.primitive_sequence()
        .compact(raw, flags, compact, count)
        .compact(payload, flags, compact_payload, payload_count)
        .sort_by_key(compact, compact_payload)
        .unique_by_key(compact, compact_payload, unique, unique_payload, unique_count)
    )
    dtypes = dict(
        source_keys=ti.u64,
        source_flags=ti.i32,
        source=ti.f32,
        out_keys=ti.u64,
        out_ids=ti.i32,
        stats=ti.i32,
        out_loss=ti.f32,
        dx=ti.f32,
    )
    symbols = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1) for name, dtype in dtypes.items()}
    builder = ti.graph.GraphBuilder()
    builder.dispatch(seed, *(symbols[name] for name in ("source_keys", "source_flags", "source")))
    plans = ()
    if prepared:
        plans = (
            ti.algorithms.prepare_compact(raw, flags, compact, count),
            ti.algorithms.prepare_compact(payload, flags, compact_payload, payload_count),
            ti.algorithms.prepare_sort(compact, compact_payload),
            ti.algorithms.prepare_unique_by_key(compact, compact_payload, unique, unique_payload, unique_count),
        )
        for plan in plans:
            builder.append_native(plan.record())
    else:
        builder.append_native(sequence)
    builder.dispatch(energy)
    builder.dispatch(energy.grad)
    builder.dispatch(report, *(symbols[name] for name in ("out_keys", "out_ids", "stats", "out_loss", "dx")))
    graph = builder.compile()
    arrays = {
        name: ti.ndarray(dtype, 3 if name == "stats" else 1 if name == "out_loss" else n)
        for name, dtype in dtypes.items()
    }
    initial_keys = np.array([7, 2, 7, 1, 2, 9, 1, 7, 4, 9, 3, 4, 2, 3, 1, 9, 2], np.uint64)
    arrays["source_keys"].from_numpy(initial_keys)
    arrays["source_flags"].from_numpy(np.array([i % 3 != 0 for i in range(n)], np.int32))
    arrays["source"].from_numpy(np.arange(n, dtype=np.float32) / 8 - 1)
    graph.run(arrays)  # Also exercise export of already-warmed compact plans.
    module = ti.aot.Module(caps=["spirv_has_int64"] if ti.lang.impl.current_cfg().arch == ti.vulkan else [])
    with pytest.raises(ti.TaichiRuntimeError, match="snapshot_state"):
        module.add_execution_plan("bad", graph)
    assert not module._fields and not module._exported_kernels
    before = raw.to_numpy().tobytes(), compact.to_numpy().tobytes(), unique.to_numpy().tobytes()
    plan = module.add_execution_plan("step", graph, snapshot_state=True)
    assert before == (raw.to_numpy().tobytes(), compact.to_numpy().tobytes(), unique.to_numpy().tobytes())
    module.add_execution_plan("again", graph, snapshot_state=True)
    assert len(module._exported_kernels) == 5  # Four user entries and one shared head helper.
    assert len(module._array_fields) == 10
    document = json.loads(plan.to_json())
    # Prepared primitives use the shared native envelope; ordinary sequence
    # compatibility variants retain their existing reader representation.
    assert [op.get("operation", op["kind"]) for op in document["operations"]] == [
        "kernel",
        "compact",
        "compact",
        "sort",
        "kernel",
        "compact",
        "kernel",
        "kernel",
        "kernel",
    ]
    assert len([r for r in document["resources"] if r["kind"] == "ndarray"]) == 1
    artifact = tmp_path / "field-sequence.tcm"
    module.archive(str(artifact))

    # No compiled kernel references these roots: snapshot closure must include
    # the native operation's retained Fields on its own.
    native_sequence = ti.algorithms.primitive_sequence().sort(raw, payload)
    native_builder = ti.graph.GraphBuilder()
    native_builder.append_native(native_sequence)
    native_graph = native_builder.compile()
    native_module = ti.aot.Module()
    native = native_module.add_execution_plan("sort", native_graph, snapshot_state=True)
    doc = json.loads(native.to_json())
    native_module.add_execution_plan(
        "read",
        ti.aot.ExecutionPlan(
            [
                dict(kind="copy", source=doc["operations"][0]["keys"], destination="keys"),
                dict(kind="copy", source=doc["operations"][0]["values"], destination="ids"),
            ],
            resources=doc["resources"],
        ),
    )
    native_artifact = tmp_path / "native-only.tcm"
    native_module.archive(str(native_artifact))
    assert not native_module._exported_kernels
    with zipfile.ZipFile(native_artifact) as bundle:
        assert any("data/" in name for name in bundle.namelist()), bundle.namelist()
    graph.close()
    native_graph.close()
    sequence.clear()
    native_sequence.clear()
    for prepared_plan in reversed(plans):
        prepared_plan.close()
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_field_sequence_probe.py")),
                mode,
                arch,
                str(artifact),
                str(native_artifact),
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Unable to release" not in result.stdout + result.stderr
