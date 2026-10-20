"""Explicit state snapshots preserve complete captured records at one boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("with_field", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_private_structure_snapshot_aliases_padding_and_suffix_ad(tmp_path, monkeypatch, with_field):
    record = ti.types.struct(tag=ti.i8, inner=ti.types.struct(value=ti.f32, other=ti.f32), tail=ti.i8)
    state = ti.ndarray(record, 17)
    host = np.empty(17, dtype=state.numpy_dtype)
    host.view(np.uint8)[:] = 0xA5
    host["tag"], host["tail"] = -3, 11
    host["inner"]["value"] = np.arange(17, dtype=np.float32) * .5
    host["inner"]["other"] = -9
    state.from_numpy(host)
    output = ti.ndarray(ti.f32, 17)
    bias = ti.field(ti.f32, shape=()) if with_field else None
    if with_field:
        bias[None] = 5

    @ti.kernel
    def observe(records: ti.types.ndarray(record, ndim=1), out: ti.types.ndarray(ti.f32, ndim=1)):
        for i in out:
            out[i] = records[i].inner.value
            if ti.static(with_field):
                out[i] += bias[None]

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)):
        for i in x:
            y[i] = .5 * x[i] * x[i]

    sequence = ti.algorithms.primitive_sequence().transform(
        state.field("inner.value"), state.field("inner.value"), scale=3, bias=7)
    builder = ti.graph.GraphBuilder()
    records_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "captured", record, ndim=1)
    builder._bind_internal_runtime_arg(records_arg, state)
    builder.append_native(sequence)
    builder.dispatch(observe, records_arg, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1))
    graph = builder.compile()
    caps = ([ti.DeviceCapability.spirv_has_int8, "spirv_has_int8_storage"]
            if ti.cfg.arch == ti.vulkan else [])
    failed = ti.aot.Module(caps=caps)
    with pytest.raises((ValueError, ti.TaichiRuntimeError)):
        failed.add_execution_plan("step", graph)
    assert not failed._ndarray_layouts and not failed._exported_kernels

    program = ti.lang.impl.get_runtime().prog
    original_copy = type(program).copy_ndarrays_to_host
    original_roots = type(program)._snapshot_dense_state
    transfers = []

    def copy(prog, sources, destinations, **options):
        assert sources == [state.arr] and options == {"snapshot": True}
        assert [value.nbytes for value in destinations] == [17 * 16]
        transfers.append("arrays")
        return original_copy(prog, sources, destinations, **options)

    def roots(prog, dependencies, arrays):
        assert arrays == [state.arr] and dependencies
        transfers.append("roots_and_arrays")
        return original_roots(prog, dependencies, arrays)

    module = ti.aot.Module(caps=caps)
    with monkeypatch.context() as context:
        context.setattr(type(program), "copy_ndarrays_to_host", copy)
        context.setattr(type(program), "_snapshot_dense_state", roots)
        plan = module.add_execution_plan("step", graph, snapshot_state=True)
    assert transfers == ["roots_and_arrays" if with_field else "arrays"]
    document = json.loads(plan.to_json())
    assert plan.argument_names == ("output",)
    arrays = [r for r in document["resources"] if r["kind"] == "ndarray"]
    assert len(arrays) == 1 and "layout" in arrays[0] and "dtype" not in arrays[0]
    assert module._data_payloads[arrays[0]["initial_data"]] == host.tobytes()
    assert len(module._ndarray_layouts) == 1
    np.testing.assert_array_equal(state.to_numpy(), host)
    graph.run(dict(output=output))
    np.testing.assert_array_equal(output.to_numpy(), 3 * host["inner"]["value"] + 7 + (5 if with_field else 0))
    warm = module.add_execution_plan("warm", graph, snapshot_state=True)
    assert warm.argument_names == ("output",)
    # Export did not rewrite the source or reuse the earlier mutable snapshot.
    warm_arrays = [
        resource for resource in json.loads(warm.to_json())["resources"]
        if resource["kind"] == "ndarray"
    ]
    assert len(warm_arrays) == 1
    assert warm_arrays[0]["initial_data"] != arrays[0]["initial_data"]
    assert module._data_payloads[warm_arrays[0]["initial_data"]] == state.to_numpy().tobytes()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="reverse")
    module.add_execution_plan("raw_and_ad", ti.aot.ExecutionPlan(
        document["operations"] + [
            dict(kind="copy", source=arrays[0]["name"], destination="records_out"),
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="fill32", destination="dy", value=0x3F800000),
            dict(kind="kernel", entry="energy", arguments=["output", "energy"]),
            dict(kind="kernel", entry="reverse", arguments=["output", "energy"]),
        ], resources=document["resources"], gradients=dict(output="dx", energy="dy"),
    ))
    artifact = tmp_path / "private-state.tcm"
    module.archive(str(artifact))
    graph.close()
    sequence.clear()
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(__file__).with_name("aot_private_array_snapshot_probe.py")),
             mode, arch, str(artifact), arrays[0]["layout"], str(int(with_field))],
            env=os.environ.copy(), capture_output=True, text=True, timeout=90,
        )
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert not any(m in text for m in ("Validation Error", "SYNC-HAZARD", "cleanup failed")), text


@test_utils.test(arch=ti.cpu, offline_cache=False, cpu_max_num_threads=4)
def test_snapshot_batches_tensor_state_and_keeps_named_inputs_public(tmp_path, monkeypatch):
    scalar = ti.ndarray(ti.f32, 3)
    tensor = ti.Matrix.ndarray(2, 2, ti.f32, 3)
    scalar.from_numpy(np.array([1, 2, 3], np.float32))
    tensor_host = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    tensor.from_numpy(tensor_host)

    @ti.kernel
    def update(x: ti.types.ndarray(ti.f32, ndim=1),
               matrix: ti.types.ndarray(ti.types.matrix(2, 2, ti.f32), ndim=1),
               out: ti.types.ndarray(ti.f32, ndim=1)):
        for i in out:
            x[i] += matrix[i][1, 0]
            out[i] = x[i]

    builder = ti.graph.GraphBuilder()
    x_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1)
    matrix_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "matrix", tensor.element_type, ndim=1)
    out_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1)
    builder._bind_internal_runtime_arg(x_arg, scalar)
    builder._bind_internal_runtime_arg(matrix_arg, tensor)
    builder.dispatch(update, x_arg, matrix_arg, out_arg)
    graph = builder.compile()
    program = ti.lang.impl.get_runtime().prog
    original = type(program).copy_ndarrays_to_host
    transfers = []

    def capture(prog, sources, destinations, **options):
        transfers.append((sources, [buffer.nbytes for buffer in destinations], options))
        return original(prog, sources, destinations, **options)

    module = ti.aot.Module()
    with monkeypatch.context() as context:
        context.setattr(type(program), "copy_ndarrays_to_host", capture)
        private = module.add_execution_plan("private", graph, snapshot_state=True)
    assert transfers == [([scalar.arr, tensor.arr], [12, 48], {"snapshot": True})]
    assert private.argument_names == ("out",)
    named = module.add_execution_plan(
        "named", graph, snapshot_state=True, source_bindings={"matrix": tensor})
    assert set(named.argument_names) == {"out", "matrix"}
    named_arrays = [r for r in json.loads(named.to_json())["resources"] if r["kind"] == "ndarray"]
    assert len(named_arrays) == 1 and named_arrays[0]["dtype"] == "f32"

    # A failed transfer cannot leave an archivable half-published module.
    def fail(*args, **kwargs):
        raise RuntimeError("snapshot transfer failed")

    failed = ti.aot.Module()
    with monkeypatch.context() as context:
        context.setattr(type(program), "copy_ndarrays_to_host", fail)
        with pytest.raises(RuntimeError, match="snapshot transfer failed"):
            failed.add_execution_plan("private", graph, snapshot_state=True)
    assert not failed._data_payloads and not failed._execution_plans
    with pytest.raises((RuntimeError, ti.TaichiRuntimeError)):
        failed.archive(str(tmp_path / "failed.tcm"))

    artifact = tmp_path / "tensor-state.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        out = runtime.ndarray("f32", 3)
        with loaded.prepare_execution("private", {"out": out}) as call:
            call()
            np.testing.assert_array_equal(np.frombuffer(out.read(), np.float32), [3, 8, 13])
            call()
            np.testing.assert_array_equal(np.frombuffer(out.read(), np.float32), [5, 14, 23])
        matrix = runtime.ndarray("f32", 3, element_shape=(2, 2))
        matrix.write((tensor_host * 2).tobytes())
        with loaded.prepare_execution("named", {"out": out, "matrix": matrix}) as call:
            call()
            np.testing.assert_array_equal(np.frombuffer(out.read(), np.float32), [5, 14, 23])
        loaded.close()
    graph.close()


@test_utils.test(arch=ti.cpu, offline_cache=False, cpu_max_num_threads=4)
def test_private_structure_data_dedup_and_reconstruction_roles():
    from taichi_forge.aot._graph_export import _ExportContext
    from taichi_forge.aot._graph_source_bindings import SourceArrayBindings

    record = ti.types.struct(tag=ti.i8, value=ti.f32)
    first, equal, scratch = [ti.ndarray(record, 3) for _ in range(3)]
    different = ti.ndarray(ti.types.struct(tag=ti.i32, value=ti.f32), 3)
    module = ti.aot.Module()
    sources = SourceArrayBindings({}, {}, module)
    context = _ExportContext(module, (), source_arrays=sources)
    bindings = context.retained_arrays
    data = bytes([0xA5] * 24)
    first_name = bindings.declare_data(first, data)
    assert bindings.declare_data(equal, bytes(bytearray(data))) == first_name
    other_name = bindings.declare_data(different, data)
    assert other_name != first_name  # Equal byte size/content is not an equal record ABI.
    scratch_name = bindings.declare_scratch(scratch)
    assert bindings.declare_scratch(scratch) == scratch_name
    assert len(context.data_payloads) == 1
    assert len(context.resources) == 3 and len(sources.pending_layouts) == 2
    assert all("layout" in r and "dtype" not in r for r in context.resources)
    assert "initial_data" not in next(r for r in context.resources if r["name"] == scratch_name)
    with pytest.raises(ti.TaichiRuntimeError, match="reconstruction role"):
        bindings.declare_snapshot(scratch)
    with pytest.raises(ValueError, match="complete array layout"):
        bindings.declare_data(ti.ndarray(record, 3), data[:-1])
