"""Grouped overwrite reconstruction preserves native strategy and resources."""

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
from tests.python.aot_scan_probe import reject


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4
)
def test_grouped_reduce_sequence_runtime_only_and_borrowed(tmp_path):
    module = ti.aot.Module()
    arch = ti.lang.impl.current_cfg().arch
    cases = []
    configurations = [
        (dtype, 17) for dtype in (ti.i32, ti.u32, ti.f32, ti.i64, ti.u64, ti.f64)
    ]
    if arch == ti.cpu:
        configurations.append((ti.f32, 262147))
    for dtype, n in configurations:
        value_type = ti.algorithms._algorithms._grouped_reduce_value_type(dtype)
        if arch == ti.vulkan and not ti.lang.impl.get_runtime().prog.vulkan_grouped_reduce_atomic_value_type_available(
            value_type
        ):
            continue
        bound = 3 if n > 17 else 7
        src, indices, dst = (
            ti.ndarray(dtype, n),
            ti.ndarray(ti.i32, n),
            ti.ndarray(dtype, bound),
        )
        numpy_type = np.dtype(
            str(dtype).replace("i", "int").replace("u", "uint").replace("f", "float")
        )
        initial = np.full(bound, 37, numpy_type)
        sequence = ti.algorithms.primitive_sequence().grouped_reduce(indices, src, dst)
        samples = []
        for iteration in (0, 1, 2):
            if n > 17:
                source = np.zeros(n, np.float32)
                index = np.zeros(n, np.int32)
                for part in range(4):
                    begin, end = n * part // 4, n * (part + 1) // 4
                    source[begin : begin + 2] = [1e20, 1]
                    source[end - 2 : end] = [-1e20, iteration + 1]
                expected = np.zeros_like(initial)
                expected[0] += 4 * (iteration + 1)
            else:
                source = (np.arange(n, dtype=np.int32) % 11).astype(numpy_type)
                index = (np.arange(n, dtype=np.int32) + iteration) % 5
                index[0], index[-1] = -1, bound
                if np.issubdtype(numpy_type, np.integer):
                    source[1:4] = np.iinfo(numpy_type).max
                elif iteration:
                    source[1:4] = [np.nan, np.inf, -np.inf]
                    index[1:4] = [1, 2, 2]
                expected = np.zeros_like(initial)
                valid = (index >= 0) & (index < bound)
                with np.errstate(invalid="ignore"):
                    np.add.at(expected, index[valid], source[valid])
            if iteration == 2:
                index.fill(-1)
                expected.fill(0)
            src.from_numpy(source)
            indices.from_numpy(index)
            dst.from_numpy(initial)
            if dtype in (ti.f32, ti.i32):
                sequence.run()
                np.testing.assert_equal(dst.to_numpy(), expected)
            samples.append(
                dict(
                    source=source.tobytes().hex(),
                    indices=index.tobytes().hex(),
                    expected=expected.tobytes().hex(),
                )
            )
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        graph = builder.compile()
        name = f"grouped_{dtype}_{n}"
        before = tuple(a.to_numpy().tobytes() for a in (src, indices, dst))
        plan = module.add_execution_plan(
            name, graph, source_bindings=dict(src=src, indices=indices, dst=dst)
        )
        assert before == tuple(a.to_numpy().tobytes() for a in (src, indices, dst))
        entry = json.loads(plan.to_json())["operations"][0]
        assert entry["operation"] == "grouped_reduce"
        if arch == ti.cpu:
            assert entry["parameters"]["cpu_max_threads"] == 4
        cases.append(
            dict(
                name=name,
                dtype=str(dtype),
                size=n,
                num_groups=bound,
                initial=initial.tobytes().hex(),
                samples=samples,
            )
        )
        graph.close()
        sequence.clear()
    assert not module._exported_kernels
    artifact = tmp_path / "grouped-add.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        assert not any(
            name.endswith((".bc", ".spv", ".ptx", ".cubin"))
            for name in archive.namelist()
        )
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf8")
    backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_grouped_reduce_probe.py")),
                mode,
                backend,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "Validation Error" not in output and "SYNC-HAZARD" not in output, output


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_grouped_reduce_offset_prefix_aliases_and_rebinding(tmp_path):
    n, bound = 13, 7
    module = ti.aot.Module()
    for call_name, count in (("step", n), ("empty", 0)):
        module.add_execution_plan(
            call_name,
            ti.aot.ExecutionPlan(
                [
                    dict(
                        kind="native",
                        provider="builtin",
                        operation="grouped_reduce",
                        version=1,
                        parameters=dict(size=count, num_groups=bound, dtype="i32"),
                        bindings=dict(values="source", keys="index", output="target"),
                    )
                ],
                resources=[
                    dict(
                        kind="ndarray_view", name=name, array=outer, offset=1, shape=[size]
                    )
                    for name, outer, size in (
                        ("source", "src", 17),
                        ("index", "indices", 15),
                        ("target", "dst", 11),
                    )
                ],
            ),
        )
    artifact = tmp_path / "grouped-views.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        src, indices, dst, other = (runtime.ndarray("i32", 19) for _ in range(4))
        source = np.arange(19, dtype=np.int32) - 8
        src.write(source.tobytes())
        frame = dict(src=src, indices=indices, dst=dst)
        for alias in (src, indices):
            reject(lambda: loaded.prepare_execution("step", dict(frame, dst=alias)))
        assert src.read() == source.tobytes()
        with (
            loaded.prepare_execution("step", frame) as call,
            loaded.prepare_execution("step", dict(frame, dst=other)) as second,
        ):
            for shift in (0, 2):
                index_data = (np.arange(19, dtype=np.int32) + shift) % 9 - 1
                indices.write(index_data.tobytes())
                for execution, target in ((call, dst), (second, other)):
                    expected = np.full(19, 37, np.int32)
                    target.write(expected.tobytes())
                    execution()
                    expected[1 : bound + 1] = 0
                    for i, key in enumerate(index_data[1 : n + 1]):
                        if 0 <= key < bound:
                            expected[1 + key] += source[1 + i]
                    np.testing.assert_array_equal(
                        np.frombuffer(target.read(), np.int32), expected
                    )
                assert (
                    indices.read() == index_data.tobytes()
                    and src.read() == source.tobytes()
                )
        with loaded.prepare_execution("empty", frame) as empty:
            expected = np.full(19, 37, np.int32)
            dst.write(expected.tobytes())
            empty()
            expected[1 : bound + 1] = 0
            np.testing.assert_array_equal(np.frombuffer(dst.read(), np.int32), expected)
        for a in (src, indices, dst, other):
            a.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_grouped_reduce_snapshot_and_explicit_adjoint(tmp_path):
    src, indices, dst = ti.field(ti.f32, 5), ti.field(ti.i32, 5), ti.field(ti.f32, 4)
    source = np.arange(5, dtype=np.float32) + 0.5
    index_data = np.array([0, 2, 1, 2, 9], np.int32)
    src.from_numpy(source)
    indices.from_numpy(index_data)

    @ti.kernel
    def initialize():
        for i in dst:
            dst[i] = 5.0

    @ti.kernel
    def publish(
        output: ti.types.ndarray(ti.f32, 1), gradient: ti.types.ndarray(ti.f32, 1)
    ):
        for i in range(4):
            output[i] = dst[i] * dst[i]
        for j in range(5):
            gradient[j] = 0
            if 0 <= indices[j] < 4:
                gradient[j] = 2.0 * dst[indices[j]]

    sequence = ti.algorithms.primitive_sequence().grouped_reduce(indices, src, dst)
    sequence.run()
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize)
    builder.append_native(sequence)
    builder.dispatch(
        publish,
        *(
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
            for name in ("output", "gradient")
        ),
    )
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("step", graph, snapshot_state=True)
    artifact = tmp_path / "grouped-snapshot.tcm"
    module.archive(str(artifact))
    graph.close()
    sequence.clear()
    src.fill(99)
    indices.fill(0)
    expected = np.zeros(4, np.float32)
    np.add.at(expected, index_data[:4], source[:4])
    grad = np.zeros(5, np.float32)
    grad[:4] = 2 * expected[index_data[:4]]
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        output, gradient = runtime.ndarray("f32", 4), runtime.ndarray("f32", 5)
        with loaded.prepare_execution(
            "step", dict(output=output, gradient=gradient)
        ) as call:
            for _ in range(2):
                call()
                np.testing.assert_array_equal(
                    np.frombuffer(output.read(), np.float32), expected**2
                )
                np.testing.assert_array_equal(
                    np.frombuffer(gradient.read(), np.float32), grad
                )
        output.close()
        gradient.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_grouped_reduce_export_rejects_changed_strategy():
    src, indices, dst = (ti.ndarray(ti.i32, 9) for _ in range(3))
    indices.fill(0)
    sequence = ti.algorithms.primitive_sequence().grouped_reduce(indices, src, dst)
    sequence.run()
    sequence._calls[0].plan.method_name = "unreconstructible"
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    with pytest.raises(ti.TaichiRuntimeError, match="selected cached strategy"):
        ti.aot.Module().add_execution_plan(
            "bad", graph, source_bindings=dict(src=src, indices=indices, dst=dst)
        )
    graph.close()
    sequence.clear()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_grouped_reduce_cuda_aliases_preserve_selected_atomic_strategy(tmp_path):
    keys, values, output = ti.ndarray(ti.i32, 7), ti.ndarray(ti.f32, 7), ti.ndarray(ti.f32, 3)
    key_data = np.array([0, 1, 2, 1, -1, 3, 0], np.int32)
    value_data = np.arange(7, dtype=np.float32) + 1
    keys.from_numpy(key_data)
    values.from_numpy(value_data)
    module = ti.aot.Module()
    methods = ("cuda_device", "cuda_segmented", "cuda_two_level", "segmented", "two_level")
    for method in methods:
        sequence = ti.algorithms.primitive_sequence().grouped_reduce(keys, values, output, method=method)
        sequence.run()
        before = output.to_numpy().copy()
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        graph = builder.compile()
        module.add_execution_plan(method, graph, source_bindings=dict(keys=keys, values=values, output=output))
        np.testing.assert_array_equal(output.to_numpy(), before)
        graph.close()
        sequence.clear()
    artifact = tmp_path / "aliases.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        k, v, out = runtime.ndarray("i32", 7), runtime.ndarray("f32", 7), runtime.ndarray("f32", 3)
        k.write(key_data.tobytes())
        v.write(value_data.tobytes())
        for method in methods:
            with loaded.prepare_execution(method, dict(keys=k, values=v, output=out)) as call:
                assert call.info["workspace_bytes"] == 0
                call()
                np.testing.assert_array_equal(np.frombuffer(out.read(), np.float32), [8, 6, 3])
        for array in (k, v, out):
            array.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_grouped_reduce_export_preserves_vulkan_bucket_strategy_boundary():
    keys, values, output = (ti.ndarray(ti.i32, n) for n in (7, 7, 3))
    keys.fill(0)
    values.fill(1)
    output.fill(37)
    for warmed in (False, True):
        sequence = ti.algorithms.primitive_sequence().grouped_reduce(
            keys, values, output, method="vulkan_segmented"
        )
        if warmed:
            sequence.run()
        before = output.to_numpy().copy()
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        graph = builder.compile()
        with pytest.raises(ti.TaichiRuntimeError, match="requested native strategy"):
            ti.aot.Module().add_execution_plan(
                "bucket", graph, source_bindings=dict(keys=keys, values=values, output=output)
            )
        np.testing.assert_array_equal(output.to_numpy(), before)
        graph.close()
        sequence.clear()
