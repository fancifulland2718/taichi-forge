"""Native indexed-copy AOT shares payload, dynamic-index and ownership contracts."""

import json
import array
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_scan_probe import CODES, reject


def _parameters(dtype, size, src_items, dst_items, scatter=False, element_shape=()):
    return dict(
        size=size,
        src_items=src_items,
        dst_items=dst_items,
        dtype=dtype,
        element_shape=list(element_shape),
        op="scatter" if scatter else "gather",
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4
)
def test_indexed_copy_offset_prefix_and_physical_aliases(tmp_path):
    module = ti.aot.Module()
    for dtype, n in (("i32", 17), ("i64", 65537)):
        for scatter in (False, True):
            module.add_execution_plan(
                f"{dtype}_{scatter}",
                ti.aot.ExecutionPlan(
                    [
                        dict(
                            kind="native",
                            provider="builtin",
                            operation="indexed_copy",
                            version=1,
                            parameters=_parameters(dtype, n, n + 3, n + 4, scatter),
                            bindings=dict(src="source", indices="index", dst="target"),
                        )
                    ],
                    resources=[
                        dict(
                            kind="ndarray_view",
                            name="source",
                            array="src",
                            offset=1,
                            shape=[n + 3],
                        ),
                        dict(
                            kind="ndarray_view",
                            name="index",
                            array="indices",
                            offset=1,
                            shape=[n + 2],
                        ),
                        dict(
                            kind="ndarray_view",
                            name="target",
                            array="dst",
                            offset=1,
                            shape=[n + 4],
                        ),
                    ],
                ),
            )
    artifact = tmp_path / "indexed-views.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        for dtype, n in (("i32", 17), ("i64", 65537)):
            src, dst = runtime.ndarray(dtype, n + 8), runtime.ndarray(dtype, n + 8)
            indices = runtime.ndarray("i32", n + 8)
            source = array.array(CODES[dtype], range(n + 8))
            src.write(source)
            for scatter in (False, True):
                name = f"{dtype}_{scatter}"
                frame = dict(src=src, indices=indices, dst=dst)
                reject(lambda: loaded.prepare_execution(name, dict(frame, dst=src)))
                if dtype == "i32":
                    reject(
                        lambda: loaded.prepare_execution(name, dict(frame, dst=indices))
                    )
                with loaded.prepare_execution(name, frame) as call:
                    for shift in (0, 1):
                        index_data = [n + 2 - i - shift for i in range(n)]
                        index_data[0], index_data[-1] = -1, n + 100
                        index_bytes = array.array(
                            "i", [91] + index_data + [92] * 7
                        ).tobytes()
                        indices.write(index_bytes)
                        sentinel = array.array(CODES[dtype], [37] * (n + 8))
                        dst.write(sentinel)
                        expected = list(sentinel)
                        for i, index in enumerate(index_data):
                            if scatter:
                                if 0 <= index < n + 4:
                                    expected[index + 1] = source[i + 1]
                            else:
                                expected[i + 1] = (
                                    source[index + 1] if 0 <= index < n + 3 else 0
                                )
                        call()
                        assert (
                            dst.read() == array.array(CODES[dtype], expected).tobytes()
                        )
                        assert src.read() == source.tobytes()
                        assert indices.read() == index_bytes
            src.close()
            dst.close()
            indices.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_indexed_copy_field_snapshot_and_explicit_adjoint(tmp_path):
    src = ti.field(ti.f32, 5)
    indices = ti.field(ti.i32, 6)
    dst = ti.field(ti.f32, 6)
    source = np.arange(5, dtype=np.float32) + 0.5
    index_data = np.array([0, 1, 2, 2, 4, 1], np.int32)
    src.from_numpy(source)
    indices.from_numpy(index_data)

    @ti.kernel
    def publish(
        output: ti.types.ndarray(ti.f32, 1), gradient: ti.types.ndarray(ti.f32, 1)
    ):
        for i in range(6):
            output[i] = dst[i] * dst[i]
        for j in range(5):
            acc = 0.0
            for i in range(6):
                if indices[i] == j:
                    acc += 2.0 * dst[i]
            gradient[j] = acc  # Explicit adjoint of sum(gather(src) ** 2).

    sequence = ti.algorithms.primitive_sequence().gather(src, indices, dst)
    sequence.run()
    builder = ti.graph.GraphBuilder()
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
    artifact = tmp_path / "indexed-snapshot.tcm"
    module.archive(str(artifact))
    graph.close()
    sequence.clear()
    src.fill(99)
    indices.fill(0)
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        output, gradient = runtime.ndarray("f32", 6), runtime.ndarray("f32", 5)
        with loaded.prepare_execution(
            "step", dict(output=output, gradient=gradient)
        ) as call:
            for _ in range(2):
                call()
                np.testing.assert_array_equal(
                    np.frombuffer(output.read(), np.float32), source[index_data] ** 2
                )
                np.testing.assert_array_equal(
                    np.frombuffer(gradient.read(), np.float32),
                    np.bincount(
                        index_data, weights=2 * source[index_data], minlength=5
                    ),
                )
        output.close()
        gradient.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_indexed_copy_export_rejects_changed_strategy():
    src, dst, indices = (ti.ndarray(ti.i32, 9) for _ in range(3))
    indices.fill(0)
    sequence = ti.algorithms.primitive_sequence().gather(src, indices, dst)
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


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4
)
def test_indexed_copy_sequence_runtime_only_and_borrowed(tmp_path, monkeypatch):
    from taichi_forge.algorithms import _algorithms as algorithms
    from taichi_forge.algorithms._primitive_definitions import IndexedCopyDefinition
    from taichi_forge.graph._native_definition import NativeOperationBinding

    module = ti.aot.Module()
    cases = []
    layouts = [
        (dtype, ()) for dtype in (ti.i32, ti.u32, ti.f32, ti.i64, ti.u64, ti.f64)
    ]
    layouts += [(ti.f32, (3,)), (ti.i64, (2, 2))]
    for dtype, shape in layouts:
        for scatter in (False, True):
            src_items, dst_items = (9, 23) if scatter else (17, 9)
            payload = (
                dtype
                if not shape
                else (
                    ti.types.vector(shape[0], dtype)
                    if len(shape) == 1
                    else ti.types.matrix(*shape, dtype)
                )
            )
            src, dst = ti.ndarray(payload, src_items), ti.ndarray(payload, dst_items)
            indices = ti.ndarray(ti.i32, 9)
            numpy_type = np.dtype(
                str(dtype)
                .replace("i", "int")
                .replace("u", "uint")
                .replace("f", "float")
            )
            source = np.arange(
                src_items * int(np.prod(shape)), dtype=numpy_type
            ).reshape((src_items, *shape))
            if dtype in (ti.f32, ti.f64):
                source.reshape(-1)[0] = -0.0
                source.reshape(-1)[1] = np.nan
            elif dtype in (ti.i64, ti.u64):
                source += 1 << 40
            sentinel = np.full((dst_items, *shape), 37, dtype=numpy_type)
            src.from_numpy(source)
            dst.from_numpy(sentinel)
            mode = "scatter" if scatter else "gather"
            sequence = getattr(ti.algorithms.primitive_sequence(), mode)(
                src, indices, dst
            )
            samples = []
            for index_data in (
                np.array([16, 2, -1, 8, 20, 3, 6, 0, 99], np.int32),
                np.array([0, 1, 4, 9, 10, 12, 13, 15, 22], np.int32),
            ):
                indices.from_numpy(index_data)
                expected = sentinel.copy() if scatter else np.zeros_like(sentinel)
                for i, index in enumerate(index_data):
                    if scatter and 0 <= index < dst_items:
                        expected[index] = source[i]
                    elif not scatter and 0 <= index < src_items:
                        expected[i] = source[index]
                # Alternate cold export and concretely selected native plans.
                if not scatter:
                    dst.from_numpy(sentinel)
                    sequence.run()
                    assert dst.to_numpy().tobytes() == expected.tobytes()
                samples.append(
                    dict(
                        indices=index_data.tobytes().hex(),
                        expected=expected.tobytes().hex(),
                    )
                )
            if not scatter:

                def no_conversion(*args, **kwargs):
                    raise AssertionError("warm indexed copy must reuse its selected native call")

                with monkeypatch.context() as patch:
                    patch.setattr(algorithms, "_indexed_copy_definition", no_conversion)
                    patch.setattr(NativeOperationBinding, "lower", no_conversion)
                    sequence.run()
                assert dst.to_numpy().tobytes() == expected.tobytes()
            builder = ti.graph.GraphBuilder()
            builder.append_native(sequence)
            graph = builder.compile()
            name = f"copy_{len(cases)}"
            before = (
                src.to_numpy().tobytes(),
                dst.to_numpy().tobytes(),
                indices.to_numpy().tobytes(),
            )
            definitions = []
            bind = IndexedCopyDefinition.bind

            def observe(definition, *args, **kwargs):
                definitions.append(definition)
                return bind(definition, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(IndexedCopyDefinition, "bind", observe)
                plan = module.add_execution_plan(name, graph, source_bindings=dict(src=src, indices=indices, dst=dst))
            assert len(definitions) == 1
            if not scatter:
                assert definitions[0] is sequence._calls[0].plan.definition
            assert before == (
                src.to_numpy().tobytes(),
                dst.to_numpy().tobytes(),
                indices.to_numpy().tobytes(),
            )
            (op,) = json.loads(plan.to_json())["operations"]
            assert op["operation"] == "indexed_copy" and op["parameters"]["op"] == mode
            cases.append(
                dict(
                    name=name,
                    dtype=str(dtype),
                    element_shape=list(shape),
                    size=9,
                    src_items=src_items,
                    dst_items=dst_items,
                    scatter=scatter,
                    source=source.tobytes().hex(),
                    sentinel=sentinel.tobytes().hex(),
                    samples=samples,
                )
            )
            graph.close()
            sequence.clear()
    assert not module._exported_kernels
    artifact = tmp_path / "indexed-copy.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        assert not any(
            n.endswith((".bc", ".spv", ".ptx", ".cubin")) for n in archive.namelist()
        )
    artifact.with_suffix(".json").write_text(json.dumps(cases))
    arch = ti.lang.impl.current_cfg().arch
    backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_indexed_copy_probe.py")),
                mode,
                backend,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
        for marker in ("Validation Error", "SYNC-HAZARD"):
            assert marker not in result.stdout + result.stderr, (
                result.stdout + result.stderr
            )
