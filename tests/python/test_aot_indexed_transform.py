"""Preserve the selected fused primitive strategy across AOT export."""

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import taichi_forge as ti
from tests import test_utils


def sequence_for(dtype):
    arrays = dict(
        src=ti.ndarray(dtype, 513), transformed=ti.ndarray(dtype, 513),
        indices=ti.ndarray(ti.i32, 7), gathered=ti.ndarray(dtype, 7),
        dst=ti.ndarray(dtype, 517),
    )
    sequence = ti.algorithms.primitive_sequence()
    sequence.transform(arrays["src"], arrays["transformed"], scale=3, bias=5)
    sequence.gather(arrays["transformed"], arrays["indices"], arrays["gathered"])
    sequence.scatter(arrays["gathered"], arrays["indices"], arrays["dst"])
    return sequence, arrays


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_fused_sequence_cold_warm_runtime_only_and_borrowed(tmp_path, monkeypatch):
    from taichi_forge.algorithms import _algorithms as algorithms
    from taichi_forge.algorithms._primitive_definitions import IndexedTransformDefinition
    from taichi_forge.graph._native_definition import NativeOperationBinding

    bound = []
    original_bind = IndexedTransformDefinition.bind

    def observe(definition, *arrays):
        binding = original_bind(definition, *arrays)
        assert isinstance(binding, NativeOperationBinding)
        bound.append(definition)
        return binding

    monkeypatch.setattr(IndexedTransformDefinition, "bind", observe)

    def no_rebuild(*args, **kwargs):
        raise AssertionError("Export/replay must not construct a replacement fused native plan")

    module = ti.aot.Module()
    cases = []
    for dtype in (ti.i32, ti.u32, ti.f32):
        for warm in (False, True):
            monkeypatch.setenv("TAICHI_FORGE_PRIMITIVE_SEQUENCE_FUSION", "1")
            sequence, arrays = sequence_for(dtype)
            for array in arrays.values():
                array.fill(2)
            if warm:
                sequence.run()
                assert sequence.fused_plan_count == 1
                # A preference change does not replace an already selected plan.
                monkeypatch.setenv("TAICHI_FORGE_PRIMITIVE_SEQUENCE_FUSION", "0")
            before = {name: value.to_numpy().tobytes() for name, value in arrays.items()}
            builder = ti.graph.GraphBuilder()
            builder.append_native(sequence)
            graph = builder.compile()
            name = f"{dtype}_{warm}"
            bound.clear()
            with monkeypatch.context() as patch:
                patch.setattr(algorithms._NativePrimitivePlan, "__init__", no_rebuild)
                plan = module.add_execution_plan(name, graph, source_bindings=arrays)
            assert len(bound) == 1
            if warm:
                assert bound[0] is sequence._fused_plan.definition
            assert before == {name: value.to_numpy().tobytes() for name, value in arrays.items()}
            assert sequence.fused_plan_count == int(warm)
            operation, = json.loads(plan.to_json())["operations"]
            assert operation["operation"] == "indexed_transform"
            assert len(operation["bindings"]) == 5
            if warm:
                with monkeypatch.context() as patch:
                    patch.setattr(algorithms, "_resolve_vulkan_indexed_transform_definition", no_rebuild)
                    patch.setattr(IndexedTransformDefinition, "bind", no_rebuild)
                    sequence.run()
            cases.append(dict(name=name, dtype=str(dtype), offset=False))
            if not warm:
                operation["bindings"] = {port: f"view_{port}" for port in arrays}
                module.add_execution_plan(
                    f"{dtype}_view", ti.aot.ExecutionPlan([operation], resources=[
                        dict(kind="ndarray_view", name=f"view_{port}", array=port,
                             offset=1, shape=list(array.shape))
                        for port, array in arrays.items()
                    ]),
                )
                cases.append(dict(name=f"{dtype}_view", dtype=str(dtype), offset=True))
            graph.close()
            sequence.clear()
    assert not module._exported_kernels
    artifact = tmp_path / "indexed-transform.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        assert not any(name.endswith((".bc", ".spv", ".ptx", ".cubin")) for name in archive.namelist())
    artifact.with_suffix(".json").write_text(json.dumps(cases))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, *(["-S"] if mode == "standalone" else []),
             str(Path(__file__).with_name("aot_indexed_transform_probe.py")), mode, str(artifact)],
            env=os.environ.copy(), capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
        assert "SYNC-HAZARD" not in result.stdout + result.stderr


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_unfused_sequence_preserves_native_operations(monkeypatch):
    monkeypatch.setenv("TAICHI_FORGE_PRIMITIVE_SEQUENCE_FUSION", "0")
    sequence, arrays = sequence_for(ti.i32)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    plan = ti.aot.Module().add_execution_plan("ordered", graph, source_bindings=arrays)
    assert [op["operation"] for op in json.loads(plan.to_json())["operations"]] == [
        "transform", "indexed_copy", "indexed_copy"
    ]
    graph.close()
    sequence.clear()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_fused_export_rejects_changed_selected_strategy():
    sequence, arrays = sequence_for(ti.i32)
    for array in arrays.values():
        array.fill(2)
    sequence.run()
    sequence._fused_plan.method_name = "foreign_strategy"
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    with pytest.raises(ti.TaichiRuntimeError, match="selected fused strategy"):
        ti.aot.Module().add_execution_plan("step", graph, source_bindings=arrays)
    graph.close()
    sequence.clear()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_fused_sequence_downstream_kernel_gradient(tmp_path):
    @ti.kernel
    def seed(dx: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1),
             dl: ti.types.ndarray(ti.f32, 1)):
        for i in dx:
            dx[i] = 0
        loss[0], dl[0] = 0, 1

    @ti.kernel
    def energy(transformed: ti.types.ndarray(ti.f32, 1), gathered: ti.types.ndarray(ti.f32, 1),
               dst: ti.types.ndarray(ti.f32, 1), x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            loss[0] += (transformed[i] + gathered[i] + dst[i]) * x[i] * x[i]

    sequence, arrays = sequence_for(ti.f32)
    symbols = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
               for name in ("transformed", "gathered", "dst", "x", "dx", "loss", "dl")}
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    builder.dispatch(seed, symbols["dx"], symbols["loss"], symbols["dl"])
    arguments = tuple(symbols[name] for name in ("transformed", "gathered", "dst", "x", "loss"))
    builder.dispatch(energy, *arguments)
    builder.dispatch(energy.grad, *arguments)
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("gradient", graph, source_bindings=arrays, gradients=dict(x="dx", loss="dl"))
    artifact = tmp_path / "fused-gradient.tcm"
    module.archive(str(artifact))
    graph.close()
    sequence.clear()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, *(["-S"] if mode == "standalone" else []),
             str(Path(__file__).with_name("aot_indexed_transform_probe.py")), mode, str(artifact), "gradient"],
            env=os.environ.copy(), capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
        assert "SYNC-HAZARD" not in result.stdout + result.stderr
