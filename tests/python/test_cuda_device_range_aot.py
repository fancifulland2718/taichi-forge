"""LLVM AOT entry policies share code without sharing mutable recording state."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang._offload_execution_plan import _bind_offload_execution_plan
from tests import test_utils
from tests.python.test_device_range_plan import graph, select


def consume(artifact, mode="range"):
    result = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).with_name("aot_llvm_range_probe.py")),
         str(artifact), mode], capture_output=True, text=True, timeout=90,
    )
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("artifact_format", ["bc", "ptx"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_aot_range_entry_shares_payload_and_survives_export(tmp_path, artifact_format):
    @ti.kernel
    def sequence(out: ti.types.ndarray(ti.i32, 1), limits: ti.types.ndarray(ti.i32, 1)):
        out[65] = 0
        ti.loop_config(block_dim=32)
        for i in range(limits[0]):
            out[i] = i + 1
        limits[0] = limits[0] // 2
        ti.loop_config(block_dim=32)
        for i in range(limits[0]):
            out[i] += 100
        out[65] = 17

    out, limits = ti.ndarray(ti.i32, 66), ti.ndarray(ti.i32, 1)
    baseline, selected = select(sequence, out, limits)
    symbols = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
               for name in ("out", "limits")]
    fixed, device = graph(sequence, baseline, symbols), graph(sequence, selected, symbols)
    module = ti.aot.Module(artifact_format=artifact_format)
    entries = {}
    # Exercise a selected canonical payload too; an ordinary alias must never
    # inherit its controller selection or public entry annotations.
    order = ("fixed", "device") if artifact_format == "bc" else ("device", "fixed")
    for name in order:
        plan = module.add_execution_plan(name, fixed if name == "fixed" else device)
        entries[name] = json.loads(plan.to_json())["operations"][0]["entry"]
    module.add_execution_plan("again", device)
    directory = tmp_path / "module"
    module.save(directory)
    metadata = json.loads((directory / "aot_metadata.json").read_text())
    assert metadata["schema_version"] == 20
    assert len(list(directory.glob("*." + artifact_format))) == 1
    assert len(metadata["kernel_aliases"]) == 1
    assert metadata["device_range_dispatch"] == [
        dict(name=entries["device"], version=1, tasks=list(selected.device_range_tasks))]
    executable = Path(os.environ.get("TI_TEST_CPP_EXECUTABLE", "build/taichi_cpp_tests.exe"))
    if executable.is_file():
        result = subprocess.run(
            [str(executable.resolve()), "--gtest_filter=CudaRangeAotEntry.*"],
            env={**os.environ, "TAICHI_CUDA_RANGE_AOT_FOLDER": str(directory),
                 "TAICHI_RANGE_ORDINARY_ENTRY": entries["fixed"],
                 "TAICHI_RANGE_SELECTED_ENTRY": entries["device"]},
            capture_output=True, text=True, timeout=60,
        )
        print(result.stdout)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[  PASSED  ] 2 tests." in result.stdout
        assert "[  SKIPPED ]" not in result.stdout
    for name in ("one", "two"):
        artifact = tmp_path / (name + ".tcm")
        module.archive(str(artifact))
        with zipfile.ZipFile(artifact) as archive:
            payloads = [item for item in archive.infolist() if item.filename.endswith("." + artifact_format)]
            assert len(payloads) == 1
            print("SHARED-PAYLOAD", artifact_format, payloads[0].file_size)
        consume(artifact)
        out.fill(-1)
        limits.from_numpy(np.array([5], dtype=np.int32))
        device.run(dict(out=out, limits=limits))
        np.testing.assert_array_equal(out.to_numpy()[:6], [101, 102, 3, 4, 5, -1])
        limits.from_numpy(np.array([5], dtype=np.int32))
        sequence(out, limits)  # Original direct JIT code also remains usable.
    fixed.close()
    device.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_aot_range_primal_and_explicit_adjoint(tmp_path):
    @ti.kernel
    def square(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               limits: ti.types.ndarray(ti.i32, 1)):
        ti.loop_config(block_dim=32)
        for i in range(limits[0]):
            y[i] = x[i] * x[i]

    x, y = [ti.ndarray(ti.f32, 65, needs_grad=True) for _ in range(2)]
    limits = ti.ndarray(ti.i32, 1)
    symbols = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)
               for name, dtype in (("x", ti.f32), ("y", ti.f32), ("limits", ti.i32))]
    builder = ti.graph.GraphBuilder()
    for kernel in (square, square.grad):
        _, selection = select(kernel, x, y, limits)
        builder.dispatch(_bind_offload_execution_plan(kernel, selection), *symbols)
    source = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("gradient", source, gradients=dict(x="dx", y="dy"))
    artifact = tmp_path / "gradient.tcm"
    module.archive(str(artifact))
    consume(artifact, "gradient")
    x.fill(3)
    x.grad.fill(0)
    y.grad.fill(1)
    limits.from_numpy(np.array([65], dtype=np.int32))
    square(x, y, limits)
    square.grad(x, y, limits)
    np.testing.assert_array_equal(y.to_numpy(), np.full(65, 9, dtype=np.float32))
    np.testing.assert_array_equal(x.grad.to_numpy(), np.full(65, 6, dtype=np.float32))
    source.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_aot_aliases_share_code_and_keep_module_state(tmp_path):
    state = ti.field(ti.i32, shape=1)
    state[0] = 7

    @ti.kernel
    def increment(out: ti.types.ndarray(ti.i32, 1)):
        state[0] += 1
        out[0] = state[0]

    module = ti.aot.Module()
    for name in ("first", "second"):
        module.add_kernel(increment, name=name)
    dependencies = module._aot_builder._used_snode_tree_ids(["first", "second"])
    assert len(dependencies) == 1
    module.save(tmp_path)
    assert len(list(tmp_path.glob("*.bc"))) == 1
    assert json.loads((tmp_path / "aot_metadata.json").read_text())["schema_version"] == 20
    with ti.aot.borrow_runtime() as runtime:
        modules = [runtime.load_module(tmp_path) for _ in range(2)]
        output = runtime.ndarray("i32", (1,))
        for index, name, expected in ((0, "first", 1), (0, "second", 2), (1, "second", 1)):
            modules[index].kernel(name)(output)
            np.testing.assert_array_equal(np.frombuffer(output.read(), dtype=np.int32), [expected])
        modules[0].close()
        modules[1].kernel("first")(output)
        np.testing.assert_array_equal(np.frombuffer(output.read(), dtype=np.int32), [2])
        modules[1].close()
    out = ti.ndarray(ti.i32, 1)
    increment(out)
    np.testing.assert_array_equal(out.to_numpy(), [8])
