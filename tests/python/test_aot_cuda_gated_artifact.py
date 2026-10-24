"""Persist generated CUDA variants without allowing ordinary launch frames."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("artifact_format", ("bc", "ptx", "cubin"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_saved_cuda_gate_code_and_launch_contract(artifact_format, tmp_path):
    executable = Path(
        os.environ.get(
            "TI_TEST_CPP_EXECUTABLE",
            str(
                Path(__file__).resolve().parents[2]
                / "build"
                / ("taichi_cpp_tests.exe" if os.name == "nt" else "taichi_cpp_tests")
            ),
        )
    )
    if not executable.is_file():
        pytest.skip("Requires the native AOT fixture executable")
    if artifact_format == "cubin" and not (os.environ.get("TI_CUDA_PTXAS_PATH") or shutil.which("ptxas")):
        pytest.skip("Requires an external ptxas for cubin export")

    @ti.kernel
    def forward(
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in x:
            y[i] = x[i] * x[i]

    @ti.kernel
    def control(state: ti.types.ndarray(ti.i32, 1), dy: ti.types.ndarray(ti.f32, 1), operation: ti.i32):
        if operation == 0:
            state[0] = 0
            state[1] = state[4] > 0
            state[9] += 1
        elif operation == 1:
            state[2] = 0
            state[3] = state[5] > 0
        elif operation == 2:
            for i in ti.static(range(4)):
                dy[i] = 1.0
            state[2] += 1
            state[6] += 1
            state[3] = state[2] < state[5]
        elif operation == 3:
            state[0] += 1
            state[1] = state[0] < state[4]
        elif operation == 4:
            state[7] = 77
        elif operation == 5:
            state[8] = state[6]
        elif operation == 6:
            state[10] = 0
            state[11] = state[12] > 0
        elif operation == 7:
            for i in ti.static(range(4)):
                dy[i] = 1.0
            state[10] += 1
            state[6] += 1
            state[11] = state[10] < state[12]
        elif operation == 8:
            state[0] = 0
            state[1] = state[4] <= 0
            state[9] += 1
        elif operation == 9:
            state[0] += 1
            state[1] = state[0] >= state[4]
        elif operation == 10:
            state[1] = state[4]
            state[9] += 1
        elif operation >= 100:
            for i in ti.static(range(4)):
                dy[i] = 1.0
            state[6] += operation - 99
            state[13] += 1
            state[1] = state[5]  # Selection must remain latched for this branch.

    caps = (
        [ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())]
        if artifact_format == "cubin"
        else None
    )
    module = ti.aot.Module(artifact_format=artifact_format, caps=caps)
    begin = time.perf_counter()
    module.add_kernel(forward, name="forward")
    module.add_kernel(forward.grad, name="reverse")
    module.add_kernel(control, name="control")
    costs = dict(ordinary_compile_ms=(time.perf_counter() - begin) * 1000)
    begin = time.perf_counter()
    # Internal selected-code producer entry. Graph lowering will call this only
    # for variants required by a frozen masked recipe, in its own change.
    for name in ("forward", "reverse", "control"):
        compiled = module._exported_kernels[name]
        module._aot_builder._add_with_launch_abi("gated_" + name, compiled, "cuda_graph_gate_v1")
        module._content.append("kernel:gated_" + name)
    costs["variant_generation_ms"] = (time.perf_counter() - begin) * 1000
    directory = tmp_path / "module"
    begin = time.perf_counter()
    module.save(directory)
    costs["save_ms"] = (time.perf_counter() - begin) * 1000
    costs["artifact_bytes"] = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    metadata = json.loads((directory / "aot_metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 10
    assert metadata["kernel_format"] == artifact_format
    assert {item["key"]: item["value"] for item in metadata["kernel_launch_abis"]} == {
        "forward": "runtime_context_v2",
        "reverse": "runtime_context_v2",
        "control": "runtime_context_v2",
        "gated_forward": "cuda_graph_gate_context_v2",
        "gated_reverse": "cuda_graph_gate_context_v2",
        "gated_control": "cuda_graph_gate_context_v2",
    }
    assert len(metadata["launch_abi_sha256"]) == 64
    assert (directory / f"gated_forward.{artifact_format}").is_file()
    assert (directory / f"gated_reverse.{artifact_format}").is_file()
    # Native process loads actual generated code and captures forward+grad,
    # checks closed/open/closed gates, rebinding and ordinary-code isolation.
    begin = time.perf_counter()
    result = subprocess.run(
        [str(executable), "--gtest_filter=CudaGatedAot.*"],
        env=dict(os.environ, TAICHI_AOT_FOLDER_PATH=str(directory)),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0 and "[  PASSED  ] 2 tests." in result.stdout, result.stdout + result.stderr
    costs["native_consumer_process_ms"] = (time.perf_counter() - begin) * 1000
    costs["native_graph"] = json.loads(
        next(line.removeprefix("GATE_COSTS ") for line in result.stdout.splitlines() if line.startswith("GATE_COSTS "))
    )
    costs["fixed_graph"] = json.loads(
        next(
            line.removeprefix("FIXED_COSTS ") for line in result.stdout.splitlines() if line.startswith("FIXED_COSTS ")
        )
    )
    costs["masked_branch"] = json.loads(
        next(
            line.removeprefix("BRANCH_COSTS ")
            for line in result.stdout.splitlines()
            if line.startswith("BRANCH_COSTS ")
        )
    )
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cuda_gated_artifact_probe.py")),
                mode,
                str(directory),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        costs[mode] = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"load_ms"')))
    (tmp_path / "costs.json").write_text(json.dumps(costs, indent=2), encoding="utf-8")


@test_utils.test(arch=[ti.cpu, ti.vulkan], offline_cache=False)
def test_gated_export_rejects_other_backends_without_publishing(tmp_path):
    @ti.kernel
    def noop():
        pass

    module = ti.aot.Module()
    module.add_kernel(noop)
    with pytest.raises(RuntimeError, match="(Unsupported|unsupported).*launch ABI|launch ABI.*unsupported"):
        module._aot_builder._add_with_launch_abi("bad", module._exported_kernels["noop"], "cuda_graph_gate_v1")
    module.save(tmp_path)
    assert not any(p.name.startswith("bad.") for p in tmp_path.iterdir())
