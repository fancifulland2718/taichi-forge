import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge._lib import utils as runtime_utils
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_source_free_aot_compact_and_explicit_gradient(tmp_path):
    native = runtime_utils._loaded_native_runtime_path
    if native is None or not hasattr(runtime_utils._native_library_handles[-1], "ti_prepare_execution_plan_ext"):
        pytest.skip("Requires the shared runtime C API")
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"

    @ti.kernel
    def indices(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in out:
            out[i] = i

    @ti.kernel
    def evaluate(
        x: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
        selected: ti.types.ndarray(dtype=ti.i32, ndim=1),
        count: ti.types.ndarray(dtype=ti.i32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
    ):
        for j in range(count[0]):
            y[j] = x[selected[j]] * x[selected[j]]

    module = ti.aot.Module()
    module.add_kernel(indices)
    module.add_kernel(evaluate)
    module.add_kernel(evaluate.grad, name="backward")
    module.add_execution_plan(
        "selected_energy",
        ti.aot.ExecutionPlan(
            [
                {"kind": "fill32", "destination": "dx", "value": 0},
                {"kind": "fill32", "destination": "dy", "value": 1065353216},
                {"kind": "fill32", "destination": "y", "value": 0},
                {"kind": "kernel", "entry": "indices", "arguments": ["indices"]},
                {"kind": "compact", "values": "indices", "flags": "flags", "output": "selected", "count": "count"},
                {"kind": "kernel", "entry": "evaluate", "arguments": ["x", "selected", "count", "y"]},
                {"kind": "kernel", "entry": "backward", "arguments": ["x", "selected", "count", "y"]},
            ]
        ),
    )
    module.archive(str(tmp_path / "compact.tcm"))
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join([str(root / "packaging/runtime/src"), *sys.path]),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(native).parent),
        TAICHI_RUNTIME_DIR=runtime_utils._runtime_bitcode_dir(),
        TI_ENABLE_VERSION_CHECK="0",
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("aot_compact_probe.py")), arch, str(tmp_path / "compact.tcm")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
