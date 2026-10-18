import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge._lib import utils as runtime_utils
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_source_free_execution_resources_and_gradients(tmp_path):
    native = runtime_utils._loaded_native_runtime_path
    if native is None or not hasattr(
        runtime_utils._native_library_handles[-1], "ti_get_prepared_execution_resource_bytes_ext"
    ):
        pytest.skip("Requires execution resource reconstruction in the shared runtime")
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"

    @ti.kernel
    def square(
        x: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
        n: ti.i32,
    ):
        for i in range(n):
            y[i] = x[i] * x[i]

    allocator = {"allocator": "pool"} if arch == "cuda" else {}
    plan = ti.aot.ExecutionPlan(
        [
            {"kind": "fill32", "destination": "scratch_grad", "value": 0},
            {"kind": "copy", "source": "x", "destination": "scratch"},
            {"kind": "kernel", "entry": "square", "arguments": ["scratch", "y", "n"]},
            {"kind": "kernel", "entry": "backward", "arguments": ["scratch", "y", "n"]},
            {"kind": "copy", "source": "scratch_grad", "destination": "dx"},
        ],
        resources=[
            {
                "kind": "ndarray",
                "name": "scratch",
                "dtype": "f32",
                "shape": [{"array": "shape_only", "axis": 0}],
                "usage": ["storage", "vertex", "uniform", "index"],
                **allocator,
            },
            {"kind": "ndarray", "name": "shape_only", "dtype": "f32", "shape": [{"array": "x", "axis": 0}, 3],
             "allocator": "unbound_shape_allocator"},
            {"kind": "extent", "name": "n", "dtype": "i32", "array": "scratch", "axis": 0},
            {"kind": "ndarray", "name": "scratch_grad", "dtype": "f32", "shape": [{"array": "scratch", "axis": 0}], **allocator},
            {"kind": "ndarray", "name": "unused", "dtype": "f64", "shape": [2147483647, 2147483647]},
            {"kind": "native", "name": "pool", "provider": "memory", "operation": "pool",
             "version": 1, "parameters": {}, "bindings": {}},
        ],
        gradients={"scratch": "scratch_grad", "y": "dy"},
    )
    assert plan.argument_names == ("dx", "dy", "x", "y")
    module = ti.aot.Module()
    module.add_kernel(square)
    module.add_kernel(square.grad, name="backward")
    module.add_execution_plan("square_with_resources", plan)
    artifact = tmp_path / "resources.tcm"
    module.archive(str(artifact))
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(root / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(native).parent),
        TAICHI_RUNTIME_DIR=runtime_utils._runtime_bitcode_dir(),
        TI_ENABLE_VERSION_CHECK="0",
    )
    result = subprocess.run(
        [sys.executable, "-S", str(Path(__file__).with_name("aot_resources_probe.py")), arch, str(artifact)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Unable to release execution resource" not in result.stdout + result.stderr
