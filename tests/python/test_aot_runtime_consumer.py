import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from taichi_forge._lib import utils as runtime_utils
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_lightweight_array_allocation_usages():
    from tests.python.aot_optix_probe import read, write

    with ti.aot.borrow_runtime() as runtime:
        value = runtime.ndarray("f32", 3, usage=("storage", "vertex", "uniform", "index", "indirect"))
        write(value, [1, 2, 3])
        assert read(value) == [1, 2, 3]
        value.close()
        for invalid in ((), "vertex", ("unknown",), (1,)):
            with pytest.raises(ValueError, match="usage"):
                runtime.ndarray("f32", 1, usage=invalid)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_lightweight_aot_consumer_and_jit_coexistence(tmp_path):
    # The ordinary monolithic build does not contain the shared-runtime C API.
    native = runtime_utils._loaded_native_runtime_path
    if native is None or not hasattr(runtime_utils._native_library_handles[-1], "ti_copy_memory_from_host_ext"):
        pytest.skip("Requires a split runtime built with TI_WITH_RUNTIME_C_API=ON")
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"

    @ti.kernel
    def forward(
        x: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
    ):
        for i in x:
            y[i] = x[i] * x[i] + 2 * x[i]

    @ti.kernel
    def adjust(scale: ti.f32, offset: ti.i32, out: ti.types.ndarray(dtype=ti.math.vec2, ndim=1)):
        for i in out:
            out[i] = ti.Vector([scale * i + offset, scale * i + offset - 1])

    @ti.kernel
    def scalar_widths(a: ti.i8, b: ti.u64, c: ti.f16, d: ti.f64, out: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        out[0], out[1], out[2], out[3] = a, b, c, d

    module = ti.aot.Module()
    module.add_kernel(forward)
    module.add_kernel(forward.grad, name="backward")
    module.add_kernel(adjust)
    if arch == "cpu":
        module.add_kernel(scalar_widths)
        scalar_graph = ti.graph.GraphBuilder()
        scalar_graph.dispatch(
            scalar_widths,
            ti.graph.Arg(ti.graph.ArgKind.SCALAR, "a", ti.i8),
            ti.graph.Arg(ti.graph.ArgKind.SCALAR, "b", ti.u64),
            ti.graph.Arg(ti.graph.ArgKind.SCALAR, "c", ti.f16),
            ti.graph.Arg(ti.graph.ArgKind.SCALAR, "d", ti.f64),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f64, ndim=1),
        )
        module.add_graph("scalar_graph", scalar_graph.compile())
    builder = ti.graph.GraphBuilder()
    x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1)
    y = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", ti.f32, ndim=1)
    builder.dispatch(forward, x, y)
    builder.dispatch(forward.grad, x, y)
    module.add_graph("differentiate", builder.compile())
    module.add_execution_plan(
        "snapshot",
        ti.aot.ExecutionPlan(
            [
                {"kind": "fill32", "destination": "dx", "value": 0},
                {"kind": "fill32", "destination": "dy", "value": 1065353216},
                {"kind": "kernel", "entry": "forward", "arguments": ["x", "y"]},
                {"kind": "copy", "source": "y", "destination": "snapshot"},
                {"kind": "kernel", "entry": "backward", "arguments": ["x", "y"]},
            ]
        ),
    )
    module.save(tmp_path)
    original = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    module.archive(str(tmp_path / "packed.tcm"))
    omitted = {"metadata.json", "metadata.lock"} if arch != "vulkan" else set()
    with ZipFile(tmp_path / "packed.tcm") as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == {
            name: data for name, data in original.items() if name not in omitted
        }
    module.archive(str(tmp_path / "debug.tcm"), include_debug_metadata=True)
    with ZipFile(tmp_path / "debug.tcm") as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == original

    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join([str(root / "packaging/runtime/src"), *sys.path]),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(native).parent),
        TAICHI_RUNTIME_DIR=runtime_utils._runtime_bitcode_dir(),
        TI_ENABLE_VERSION_CHECK="0",
    )
    for order, source in (
        ("pure", tmp_path / "packed.tcm"),
        ("aot-first", tmp_path),
        ("jit-first", tmp_path),
    ):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("aot_consumer_probe.py")), arch, str(source), order],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
