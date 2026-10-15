"""Fused maps and opaque library commands retain their physical ordering."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_mixed_fusion_preserves_native_barriers(tmp_path):
    directory = os.environ.get("TI_TEST_CUFFT_DIRECTORY")
    dll = os.add_dll_directory(directory) if directory and os.name == "nt" else None
    if not ti.hardware.fft.is_available():
        pytest.skip("external cuFFT required")
    layout = ti.hardware.fft.CufftLayout(embed=(2, 4))
    fft = ti.hardware.fft.CufftPlanND((2, 4), input_layout=layout, output_layout=layout)

    @ti.kernel
    def scale(x: ti.types.ndarray(ti.f32, ndim=1), y: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(16):
            y[i] = 2 * x[i]

    @ti.kernel
    def shift(x: ti.types.ndarray(ti.f32, ndim=1), y: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(16):
            y[i] = x[i] + 0.25

    @ti.kernel
    def halve(x: ti.types.ndarray(ti.f32, ndim=1), y: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(16):
            y[i] = x[i] * 0.5

    @ti.kernel
    def normalize(x: ti.types.ndarray(ti.f32, ndim=1), y: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(16):
            y[i] = x[i] / 8

    names = ("x", "prefix", "source", "spectrum", "middle", "filtered", "inverted", "suffix", "y")
    args = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in names}
    builder = ti.graph.GraphBuilder(_map_recipe="map2")
    # Sequential source maps must also export their selected physical kernel.
    prefix = builder.create_sequential()
    prefix.dispatch(scale, args["x"], args["prefix"])
    prefix.dispatch(shift, args["prefix"], args["source"])
    builder.append(prefix)
    builder.append_native(fft.record(input="source", output="spectrum"))
    builder.dispatch(halve, args["spectrum"], args["middle"])
    builder.dispatch(shift, args["middle"], args["filtered"])
    builder.append_native(fft.record(input="filtered", output="inverted", direction="inverse"))
    builder.dispatch(normalize, args["inverted"], args["suffix"])
    builder.dispatch(scale, args["suffix"], args["y"])
    graph = builder.compile()
    assert graph.physical_plan()["physical_dispatch_count"] == 5
    module = ti.aot.Module()
    for name in ("step", "again"):
        description = json.loads(module.add_execution_plan(name, graph).to_json())
        assert [op["kind"] for op in description["operations"]] == ["kernel", "native", "kernel", "native", "kernel"]
        assert len(description["resources"]) == 1
        assert len(module._exported_kernels) == 3
    graph.close()
    del graph, builder, prefix
    gc.collect()
    fft.close()
    artifact = tmp_path / "mixed-physical.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_mixed_physical_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
    if dll is not None:
        dll.close()
