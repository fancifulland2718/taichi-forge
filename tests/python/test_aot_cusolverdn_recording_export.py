"""Cholesky export freezes real factors and preserves public device status."""

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from tests import test_utils


@pytest.mark.parametrize("dtype,count,alias", [(ti.f32, 1, False), (ti.f64, 3, True)])
@pytest.mark.parametrize("phase", ("solve", "factor_and_solve"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusolverdn_recording_export(dtype, count, alias, phase, tmp_path):
    vendor = os.environ.get("TI_FORGE_TEST_CUSOLVERDN_LIBRARY_PATH")
    adapter = os.environ.get("TI_CUSOLVERDN_TEST_ADAPTER")
    if not vendor or not adapter:
        pytest.skip("real cuSOLVER adapter/vendor required")
    n = 17
    shape = (n,) if count == 1 else (count, n)
    dimensions = len(shape)
    a = ti.ndarray(dtype, (n, n))
    arrays = {key: ti.ndarray(dtype, shape) for key in ("source", "rhs", "result")}
    arrays["solution"] = arrays["rhs"] if alias else ti.ndarray(dtype, shape)
    arrays.update({key: ti.ndarray(ti.i32, (2,)) for key in ("before", "after")})
    arrays["a"] = a
    dense = np.diag(np.full(n, 4.0)) + np.diag(np.full(n - 1, -1.0), 1) + np.diag(np.full(n - 1, -1.0), -1)
    a.from_numpy(dense.astype(np.float32 if dtype == ti.f32 else np.float64))
    provider = ti.hardware.linalg.CusolverDnProvider(vendor)
    plan = provider.cholesky_plan(n, rhs_count=count, dtype=dtype)
    binding = plan.bind(a, arrays["rhs"], arrays["solution"])
    if phase == "solve":
        binding.factor()
        a.fill(0)  # This is no longer a valid factor seed.
    captured = binding.capture(mode=phase)
    arrays["status"] = plan.info
    initial_status = plan.info.to_numpy()

    @ti.kernel
    def produce(source: ti.types.ndarray(dtype, dimensions), rhs: ti.types.ndarray(dtype, dimensions)):
        for index in ti.grouped(source):
            rhs[index] = 2 * source[index]

    @ti.kernel
    def copy_status(status: ti.types.ndarray(ti.i32, 1), destination: ti.types.ndarray(ti.i32, 1)):
        for i in range(2):
            destination[i] = status[i]

    @ti.kernel
    def publish(solution: ti.types.ndarray(dtype, dimensions), result: ti.types.ndarray(dtype, dimensions)):
        for index in ti.grouped(solution):
            result[index] = 2 * solution[index]

    symbols = {
        key: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, key, array.dtype, ndim=len(array.shape))
        for key, array in arrays.items()
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(copy_status, symbols["status"], symbols["before"])
    builder.dispatch(produce, symbols["source"], symbols["rhs"])
    builder.append_native(captured.record())
    builder.dispatch(copy_status, symbols["status"], symbols["after"])
    builder.dispatch(publish, symbols["solution"], symbols["result"])
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="snapshot_state"):
        module.add_execution_plan("missing_snapshot", graph, source_bindings={"status": plan.info})

    def export(name, executable):
        result = module.add_execution_plan(name, executable, snapshot_state=True, source_bindings={"status": plan.info})
        native = [item for item in json.loads(result.to_json())["resources"] if item.get("kind") == "native"]
        assert len(native) == 1
        assert native[0]["parameters"]["seeded"] == (phase == "solve")
        np.testing.assert_array_equal(plan.info.to_numpy(), initial_status)

    export("cold", graph)
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(),))
    catalog.build_compatible_stage(candidate_limit=4)
    assert len(catalog.entries()) == 1
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(catalog.entries()[0].recipe) as selected:
            export("selected", selected.executor)
    minimal_builder = ti.graph.GraphBuilder()
    minimal_builder.append_native(captured.record())
    minimal = minimal_builder.compile()
    module.add_execution_plan("minimal", minimal, snapshot_state=phase == "solve")
    minimal.close()
    # No unused old factor is serialized for a graph that refactors first.
    assert len(module._data_payloads) == (2 if phase == "solve" else 1)
    artifact = tmp_path / "cusolverdn-recording.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            assert vendor.encode() not in data and adapter.encode() not in data
    graph.close()
    captured.close()
    plan.close()
    provider.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_cusolverdn_recording_probe.py")),
                mode,
                str(artifact),
                adapter,
                vendor,
                str(dtype),
                phase,
                str(count),
                str(alias),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
