"""Pinned operator generations retain public ports and a portable data closure."""

import gc
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
from tests.python.test_linear_operator_graph_action import _rectangular_operator


def operator_graph(operator, policy="pinned", private=False):
    args = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y", "dx")}
    builder = ti.graph.GraphBuilder()
    if private:
        args["y"] = builder.private_ndarray("y", ti.f32, (2,))
    builder.append_native(operator.graph_action(args["x"], args["y"], update_policy=policy))
    builder.append_native(operator.graph_action(args["y"], args["dx"], adjoint=True, update_policy=policy))
    return builder.compile()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_pinned_operator_generation_and_adjoint(tmp_path, monkeypatch):
    matrix = np.array([[2, -1, 0.5], [3, 4, -2]], dtype=np.float32)
    operator = _rectangular_operator(matrix)
    pinned = operator_graph(operator)
    private = operator_graph(operator, private=True)
    dynamic = operator_graph(operator, "rebind")
    arrays = {name: ti.ndarray(ti.f32, extent) for name, extent in (("x", 3), ("y", 2), ("dx", 3))}
    x = np.array([0.25, -0.5, 2], dtype=np.float32)
    arrays["x"].from_numpy(x)
    frame = pinned.bind(arrays)
    updated = ti.ndarray(ti.f32, 6)
    updated.from_numpy((matrix * 2).reshape(-1))
    operator.update_numeric(updated, expected_topology_version=1, expected_numeric_version=1)
    updated.fill(0)  # Publication owns an immutable copy.
    for graph, args, scale in ((pinned, frame, 1), (dynamic, arrays, 2)):
        graph.run(args)
        np.testing.assert_allclose(arrays["y"].to_numpy(), scale * matrix @ x)
        np.testing.assert_allclose(arrays["dx"].to_numpy(), scale * scale * matrix.T @ matrix @ x)
    with pytest.raises(ti.TaichiRuntimeError, match="exactly"):
        pinned.bind(dict(arrays, dx=arrays["y"]))
    with pytest.raises(ti.TaichiRuntimeError, match="exactly"):
        pinned.bind(dict(arrays, x=ti.ndarray(ti.f32, 4)))

    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="fixed bindings"):
        module.add_execution_plan("dynamic", dynamic)
    assert not module._data_payloads and not module._exported_kernels

    invalid = ti.graph.GraphBuilder()
    symbols = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y")]
    invalid.append_native(operator.graph_action(*symbols, update_policy="pinned"))
    invalid.observe(ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0))
    invalid_graph = invalid.compile()
    with pytest.raises(ti.TaichiRuntimeError, match="Observation"):
        module.add_execution_plan("invalid", invalid_graph)
    assert not module._data_payloads and not module._exported_kernels
    invalid_graph.close()

    # Export never asks the operator to select its newer generation.
    from taichi_forge.linalg._runtime import _LinearOperatorGraphExecutable

    def forbidden(*args, **kwargs):
        raise AssertionError("Export tried to bind a live provider generation")

    before = ti.runtime.stats().transfer.device_to_host_bytes
    with monkeypatch.context() as check:
        check.setattr(_LinearOperatorGraphExecutable, "_current_compatible_record", forbidden)
        description = module.add_execution_plan("step", pinned)
    assert ti.runtime.stats().transfer.device_to_host_bytes - before == 48
    document = json.loads(description.to_json())
    assert set(description.argument_names) == {"x", "y", "dx"}
    assert len(module._exported_kernels) == 2
    data_resources = [resource for resource in document["resources"] if "initial_data" in resource]
    assert len(data_resources) == 2
    assert len(document["constraints"]) == 5, "Repeated port certificates should be stored only once"
    assert len(module._data_payloads) == 2
    module.add_execution_plan("again", pinned)
    assert len(module._exported_kernels) == 2 and len(module._data_payloads) == 2
    private_plan = module.add_execution_plan("private", private)
    assert set(private_plan.argument_names) == {"x", "dx"}
    private.run({name: arrays[name] for name in ("x", "dx")})
    np.testing.assert_allclose(arrays["dx"].to_numpy(), matrix.T @ matrix @ x)

    refreshed = operator_graph(operator)
    refreshed.run(arrays)
    np.testing.assert_allclose(arrays["y"].to_numpy(), 2 * matrix @ x)
    refreshed.close()
    pinned.close()
    dynamic.close()
    private.close()
    del frame, pinned, dynamic, refreshed, private, operator
    gc.collect()
    artifact = tmp_path / "operator.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        assert sum(len(archive.read(name)) for name in archive.namelist() if name.startswith("data/")) == 48
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_pinned_operator_probe.py")),
                mode,
                "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_pinned_operator_requires_an_explicit_export_contract():
    from taichi_forge.graph._native import NativeGraphNode
    from taichi_forge.linalg._pinned_graph_operator import PinnedOperatorGraphExecutable

    operator = _rectangular_operator(np.arange(6, dtype=np.float32))
    x, y = (ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y"))
    with pytest.raises(ValueError, match="update_policy"):
        operator.graph_action(x, y, update_policy="snapshot")

    class AdditionalValidation(PinnedOperatorGraphExecutable):
        def validate_graph_bindings(self, arguments):
            super().validate_graph_bindings(arguments)
            raise RuntimeError("An unqualified subclass cannot inherit the base AOT contract")

    class Node(NativeGraphNode):
        def compile(self):
            return AdditionalValidation(operator, x, y, False)

    builder = ti.graph.GraphBuilder()
    builder.append_native(Node())
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="fixed bindings"):
        module.add_execution_plan("unqualified", graph)
    assert not module._data_payloads and not module._exported_kernels
    graph.close()
