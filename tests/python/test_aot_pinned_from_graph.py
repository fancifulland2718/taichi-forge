"""Export multi-dispatch providers with explicit, instance-owned state snapshots."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_experimental_rectangular_graph_operator import _array, _compile_action_graph


def provider(*, legacy=False, stateful=True):
    matrix = np.array([[2, -1, 0.5], [3, 4, -2]], dtype=np.float32)
    if legacy:
        matrix = matrix[:, :2].copy()
    topology = _array(np.int32, matrix.shape)
    numeric = _array(np.float32, matrix.reshape(-1))

    @ti.kernel
    def plain_forward(
        t: ti.types.ndarray(ti.i32, 1),
        a: ti.types.ndarray(ti.f32, 1),
        x: ti.types.ndarray(ti.f32, 1),
        y: ti.types.ndarray(ti.f32, 1),
    ):
        for row in range(t[0]):
            value = 0.0
            for col in range(t[1]):
                value += a[row * t[1] + col] * x[col]
            y[row] = value

    @ti.kernel
    def plain_adjoint(
        t: ti.types.ndarray(ti.i32, 1),
        a: ti.types.ndarray(ti.f32, 1),
        x: ti.types.ndarray(ti.f32, 1),
        y: ti.types.ndarray(ti.f32, 1),
    ):
        for col in range(t[1]):
            value = 0.0
            for row in range(t[0]):
                value += a[row * t[1] + col] * x[row]
            y[col] = value

    @ti.kernel
    def finish(
        factor: ti.i32, u: ti.types.ndarray(ti.f32, 1), v: ti.types.ndarray(ti.f32, 1), y: ti.types.ndarray(ti.f32, 1)
    ):
        for i in range(y.shape[0]):
            y[i] *= factor * (u[0] + v[0])

    @ti.kernel
    def tick(u: ti.types.ndarray(ti.f32, 1), v: ti.types.ndarray(ti.f32, 1)):
        u[0] += 1
        v[0] += 2

    workspace = {"u": _array(np.float32, [2]), "v": _array(np.float32, [5])} if stateful else {}

    def graph(kernel):
        if not stateful:
            return _compile_action_graph(kernel)
        args = {
            name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)
            for name, dtype in (
                ("topology", ti.i32),
                ("numeric", ti.f32),
                ("input", ti.f32),
                ("output", ti.f32),
                ("u", ti.f32),
                ("v", ti.f32),
            )
        }
        builder = ti.graph.GraphBuilder()
        builder.dispatch(kernel, *(args[name] for name in ("topology", "numeric", "input", "output")))
        builder.dispatch(
            finish, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "factor", ti.i32), args["u"], args["v"], args["output"]
        )
        builder.dispatch(tick, args["u"], args["v"])
        return builder.compile()

    operator = ti.linalg.LinearOperator.from_graph(
        graph(plain_forward),
        2 if legacy else matrix.shape,
        adjoint=None if legacy else graph(plain_adjoint),
        fixed_i32={"factor": 2} if stateful else None,
        topology={"topology": topology},
        numeric={"numeric": numeric},
        workspace=workspace,
    )
    # Constructor snapshots, rather than borrowing the source arrays.
    numeric.fill(0)
    for array in workspace.values():
        array.fill(0)
    return operator, matrix


def outer(operator, *, adjoint=True):
    builder = ti.graph.GraphBuilder()
    args = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y", "dx")}
    builder.append_native(operator.graph_action(args["x"], args["y"], update_policy="pinned"))
    if adjoint:
        builder.append_native(operator.graph_action(args["y"], args["dx"], adjoint=True, update_policy="pinned"))
    return builder.compile()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_pinned_from_graph_state_and_generation_export(tmp_path, monkeypatch):
    operator, matrix = provider()
    legacy, legacy_matrix = provider(legacy=True)
    plain, _ = provider(stateful=False)
    expected_roles = {"topology": "topology", "numeric": "numeric", "u": "workspace", "v": "workspace"}
    for op in (operator, legacy):
        assert dict(op._handle._recordable_kernel(False)._fixed_ndarray_roles) == expected_roles
    graph, legacy_graph, plain_graph = outer(operator), outer(legacy, adjoint=False), outer(plain)
    composition = outer(plain.adjoint().compose(plain))
    arrays = {name: ti.ndarray(ti.f32, size) for name, size in (("x", 3), ("y", 2), ("dx", 3))}
    x = np.array([0.25, -0.5, 2], dtype=np.float32)
    arrays["x"].from_numpy(x)
    graph.run(arrays)
    np.testing.assert_allclose(arrays["y"].to_numpy(), 14 * matrix @ x)
    np.testing.assert_allclose(arrays["dx"].to_numpy(), 20 * matrix.T @ (14 * matrix @ x))
    operator.update_numeric(
        {"numeric": _array(np.float32, 2 * matrix.reshape(-1))}, expected_topology_version=1, expected_numeric_version=1
    )
    current = outer(operator)
    module = ti.aot.Module()
    before = ti.runtime.stats().transfer.device_to_host_bytes
    with pytest.raises(ti.TaichiRuntimeError, match="snapshot_state=True"):
        module.add_execution_plan("missing_choice", graph)
    assert not module._data_payloads and not module._exported_kernels
    assert ti.runtime.stats().transfer.device_to_host_bytes == before
    with pytest.raises(TypeError, match="bool"):
        module.add_execution_plan("invalid_choice", graph, snapshot_state=1)

    from taichi_forge.linalg._runtime import _LinearOperatorGraphExecutable

    def forbidden(*args, **kwargs):
        raise AssertionError("Export selected a live provider generation")

    program = ti.lang.impl.get_runtime().prog
    original_copy = type(program).copy_ndarrays_to_host
    copies = []

    def copy(program, sources, destinations, **options):
        assert options == {"snapshot": True}
        copies.append(len(sources))
        return original_copy(program, sources, destinations, **options)

    with monkeypatch.context() as guard:
        guard.setattr(_LinearOperatorGraphExecutable, "_current_compatible_record", forbidden)
        guard.setattr(type(program), "copy_ndarrays_to_host", copy)
        description = module.add_execution_plan("stateful", graph, snapshot_state=True)
        assert ti.runtime.stats().transfer.device_to_host_bytes - before == 40
        document = json.loads(description.to_json())
        assert len([r for r in document["resources"] if "initial_data" in r]) == 4
        assert not any(r.get("zero_initialize") for r in document["resources"])
        assert len(description.kernel_entries) == 4  # forward, transpose, shared finish/tick
        module.add_execution_plan("current", current, snapshot_state=True)
        module.add_execution_plan("legacy", legacy_graph, snapshot_state=True)
        module.add_execution_plan("plain", plain_graph)
        module.add_execution_plan("composition", composition)
    assert copies == [4, 4, 4, 2, 2], "Shared workspace must be sampled once across forward/adjoint records"
    # Mutation after export cannot change the archived state. Pinned replay still
    # uses the original numeric generation and live producer workspace.
    graph.run(arrays)
    np.testing.assert_allclose(arrays["y"].to_numpy(), 26 * matrix @ x)
    np.testing.assert_allclose(arrays["dx"].to_numpy(), 32 * matrix.T @ (26 * matrix @ x))
    for item in (graph, current, legacy_graph, plain_graph, composition):
        item.close()
    del item, graph, current, legacy_graph, plain_graph, composition, operator, legacy, plain, op
    gc.collect()
    artifact = tmp_path / "graph-operator.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_pinned_from_graph_probe.py")),
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


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_pinned_from_graph_live_field_and_workspace_snapshot(tmp_path, monkeypatch):
    state = ti.field(ti.f32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 2).place(state)
    tree = fields.finalize()
    state.from_numpy(np.array([2, 3], np.float32))
    topology = _array(np.int32, [0, 1])
    workspace = _array(np.float32, [5])

    @ti.kernel
    def apply(
        t: ti.types.ndarray(ti.i32, 1),
        w: ti.types.ndarray(ti.f32, 1),
        x: ti.types.ndarray(ti.f32, 1),
        y: ti.types.ndarray(ti.f32, 1),
    ):
        for i in range(2):
            y[i] = (state[i] + w[0]) * x[t[i]]

    @ti.kernel
    def tick(w: ti.types.ndarray(ti.f32, 1)):
        for i in range(2):
            state[i] += 1
        w[0] += 2

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        apply,
        *(
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)
            for name, dtype in (("t", ti.i32), ("w", ti.f32), ("input", ti.f32), ("output", ti.f32))
        ),
    )
    builder.dispatch(tick, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "w", ti.f32, ndim=1))
    operator = ti.linalg.LinearOperator.from_graph(
        builder.compile(), 2, topology={"t": topology}, workspace={"w": workspace}, state={"field": state}
    )
    graph = outer(operator, adjoint=False)
    arrays = {"x": _array(np.float32, [1, 2]), "y": ti.ndarray(ti.f32, 2)}
    graph.run(arrays)
    np.testing.assert_array_equal(arrays["y"].to_numpy(), [7, 16])
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="snapshot_state=True"):
        module.add_execution_plan("missing_choice", graph)
    assert not module._data_payloads and not module._exported_kernels
    program = ti.lang.impl.get_runtime().prog
    original = type(program)._snapshot_dense_state
    calls = []

    def snapshot(program, roots, arrays):
        calls.append((len(roots), len(arrays)))
        return original(program, roots, arrays)

    def forbidden(*args, **kwargs):
        raise AssertionError("Field and provider state must be sampled together")

    before = ti.runtime.stats().transfer.device_to_host_bytes
    with monkeypatch.context() as guard:
        guard.setattr(type(program), "_snapshot_dense_state", snapshot)
        guard.setattr(type(program), "copy_ndarrays_to_host", forbidden)
        module.add_execution_plan("step", graph, snapshot_state=True)
    assert calls == [(1, 2)]
    assert ti.runtime.stats().transfer.device_to_host_bytes - before == 20
    graph.run(arrays)
    np.testing.assert_array_equal(arrays["y"].to_numpy(), [10, 22])
    artifact = tmp_path / "field-provider.tcm"
    module.archive(str(artifact))
    tree.destroy()
    with pytest.raises((ti.TaichiRuntimeError, RuntimeError), match="destroyed|stale|invalid"):
        graph.run(arrays)
    graph.close()
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_dense_pinned_graph_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
