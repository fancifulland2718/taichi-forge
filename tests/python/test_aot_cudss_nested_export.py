"""Nested qualified provider bodies preserve selection, boundaries and adjoints."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.hardware._cudss_capture import CudssCaptureRecording
from tests import test_utils
from tests.python.aot_cudss_nested_probe import CASES, reference, shapes
from tests.python.test_graph_cudss_capture import _owner


class ExpandedSolveRecording(CudssCaptureRecording):
    """A test adapter's valid two-operation reconstruction of an idempotent solve.

    Strict disjoint inputs/outputs make repeating the same solve equivalent.
    This checks one physical dispatch expanding at an actual inner boundary;
    the production cuDSS exporter still emits one complete grouped operation.
    """

    def _export_aot(self, context):
        result = super()._export_aot(context)
        super()._export_aot(context)
        return result


@pytest.mark.parametrize(
    "phase,lowering,expanded",
    (("solve", "auto", False), ("refactor_solve", "native_required", False), ("solve", "native_required", True)),
)
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_nested_cudss_export_and_replay(phase, lowering, expanded, tmp_path, monkeypatch):
    if not dict(core.cuda_conditional_graph_capabilities()).get("nested_conditional_graph_available", False):
        pytest.skip("nested CUDA conditional Graph unavailable")
    monkeypatch.delenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", raising=False)
    owner, _, _ = _owner()
    adapter, vendor = owner.provider_identity["library_candidate"], owner.library_path

    @ti.kernel
    def zero(value: ti.types.ndarray(ti.i32, 1)):
        value[0] = 0

    @ti.kernel
    def seed(
        source: ti.types.ndarray(ti.f32, 1),
        rhs: ti.types.ndarray(ti.f32, 1),
        x: ti.types.ndarray(ti.f32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
    ):
        for i in source:
            rhs[i], x[i], dx[i] = source[i], source[i], 0

    @ti.kernel
    def seed_loss(loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def condition(
        count: ti.types.ndarray(ti.i32, 1),
        pred: ti.types.ndarray(ti.i32, 1),
        checks: ti.types.ndarray(ti.i32, 1),
        limit: ti.types.ndarray(ti.i32, 1),
    ):
        pred[0] = int(count[0] < limit[0])
        checks[0] += 1

    @ti.kernel
    def advance(count: ti.types.ndarray(ti.i32, 1)):
        count[0] += 1

    @ti.kernel
    def copy(source: ti.types.ndarray(ti.f32, 1), destination: ti.types.ndarray(ti.f32, 1)):
        for i in source:
            destination[i] = source[i]

    @ti.kernel
    def update(
        base: ti.types.ndarray(ti.f32, 1), values: ti.types.ndarray(ti.f32, 1), total: ti.types.ndarray(ti.i32, 1)
    ):
        for i in base:
            values[i] = base[i] * (total[0] + 2)

    @ti.kernel
    def energy(
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        x2: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in x:
            loss[0] += 0.5 * (x[i] ** 2 + x2[i] ** 2)

    @ti.kernel
    def primal_energy(
        x: ti.types.ndarray(ti.f32, 1),
        x2: ti.types.ndarray(ti.f32, 1),
        loss: ti.types.ndarray(ti.f32, 1),
    ):
        for i in x:
            loss[0] += 0.5 * (x[i] ** 2 + x2[i] ** 2)

    @ti.kernel
    def reverse_seed(total: ti.types.ndarray(ti.i32, 1), count: ti.types.ndarray(ti.i32, 1)):
        count[0] = total[0]

    @ti.kernel
    def reverse_condition(count: ti.types.ndarray(ti.i32, 1), pred: ti.types.ndarray(ti.i32, 1)):
        pred[0] = int(count[0] > 0)

    @ti.kernel
    def decrement(count: ti.types.ndarray(ti.i32, 1)):
        count[0] -= 1

    @ti.kernel
    def terminal(
        outer: ti.types.ndarray(ti.i32, 1),
        inner: ti.types.ndarray(ti.i32, 1),
        total: ti.types.ndarray(ti.i32, 1),
        oc: ti.types.ndarray(ti.i32, 1),
        ic: ti.types.ndarray(ti.i32, 1),
        op: ti.types.ndarray(ti.i32, 1),
        ip: ti.types.ndarray(ti.i32, 1),
        result: ti.types.ndarray(ti.i32, 1),
    ):
        result[0], result[1], result[2] = outer[0], inner[0], total[0]
        result[3], result[4], result[5], result[6] = oc[0], ic[0], op[0], ip[0]

    floats, integers = shapes(phase)
    symbols = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in floats}
    symbols.update({name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1) for name in integers})
    args = lambda *names: [symbols[name] for name in names]

    def build(use_autodiff):
        builder = ti.graph.GraphBuilder()
        for name in ("outer", "inner", "outer_checks", "inner_checks", "total", "op", "ip"):
            builder.dispatch(zero, symbols[name])
        builder.dispatch(seed, *args("source", "rhs", "x", "dx"))
        builder.dispatch(seed, *args("source2", "rhs2", "x2", "dx2"))
        builder.dispatch(seed_loss, *args("loss", "dloss"))
        oc, ob, ic, ib = (builder.create_sequential() for _ in range(4))
        oc.dispatch(condition, *args("outer", "op", "outer_checks", "outer_limit"))
        ic.dispatch(condition, *args("inner", "ip", "inner_checks", "inner_limit"))
        if phase != "solve":
            ib.dispatch(update, *args("base_values", "values", "total"))
        recording = ExpandedSolveRecording if expanded else CudssCaptureRecording
        ib.append_native(recording(owner, phase=phase, values="values", _rhs_pairs=(("rhs", "x"), ("rhs2", "x2"))))
        ib.dispatch(copy, *args("x", "rhs"))
        ib.dispatch(copy, *args("x2", "rhs2"))
        ib.dispatch(advance, symbols["inner"])
        ib.dispatch(advance, symbols["total"])
        ob.dispatch(zero, symbols["inner"])
        ob.while_loop(ic, ib, predicate=symbols["ip"], counter=symbols["inner"], max_iterations=3, name="inner")
        ob.dispatch(advance, symbols["outer"])
        builder.while_loop(
            oc,
            ob,
            predicate=symbols["op"],
            counter=symbols["outer"],
            max_iterations=2,
            lowering_mode=lowering,
            name="outer",
        )
        builder.dispatch(energy if use_autodiff else primal_energy, *args("x", "x2", "loss"))
        if use_autodiff:
            builder.dispatch(energy.grad, *args("x", "x2", "loss"))
        else:
            builder.dispatch(copy, *args("x", "dx"))
            builder.dispatch(copy, *args("x2", "dx2"))
        builder.dispatch(reverse_seed, *args("total", "reverse"))
        builder.dispatch(copy, *args("dx", "adjoint_rhs"))
        builder.dispatch(copy, *args("dx2", "adjoint_rhs2"))
        rc, rb = builder.create_sequential(), builder.create_sequential()
        rc.dispatch(reverse_condition, *args("reverse", "rp"))
        rb.dispatch(decrement, symbols["reverse"])
        if phase != "solve":
            rb.dispatch(update, *args("base_values", "values", "reverse"))
        rb.append_native(
            CudssCaptureRecording(
                owner,
                phase=phase,
                values="values",
                _rhs_pairs=(("adjoint_rhs", "gradient"), ("adjoint_rhs2", "gradient2")),
            )
        )
        rb.dispatch(copy, *args("gradient", "adjoint_rhs"))
        rb.dispatch(copy, *args("gradient2", "adjoint_rhs2"))
        builder.while_loop(
            rc,
            rb,
            predicate=symbols["rp"],
            counter=symbols["reverse"],
            max_iterations=6,
            lowering_mode="native_required",
            name="reverse",
        )
        builder.dispatch(copy, *args("adjoint_rhs", "gradient"))
        builder.dispatch(copy, *args("adjoint_rhs2", "gradient2"))
        builder.dispatch(
            terminal, *args("outer", "inner", "total", "outer_checks", "inner_checks", "op", "ip", "terminal")
        )
        return builder.freeze()

    definition = build(False)
    module = ti.aot.Module()
    # JIT Graph keeps explicit adjoints; AOT binds real ndarray gradients.
    aot_definition = build(True)
    with aot_definition.materialize() as aot_product:
        plan = module.add_execution_plan("step", aot_product.executor, gradients=dict(x="dx", x2="dx2", loss="dloss"))
        document = json.loads(plan.to_json())
        outer = next(op for op in document["operations"] if op["kind"] == "while")
        inner = next(op for op in outer["body"] if op["kind"] == "while")
        assert sum(op.get("provider") == "cudss" for op in inner["body"]) == (2 if expanded else 1)
    with definition.materialize() as materialized:
        graph = materialized.executor
        control = graph._spec.structured_control_nodes[0]
        assert control._cuda_nested_control_lowering == "cuda_conditional_nested_graph"
        assert not control._cuda_nested.nested_expanded_eligible
        assert control._cuda_nested_submission_options["execution_route"] == "conditional"
        frames = []
        for index in range(2):
            arrays = {
                name: ti.ndarray(ti.f32, count, needs_grad=name in ("x", "x2", "loss"))
                for name, count in floats.items()
            }
            arrays.update({name: ti.ndarray(ti.i32, count) for name, count in integers.items()})
            for primal, gradient in (("x", "dx"), ("x2", "dx2"), ("loss", "dloss")):
                arrays[gradient] = arrays[primal].grad
            if phase != "solve":
                arrays["base_values"].from_numpy(np.array([4, -1, -1, 4, -1, -1, 4, -1, -1, 3], np.float32))
            source = [(i - 1) / 4 + index for i in range(4)]
            source2 = [-2 * x + 0.125 for x in source]
            arrays["source"].from_numpy(np.array(source, np.float32))
            arrays["source2"].from_numpy(np.array(source2, np.float32))
            frames.append((arrays, source, source2))
        for limits in CASES:
            for arrays, source, source2 in frames:
                for name, value in zip(("outer_limit", "inner_limit"), limits):
                    arrays[name].fill(value)
                graph.submit(arrays).wait()
                expected, state = reference(source, source2, limits, phase)
                for name, values in expected.items():
                    np.testing.assert_allclose(arrays[name].to_numpy(), values, atol=3e-5, rtol=3e-5)
                np.testing.assert_array_equal(arrays["terminal"].to_numpy(), state)
        assert any(stat["last_path"].startswith("cuda_conditional_nested_") for stat in graph._graph_stats)
    artifact = tmp_path / "cudss-nested.tcm"
    module.archive(str(artifact))
    del graph, materialized, control, definition, aot_definition, aot_product
    gc.collect()
    owner.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cudss_nested_probe.py")),
                mode,
                str(artifact),
                adapter,
                vendor,
                phase,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
