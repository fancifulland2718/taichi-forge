"""Preserve captured cuDSS groups through selected device loops and AOT."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.hardware._cudss_capture import CudssCaptureRecording
from taichi_forge.graph._native import BackendCommandGraphAction, NativeGraphNode
from tests import test_utils
from tests.python.test_graph_cudss_capture import _owner


@pytest.mark.parametrize("phase", ("solve", "factor_solve", "refactor_solve"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_selected_control_export_and_adjoint(tmp_path, phase):
    owner, _, _ = _owner()
    adapter, vendor = owner.provider_identity["library_candidate"], owner.library_path

    @ti.kernel
    def initialize(
        source: ti.types.ndarray(ti.f32, 1),
        source2: ti.types.ndarray(ti.f32, 1),
        rhs: ti.types.ndarray(ti.f32, 1),
        rhs2: ti.types.ndarray(ti.f32, 1),
        x: ti.types.ndarray(ti.f32, 1),
        x2: ti.types.ndarray(ti.f32, 1),
        count: ti.types.ndarray(ti.i32, 1),
        checks: ti.types.ndarray(ti.i32, 1),
    ):
        count[0], checks[0] = 0, 0
        for i in source:
            rhs[i], x[i] = source[i], source[i]
            rhs2[i], x2[i] = source2[i], source2[i]

    @ti.kernel
    def condition(
        count: ti.types.ndarray(ti.i32, 1),
        predicate: ti.types.ndarray(ti.i32, 1),
        checks: ti.types.ndarray(ti.i32, 1),
        target: ti.types.ndarray(ti.i32, 1),
        reverse: ti.template(),
    ):
        if ti.static(reverse):
            predicate[0] = int(count[0] > 0)
        else:
            predicate[0] = int(count[0] < target[0])
        checks[0] += 1

    @ti.kernel
    def scale_values(
        base: ti.types.ndarray(ti.f32, 1), values: ti.types.ndarray(ti.f32, 1), count: ti.types.ndarray(ti.i32, 1)
    ):
        for i in values:
            values[i] = base[i] * (count[0] + 2)

    @ti.kernel
    def advance(
        x: ti.types.ndarray(ti.f32, 1),
        x2: ti.types.ndarray(ti.f32, 1),
        rhs: ti.types.ndarray(ti.f32, 1),
        rhs2: ti.types.ndarray(ti.f32, 1),
        count: ti.types.ndarray(ti.i32, 1),
    ):
        for i in x:
            rhs[i], rhs2[i] = x[i], x2[i]
        count[0] += 1

    @ti.kernel
    def adjoint_seed(
        x: ti.types.ndarray(ti.f32, 1),
        rhs: ti.types.ndarray(ti.f32, 1),
        count: ti.types.ndarray(ti.i32, 1),
        reverse_count: ti.types.ndarray(ti.i32, 1),
        checks: ti.types.ndarray(ti.i32, 1),
        loss: ti.types.ndarray(ti.f32, 1),
    ):
        reverse_count[0], checks[0], loss[0] = count[0], 0, 0.0
        for i in x:
            rhs[i] = x[i]
            loss[0] += 0.5 * x[i] * x[i]

    @ti.kernel
    def decrement(count: ti.types.ndarray(ti.i32, 1)):
        count[0] -= 1

    @ti.kernel
    def copy(source: ti.types.ndarray(ti.f32, 1), destination: ti.types.ndarray(ti.f32, 1)):
        for i in source:
            destination[i] = source[i]

    @ti.kernel
    def terminal(
        count: ti.types.ndarray(ti.i32, 1),
        checks: ti.types.ndarray(ti.i32, 1),
        active: ti.types.ndarray(ti.i32, 1),
        reverse_count: ti.types.ndarray(ti.i32, 1),
        reverse_checks: ti.types.ndarray(ti.i32, 1),
        reverse_active: ti.types.ndarray(ti.i32, 1),
        output: ti.types.ndarray(ti.i32, 1),
    ):
        output[0], output[1], output[2] = count[0], checks[0], active[0]
        output[3], output[4], output[5] = reverse_count[0], reverse_checks[0], reverse_active[0]

    from tests.python.aot_cudss_control_probe import reference, shapes

    floats, ints = shapes(phase)
    symbols = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in floats}
    symbols.update({name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1) for name in ints})
    args = lambda *names: [symbols[name] for name in names]
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        initialize, *args("source", "source2", "rhs", "rhs2", "solution", "solution2", "counter", "checks")
    )
    forward_condition, forward = builder.create_sequential(), builder.create_sequential()
    forward_condition.dispatch(condition, *args("counter", "active", "checks", "target"), template_args={"reverse": 0})
    if phase == "solve":
        recording = CudssCaptureRecording(owner, phase=phase, rhs="rhs", solution="solution")

        class UnqualifiedCapture(NativeGraphNode):
            def compile(self):
                executable = recording._as_graph_native_node().compile()
                executable._action = BackendCommandGraphAction(recording)
                return executable

        rejected = builder.create_sequential()
        rejected.append_native(UnqualifiedCapture())
        with pytest.raises(ti.TaichiRuntimeError, match="not qualified.*region"):
            builder.while_loop(forward_condition, rejected, predicate=symbols["active"], max_iterations=3)
    if phase != "solve":
        forward.dispatch(scale_values, *args("base_values", "values", "counter"))
    forward.append_native(
        CudssCaptureRecording(
            owner, phase=phase, values="values", _rhs_pairs=(("rhs", "solution"), ("rhs2", "solution2"))
        )
    )
    forward.dispatch(advance, *args("solution", "solution2", "rhs", "rhs2", "counter"))
    builder.while_loop(
        forward_condition,
        forward,
        predicate=symbols["active"],
        counter=symbols["counter"],
        max_iterations=3,
        lowering_mode="native_required",
    )
    builder.dispatch(
        adjoint_seed, *args("solution", "adjoint_rhs", "counter", "reverse_count", "reverse_checks", "loss")
    )
    reverse_condition, reverse = builder.create_sequential(), builder.create_sequential()
    reverse_condition.dispatch(
        condition, *args("reverse_count", "reverse_active", "reverse_checks", "target"), template_args={"reverse": 1}
    )
    reverse.dispatch(decrement, *args("reverse_count"))
    if phase != "solve":
        reverse.dispatch(scale_values, *args("base_values", "values", "reverse_count"))
    reverse.append_native(
        CudssCaptureRecording(owner, phase=phase, values="values", rhs="adjoint_rhs", solution="adjoint")
    )
    reverse.dispatch(copy, *args("adjoint", "adjoint_rhs"))
    builder.while_loop(
        reverse_condition,
        reverse,
        predicate=symbols["reverse_active"],
        counter=symbols["reverse_count"],
        max_iterations=3,
        lowering_mode="native_required",
    )
    builder.dispatch(copy, *args("adjoint_rhs", "gradient"))
    builder.dispatch(
        terminal, *args("counter", "checks", "active", "reverse_count", "reverse_checks", "reverse_active", "terminal")
    )
    graph = builder.compile()
    module = ti.aot.Module()
    plan = module.add_execution_plan("step", graph)
    data = json.loads(plan.to_json())
    loop_index = next(i for i, op in enumerate(data["operations"]) if op["kind"] == "while")
    for name in ("zero", "nested"):
        variant = json.loads(plan.to_json())
        loop = variant["operations"][loop_index]
        if name == "zero":
            loop["max_iterations"] = 0
        else:
            variant["operations"][loop_index] = dict(
                kind="while", predicate=loop["predicate"], max_iterations=1, condition=[], body=[loop]
            )
        module.add_execution_plan(name, ti.aot.ExecutionPlan.from_json(json.dumps(variant)))
    binding_frames = []
    for index in range(2):
        frames = {name: ti.ndarray(ti.f32, count) for name, count in floats.items()}
        frames.update({name: ti.ndarray(ti.i32, count) for name, count in ints.items()})
        source = np.arange(4, dtype=np.float32) / 4 - 0.25 + index
        source2 = -2 * source + 0.125
        frames["source"].from_numpy(source)
        frames["source2"].from_numpy(source2)
        if phase != "solve":
            frames["base_values"].from_numpy(np.array([4, -1, -1, 4, -1, -1, 4, -1, -1, 3], np.float32))
        binding_frames.append((frames, source, source2))
    for limit in (0, 2, 9, 1):
        for frames, source, source2 in binding_frames:
            frames["target"].fill(limit)
            graph.submit(frames).wait()
            expected, state = reference(source.tolist(), source2.tolist(), limit, phase)
            for name, values in expected.items():
                np.testing.assert_allclose(frames[name].to_numpy(), values, rtol=3e-5, atol=3e-5)
            np.testing.assert_array_equal(frames["terminal"].to_numpy(), state)
    artifact = tmp_path / "cudss-control.tcm"
    module.archive(str(artifact))
    graph.close()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cudss_control_probe.py")),
                mode,
                str(artifact),
                adapter,
                vendor,
                phase,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
