"""A real host-controlled AmgX step exports the same frozen definitions."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_amgx_probe import N, NNZ, coefficients, reference, solver_resource


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("scalar", [ti.f32, ti.f64])
def test_amgx_graph_automatic_export(scalar, tmp_path, monkeypatch):
    vendor = os.environ.get("TI_FORGE_TEST_AMGX_LIBRARY_PATH")
    if not vendor:
        pytest.skip("Real AmgX library required")
    from taichi_forge.hardware._bundled_runtime_provider import _bundled_provider_candidates
    from taichi_forge.hardware._amgx import DEFINITION
    from taichi_forge.graph._native_definition import NativeOperationBinding

    observed = []
    lower = NativeOperationBinding.lower

    def observe(self, *args, **kwargs):
        observed.append(self.definition)
        return lower(self, *args, **kwargs)

    monkeypatch.setattr(NativeOperationBinding, "lower", observe)
    dtype = np.float32 if scalar == ti.f32 else np.float64
    offsets = np.array([0] + [3 * i - 1 for i in range(1, N)] + [NNZ], np.int32)
    columns = np.array([j for i in range(N) for j in range(max(0, i - 1), min(N, i + 2))], np.int32)
    values, rhs, solution = ti.ndarray(scalar, NNZ), ti.ndarray(scalar, N), ti.ndarray(scalar, N)
    seed, diagonal, output = ti.ndarray(scalar, N), ti.ndarray(scalar, 1), ti.ndarray(scalar, N)
    values.from_numpy(np.array(coefficients(4), dtype))
    seed.fill(1)
    diagonal.fill(4)
    solution.fill(0)

    @ti.kernel
    def produce(
        seed: ti.types.ndarray(scalar),
        diagonal: ti.types.ndarray(scalar),
        values: ti.types.ndarray(scalar),
        rhs: ti.types.ndarray(scalar),
    ):
        for i in range(N):
            rhs[i] = 2 * seed[i] + 0.125
            values[3 * i] = diagonal[0]
            if i > 0:
                values[3 * i - 1] = -1
            if i + 1 < N:
                values[3 * i + 1] = -1

    @ti.kernel
    def publish(solution: ti.types.ndarray(scalar), output: ti.types.ndarray(scalar)):
        for i in range(N):
            output[i] = 2 * solution[i] + 1

    config = solver_resource(str(scalar), compute_residual=False)["parameters"]["config"]
    config_path = tmp_path / "amgx.json"
    config_path.write_text(config, encoding="utf8")
    module = ti.aot.Module()
    with ti.hardware.linalg.AmgxProvider(vendor) as provider:
        with provider.solver(offsets, columns, values, config_path, config_file=True, compute_residual=False) as solver:
            config_path.write_text("invalid edited configuration", encoding="utf8")
            offsets.fill(99)
            columns.fill(99)
            binding = solver.bind_device(
                rhs, solution, values=values, zero_initial_guess=False, retain_initial_guess=True
            )
            builder = ti.graph.GraphBuilder()
            arg = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, scalar, ndim=1)
            builder.dispatch(produce, arg("seed"), arg("diagonal"), arg("values"), arg("rhs"))
            builder.append_native(binding.graph_action(update_coefficients=True), admission="explicit")
            builder.dispatch(publish, arg("solution"), arg("output"))
            graph = builder.compile()
            args = dict(seed=seed, diagonal=diagonal, values=values, rhs=rhs, solution=solution, output=output)
            graph.run(args)
            tolerance = 3e-5 if scalar == ti.f32 else 1e-9
            np.testing.assert_allclose(
                output.to_numpy(), 2 * np.array(reference(4, [2.125] * N)) + 1, rtol=tolerance, atol=tolerance
            )
            failed = ti.aot.Module()
            with pytest.raises(ti.TaichiRuntimeError, match="resource_bindings"):
                failed.add_execution_plan("missing", graph)
            assert not failed._execution_plans and not failed._exported_kernels
            for name in ("step", "again"):
                plan = module.add_execution_plan(
                    name,
                    graph,
                    source_bindings=dict(values=values, rhs=rhs, solution=solution),
                    resource_bindings={solver: {"values": "initial"}},
                )
                doc = json.loads(plan.to_json())
                definition = next(r for r in doc["resources"] if r.get("provider") == "amgx")
                assert definition["parameters"] == solver._definition.parameter_dict()
                assert definition["parameters"]["config"] == config
                assert any(
                    d.operation == "solve"
                    and all(d.parameters[k] == v for k, v in binding._definition.parameters.items())
                    for d in observed
                )
                assert [op.get("operation", op["kind"]) for op in doc["operations"]] == [
                    "kernel",
                    "replace_coefficients",
                    "solve",
                    "kernel",
                ]
                assert [op["operation"] for op in doc["initialization"]] == ["setup_csr"]
            # Ordinary execution retains its direct callable and current session.
            diagonal.fill(5)
            solution.fill(float("nan"))
            graph.run(args)
            np.testing.assert_allclose(
                output.to_numpy(), 2 * np.array(reference(5, [2.125] * N)) + 1, rtol=tolerance, atol=tolerance
            )
        with pytest.raises(ti.TaichiRuntimeError, match="closed"):
            graph.run(args)
        graph.close()
        if scalar == ti.f64:
            # Two bindings share numerical solver state, but initialize their
            # guesses independently. Repeated occurrences share binding state.
            offsets = np.array([0] + [3 * i - 1 for i in range(1, N)] + [NNZ], np.int32)
            columns = np.array([j for i in range(N) for j in range(max(0, i - 1), min(N, i + 2))], np.int32)
            values.from_numpy(np.array(coefficients(4), dtype))
            limited_config = solver_resource("f64", compute_residual=False, max_iters=1)["parameters"]["config"]
            with provider.solver(offsets, columns, values, limited_config, compute_residual=False) as limited:
                other_rhs, other_solution = ti.ndarray(scalar, N), ti.ndarray(scalar, N)
                first = limited.bind_device(rhs, solution, zero_initial_guess=False, retain_initial_guess=True)
                second = limited.bind_device(
                    other_rhs, other_solution, zero_initial_guess=False, retain_initial_guess=True
                )
                with pytest.raises(ValueError, match="bound values"):
                    first.graph_action(update_coefficients=True)
                builder = ti.graph.GraphBuilder()
                for bound in (first, second, first):
                    builder.append_native(bound.graph_action(), admission="explicit")
                shared = builder.compile()
                plan = module.add_execution_plan(
                    "binding_states",
                    shared,
                    source_bindings=dict(a_rhs=rhs, a_solution=solution, b_rhs=other_rhs, b_solution=other_solution),
                    resource_bindings={limited: {"values": "initial"}},
                )
                ids = [op["parameters"]["binding_id"] for op in json.loads(plan.to_json())["operations"]]
                assert ids[0] == ids[2] and ids[0] != ids[1]
                # Save ordinary-entry output for comparison with source-free
                # reconstruction and a separate one-step PCG calculation.
                rhs.from_numpy(np.array([1 + i % 3 for i in range(N)], dtype))
                other_rhs.from_numpy(np.array([2 - i % 2 for i in range(N)], dtype))
                solution.fill(0.25)
                other_solution.fill(-0.5)
                shared.run({})
                jit_reference = [solution.to_numpy().tolist(), other_solution.to_numpy().tolist()]
                shared.close()
    artifact = tmp_path / "amgx-recorded.tcm"
    module.archive(str(artifact))
    if scalar == ti.f64:
        artifact.with_suffix(".json").write_text(json.dumps(jit_reference), encoding="utf8")
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_amgx_recording_probe.py")),
                mode,
                str(artifact),
                _bundled_provider_candidates(DEFINITION)[0],
                vendor,
                str(scalar),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
