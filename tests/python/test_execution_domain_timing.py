"""Sampled completion timing preserves ticket identity and immutable results."""

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import ExecutionPlan, Runtime, Scalar
from tests import test_utils
from tests.python.test_submission_parameters import NAMES, graph, publish, values


def samples(call):
    kept = []
    for index in range(16):
        sampled = index % 3 == 0
        ticket = call.submit(values=values(index, 0), sample_gpu=sampled)
        assert ticket is not None
        before = ticket.gpu_timing
        assert before["status"] in (("pending", "ready") if sampled else ("not_sampled",))
        if before["status"] != "ready":
            assert before["duration_ns"] is None
        ticket.wait()
        result = ticket.gpu_timing
        assert result["owner"] == ticket.info["owner"]
        assert result["sequence"] == ticket.info["sequence"]
        assert result["scope"] == "compute_envelope"
        if sampled:
            assert result["status"] == "ready" and result["duration_ns"] > 0
        else:
            assert result["status"] == "not_sampled" and result["duration_ns"] is None
        kept.append((ticket, result))
    return kept


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_jit_completion_timing_is_opt_in_and_survives_close():
    source = graph()
    output = ti.ndarray(ti.u64, (1000, 3))
    args = dict(out=output, **values(0, 0))
    ordinary = ti.graph.execution_domain()
    fixed = source.prepare(args, execution_domain=ordinary, mutable_parameters=NAMES)
    fixed.initialize()
    with pytest.raises(RuntimeError, match="enabled"):
        fixed.submit(values=values(0, 0), sample_gpu=True)
    assert ordinary.latest_completion().info["sequence"] == 0
    domain = ti.graph.execution_domain(capacity=2, gpu_timing=True)
    call = source.prepare(args, execution_domain=domain, mutable_parameters=NAMES)
    call.initialize()
    kept = samples(call)
    for index in range(16):
        np.testing.assert_array_equal(output.to_numpy()[index], tuple(values(index, 0).values()))
    call.close()
    fixed.close()
    domain.close()
    ordinary.close()
    for ticket, expected in kept:
        assert ticket.gpu_timing == expected
        ticket.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_completion_timing_sparse_samples(tmp_path):
    artifact = ti.aot.Module()
    artifact.add_kernel(publish)
    artifact.add_execution_plan("publish", ExecutionPlan([
        dict(kind="kernel", entry="publish", arguments=["out", *NAMES])]))
    artifact.save(tmp_path / "module")
    with Runtime("cuda") as runtime:
        module = runtime.load_module(tmp_path / "module")
        domain = runtime.execution_domain(gpu_timing=True)
        output = runtime.ndarray("u64", (1000, 3))
        call = module.prepare_execution("publish", dict(out=output, **{
            name: Scalar("u64", value) for name, value in values(0, 0).items()}),
            execution_domain=domain, mutable_parameters=NAMES)
        call.initialize()
        kept = samples(call)
    for ticket, expected in kept:
        assert ticket.gpu_timing == expected
        ticket.close()
