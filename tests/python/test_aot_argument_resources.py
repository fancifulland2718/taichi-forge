"""Explicit persistent argument policies retain AD and independent owner state."""

import array
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import ExecutionPlan, Scalar
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_argument_resources_graph_export_and_consumption(tmp_path):
    @ti.kernel
    def seed(
        counter: ti.types.ndarray(ti.i32, 1),
        count: ti.types.ndarray(ti.i32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
        loss: ti.types.ndarray(ti.f32, 1),
        dl: ti.types.ndarray(ti.f32, 1),
    ):
        counter[0] += 1
        count[0] = counter[0]
        for i in dx:
            dx[i] = 0
        loss[0], dl[0] = 0, 1

    @ti.kernel
    def energy(
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        scale: ti.f32,
    ):
        for i in x:
            loss[0] += scale * x[i] * x[i]

    dtypes = dict(internal_counter=ti.i32, count=ti.i32, dx=ti.f32, loss=ti.f32, dl=ti.f32, internal_x=ti.f32)
    args = {k: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, k, v, ndim=1) for k, v in dtypes.items()}
    scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "internal_scale", ti.f32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(seed, *(args[k] for k in ("internal_counter", "count", "dx", "loss", "dl")))
    builder.dispatch(energy, args["internal_x"], args["loss"], scale)
    builder.dispatch(energy.grad, args["internal_x"], args["loss"], scale)
    graph = builder.compile()
    module = ti.aot.Module()
    gradients = dict(internal_x="dx", loss="dl")
    policies = [
        dict(name="internal_counter", kind="ndarray", dtype="i32", shape=[1], zero_initialize=True),
        Scalar("f32", 3).resource("internal_scale"),
        dict(name="internal_x", kind="ndarray_view", array="input", offset=0, shape=[5]),
    ]
    for invalid in (
        [dict(name="absent", kind="ndarray", dtype="i32", shape=[1])],
        [dict(name="internal_counter", kind="ndarray", shape=[1])],
        [policies[0], policies[0]],
        [dict(name="internal_counter", kind="ndarray", dtype="i32", shape=[1], initial_data="0" * 64)],
    ):
        with pytest.raises(ValueError):
            module.add_execution_plan("invalid", graph, gradients=gradients, argument_resources=invalid)
        assert not module._fields and not module._array_fields and not module._exported_kernels
        assert not module._content and not module._execution_plans and not module._graph_export_failed
    plan = module.add_execution_plan("step", graph, gradients=gradients, argument_resources=iter(policies))
    assert set(plan.argument_names) == {"input", "count", "dx", "loss", "dl"}
    seed = module.add_data(array.array("i", [40]))
    module.add_execution_plan(
        "seeded",
        graph,
        gradients=gradients,
        argument_resources=[
            dict(name="internal_counter", kind="ndarray", dtype="i32", shape=[1], initial_data=seed),
            *policies[1:],
        ],
    )
    # Explicit plans use the same public policy, without regenerating kernels.
    module.add_execution_plan(
        "explicit",
        plan,
        argument_resources=[dict(name="dl", kind="ndarray", dtype="f32", shape=[1], zero_initialize=True)],
    )
    if ti.cfg.arch == ti.cuda:
        family = json.loads(plan.to_json())
        family.pop("version")
        family["binding_strategy"] = "cuda_immutable_argument_frames"
        module.add_execution_plan(
            "family",
            ExecutionPlan(**family),
            argument_resources=[dict(name="dl", kind="ndarray", dtype="f32", shape=[1], zero_initialize=True)],
        )
    policies[0]["shape"][0] = 100
    counter = next(r for r in json.loads(plan.to_json())["resources"] if r["name"] == "internal_counter")
    assert counter["shape"] == [1]
    artifact = tmp_path / "arguments.tcm"
    module.archive(str(artifact))
    graph.close()
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_argument_resources_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_argument_resources_preserve_original_plan_and_format_features():
    original = ExecutionPlan([dict(kind="copy", source="input", destination="output")])
    specialized = original.with_argument_resources(
        [dict(name="output", kind="ndarray", dtype="i32", shape=[3], usage=["storage"])]
    )
    assert original.argument_names == ("input", "output")
    assert specialized.argument_names == ("input",)
    assert json.loads(original.to_json())["version"] == 1
    assert json.loads(specialized.to_json())["version"] == 18
    assert json.loads(original.to_json())["operations"] == json.loads(specialized.to_json())["operations"]
    with pytest.raises(ValueError, match="public argument"):
        specialized.with_argument_resources([dict(name="output", kind="ndarray", dtype="i32", shape=[4])])
    with pytest.raises(ValueError):
        original.with_argument_resources(
            [
                dict(name="input", kind="ndarray_view", array="output", offset=0, shape=[3]),
                dict(name="output", kind="ndarray_view", array="input", offset=0, shape=[3]),
            ]
        )
    # Binding policy must not discard a selected storage/submission recipe.
    for options in (
        dict(storage_recipe=dict(kind="ordered_temporary_ring", version=1, capacity=3, arrays=["scratch"])),
        dict(submission_recipe=dict(kind="vulkan_segmented", version=1, ends=[1])),
    ):
        selected = ExecutionPlan(
            [dict(kind="kernel", entry="step", arguments=["x", "scratch", "out"])],
            resources=[dict(kind="ndarray", name="scratch", dtype="f32", shape=[3])],
            gradients=dict(x="dx"),
            **options,
        )
        rebound = selected.with_argument_resources(
            [dict(name="x", kind="ndarray_view", array="input", offset=0, shape=[3])]
        )
        assert set(rebound.argument_names) == {"input", "dx", "out"}
        before, after = json.loads(selected.to_json()), json.loads(rebound.to_json())
        for name in ("version", "operations", "gradients", *options):
            assert before[name] == after[name]
