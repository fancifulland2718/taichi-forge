import json

import pytest

from taichi_forge_runtime.aot import ExecutionPlan


def test_execution_plan_snapshot_and_roundtrip(tmp_path):
    steps = [{"kind": "kernel", "entry": "first", "arguments": ["x"]}]
    plan = ExecutionPlan(steps)
    steps[0]["entry"] = "mutated"
    steps[0]["arguments"].append("unbound")
    path = tmp_path / "plan.json"
    plan.save(path)
    restored = ExecutionPlan.load(path)
    assert restored.kernel_entries == ("first",)
    assert json.loads(restored.to_json())["operations"][0]["arguments"] == ["x"]


@pytest.mark.parametrize(
    "document",
    [
        '{"version":1,"version":2,"operations":[]}',
        '{"version":true,"operations":[]}',
        '{"version":1,"operations":[{"kind":"callback"}]}',
        '{"version":1,"operations":[{"kind":"kernel","entry":"x","arguments":[],"hidden":1}]}',
        '{"version":1,"operations":[{"kind":"copy","source":"x","destination":"y","size_bytes":-1}]}',
        '{"version":1,"operations":[{"kind":"fill32","destination":"x","value":4294967296}]}',
        '{"version":1,"operations":[{"kind":"compact","flags":"f","count":"n","columns":[]}]}',
        '{"version":1,"operations":[{"kind":"compact","flags":"f","count":"n","columns":[{"values":"x"}]}]}',
        '{"version":1,"operations":[{"kind":"compact","flags":"f","count":"n","columns":[{"values":"x","output":"y"}],"values":"z","output":"w"}]}',
    ],
)
def test_execution_plan_rejects_ambiguous_contracts(document):
    with pytest.raises(ValueError):
        ExecutionPlan.from_json(document)


def test_execution_resources_have_only_external_slots_and_snapshot_metadata():
    resources = [
        {"kind": "ndarray", "name": "temporary", "dtype": "f32", "shape": [{"array": "shape_only", "axis": 0}]},
        {"kind": "ndarray", "name": "shape_only", "dtype": "f32", "shape": [{"array": "x", "axis": 0}, 3]},
        {"kind": "extent", "name": "n", "dtype": "i32", "array": "temporary", "axis": 0},
        {"kind": "ndarray", "name": "unused", "dtype": "f32", "shape": [{"array": "not_required", "axis": 0}]},
    ]
    plan = ExecutionPlan(
        [{"kind": "kernel", "entry": "forward", "arguments": ["temporary", "n", "y"]}],
        resources=resources,
        gradients={"temporary": "dx"},
    )
    resources[0]["shape"][0]["axis"] = 7
    restored = ExecutionPlan.from_json(plan.to_json())
    assert restored.argument_names == ("dx", "x", "y")
    assert json.loads(restored.to_json())["resources"][0]["shape"][0]["axis"] == 0


@pytest.mark.parametrize(
    "resources",
    [
        [{"kind": "ndarray", "name": "x", "dtype": "i32", "shape": [True]}],
        [{"kind": "ndarray", "name": "x", "dtype": "i32", "shape": [], "element_shape": [0]}],
        [{"kind": "extent", "name": "n", "dtype": "f32", "array": "x", "axis": 0}],
        [{"kind": "ndarray", "name": "x", "dtype": "i32", "shape": []}] * 2,
        [{"kind": "ndarray", "name": "x", "dtype": "i32", "shape": [{"array": "x", "axis": 0}]}],
        [
            {"kind": "ndarray", "name": "x", "dtype": "i32", "shape": [{"array": "n", "axis": 0}]},
            {"kind": "extent", "name": "n", "dtype": "i32", "array": "input", "axis": 0},
        ],
    ],
)
def test_execution_resource_rejects_invalid_descriptions(resources):
    with pytest.raises(ValueError):
        ExecutionPlan([{"kind": "fill32", "destination": "x", "value": 0}], resources=resources)


def test_execution_resource_dependency_walk_is_iterative():
    resources = [
        {"kind": "ndarray", "name": f"r{i}", "dtype": "i32", "shape": [{"array": f"r{i + 1}", "axis": 0}]}
        for i in range(257)
    ]
    plan = ExecutionPlan([{"kind": "fill32", "destination": "r0", "value": 0}], resources=resources)
    assert plan.argument_names == ("r257",)
