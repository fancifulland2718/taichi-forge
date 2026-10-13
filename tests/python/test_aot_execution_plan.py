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


def test_initialization_version_and_runtime_closure():
    initial = [dict(kind="copy", source="input", destination="weights")]
    plan = ExecutionPlan(
        [
            dict(
                kind="native",
                provider="test",
                operation="use",
                version=1,
                parameters={},
                bindings={"plan": "p", "out": "output"},
            )
        ],
        initialization=initial,
        resources=[
            dict(
                kind="native",
                name="p",
                provider="test",
                operation="plan",
                version=1,
                parameters={},
                bindings={"values": "weights"},
            ),
            dict(kind="ndarray", name="weights", dtype="f32", shape=[dict(array="shape_only", axis=0)]),
        ],
        gradients={"weights": "dw"},
    )
    initial[0]["source"] = "changed"
    restored = ExecutionPlan.from_json(plan.to_json())
    assert restored.argument_names == ("dw", "input", "output", "shape_only")
    assert restored._replay_argument_names == {"p", "weights", "dw", "output"}
    assert restored._retained_argument_names == {"p", "weights", "dw", "output", "input"}
    document = json.loads(plan.to_json())
    assert document["version"] == 2
    for patch in ({"version": 1}, {"initialization": []}, {"initialization": None}):
        with pytest.raises(ValueError):
            ExecutionPlan.from_json(json.dumps(dict(document, **patch)))


def test_native_description_is_provider_owned_and_has_symbolic_resources():
    op = dict(
        kind="native",
        provider="application.library",
        operation="custom_operation",
        version=7,
        parameters={"nested": {"policy": "frozen"}},
        bindings={"input": "x", "output": "scratch"},
    )
    plan = ExecutionPlan([op], resources=[{"kind": "ndarray", "name": "scratch", "dtype": "f32", "shape": [3]}])
    op["parameters"]["nested"]["policy"] = "changed"
    restored = ExecutionPlan.from_json(plan.to_json())
    assert restored.argument_names == ("x",)
    assert restored.kernel_entries == ()
    assert json.loads(restored.to_json())["operations"][0]["parameters"]["nested"]["policy"] == "frozen"


@pytest.mark.parametrize(
    "invalid",
    [
        {"version": 0},
        {"version": True},
        {"version": 1 << 32},
        {"parameters": []},
        {"bindings": {"input": ""}},
        {"bindings": {"": "x"}},
        {"provider": ""},
        {"operation": "bad\0name"},
        {"callback": "python.function"},
    ],
)
def test_native_description_rejects_invalid_envelopes(invalid):
    op = dict(kind="native", provider="library", operation="operation", version=1, parameters={}, bindings={})
    with pytest.raises(ValueError):
        ExecutionPlan([dict(op, **invalid)])


@pytest.mark.parametrize(
    "document",
    [
        '{"version":1,"version":2,"operations":[]}',
        '{"version":true,"operations":[]}',
        '{"version":1,"operations":[{"kind":"callback"}]}',
        '{"version":1,"operations":[{"kind":"kernel","entry":"x","arguments":[],"hidden":1}]}',
        '{"version":1,"operations":[{"kind":"copy","source":"x","destination":"y","size_bytes":-1}]}',
        '{"version":1,"operations":[{"kind":"fill32","destination":"x","value":4294967296}]}',
        '{"version":1,"operations":[{"kind":"fft_c2c_f32","input":"x","output":"y","dimensions":[],"direction":"forward"}]}',
        '{"version":1,"operations":[{"kind":"fft_c2c_f32","input":"x","output":"y","dimensions":[true],"direction":"forward"}]}',
        '{"version":1,"operations":[{"kind":"fft_c2c_f32","input":"x","output":"y","dimensions":[2],"direction":"forward","batch_count":false}]}',
        '{"version":1,"operations":[{"kind":"fft_c2c_f32","input":"x","output":"y","dimensions":[2],"direction":"adjoint"}]}',
        '{"version":1,"operations":[{"kind":"compact","flags":"f","count":"n","columns":[]}]}',
        '{"version":1,"operations":[{"kind":"compact","flags":"f","count":"n","columns":[{"values":"x"}]}]}',
        '{"version":1,"operations":[{"kind":"compact","flags":"f","count":"n","columns":[{"values":"x","output":"y"}],"values":"z","output":"w"}]}',
    ],
)
def test_execution_plan_rejects_ambiguous_contracts(document):
    with pytest.raises(ValueError):
        ExecutionPlan.from_json(document)


@pytest.mark.parametrize(
    "invalid",
    [
        {"rows": 0},
        {"inner": True},
        {"columns": -1},
        {"alpha": True},
        {"beta": 1e100},
        {"transpose_a": 1},
        {"transpose_b": "N"},
    ],
)
def test_gemm_description_rejects_invalid_scalar_shape_and_transpose(invalid):
    op = dict(kind="gemm_f32", a="a", b="b", output="c", rows=17, columns=9, inner=13)
    with pytest.raises(ValueError):
        ExecutionPlan([dict(op, **invalid)])


def test_gemm_description_snapshot_and_roundtrip():
    op = dict(
        kind="gemm_f32", a="a", b="b", output="c", rows=17, columns=9, inner=13, alpha=-0.75, beta=2, transpose_b=True
    )
    plan = ExecutionPlan([op])
    op["transpose_b"] = False
    restored = ExecutionPlan.from_json(plan.to_json())
    assert restored.argument_names == ("a", "b", "c")
    assert restored.kernel_entries == ()
    assert json.loads(restored.to_json())["operations"][0]["transpose_b"] is True


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


def test_native_resource_closure_and_kind_validation():
    def native(name, **bindings):
        return dict(
            kind="native", name=name, provider="test", operation="plan", version=1, parameters={}, bindings=bindings
        )

    op = {key: value for key, value in native("op", plan="child", output="y").items() if key != "name"}
    resources = [
        native("child", context="parent", input="scratch"),
        native("parent"),
        dict(kind="ndarray", name="scratch", dtype="f32", shape=[{"array": "x", "axis": 0}]),
        native("unused", missing="not_required"),
    ]
    plan = ExecutionPlan([op], resources=resources)
    assert plan.argument_names == ("x", "y")
    assert ExecutionPlan.from_json(plan.to_json()).argument_names == plan.argument_names
    with pytest.raises(ValueError, match="gradient reference"):
        ExecutionPlan([op], resources=resources, gradients={"parent": "scratch"})
    for bad in (
        [native("child", cycle="child")],
        [native("child")] * 2,
        [dict(native("child"), version=True)],
        [dict(native("child"), dtype="f32")],
        [native("parent"), dict(kind="ndarray", name="child", dtype="f32", shape=[{"array": "parent", "axis": 0}])],
    ):
        with pytest.raises(ValueError):
            ExecutionPlan([op], resources=bad)


def test_fft_description_preserves_direction_and_static_shape():
    plan = ExecutionPlan(
        [
            {
                "kind": "fft_c2c_f32",
                "input": "x",
                "output": "y",
                "dimensions": [3, 5],
                "batch_count": 2,
                "direction": "inverse",
            }
        ]
    )
    restored = ExecutionPlan.from_json(plan.to_json())
    assert restored.argument_names == ("x", "y")
    assert restored.kernel_entries == ()
    assert json.loads(restored.to_json())["operations"][0]["dimensions"] == [3, 5]
