import json

import pytest

from taichi_forge_runtime.aot import ExecutionPlan, Scalar


def test_selected_vulkan_control_recipe_closure_and_version():
    lowering = dict(
        kind="vulkan_structured",
        version=1,
        counter="counter",
        status="status",
        chunks=[dict(iterations=2, strategy="compact"), dict(iterations=1, strategy="coarse_conditional")],
    )
    loop = dict(
        kind="while",
        predicate="p",
        max_iterations=3,
        condition=[],
        body=[dict(kind="kernel", entry="advance", arguments=["p", "x"])],
        lowering=lowering,
    )
    plan = ExecutionPlan([loop], resources=[dict(kind="ndarray", name="counter", dtype="i32", shape=[1])])
    restored = ExecutionPlan.from_json(plan.to_json())
    assert restored.argument_names == ("p", "status", "x")
    assert restored._replay_argument_names == {"p", "status", "x", "counter"}
    assert restored.kernel_entries == ("advance",)
    document = json.loads(restored.to_json())
    assert document["version"] == 4
    for version in (1, 2, 3):
        with pytest.raises(ValueError):
            ExecutionPlan.from_json(json.dumps(dict(document, version=version)))
    for patch in (
        dict(version=True),
        dict(version=2),
        dict(counter=""),
        dict(chunks=[]),
        dict(chunks=None),
        dict(chunks=[dict(iterations=2, strategy="compact")]),
        dict(chunks=[dict(iterations=4, strategy="compact")]),
        dict(chunks=[dict(iterations=True, strategy="compact")]),
        dict(chunks=[dict(iterations=3, strategy="auto")]),
        dict(chunks=[dict(iterations=3, strategy=[])]),
    ):
        with pytest.raises(ValueError):
            ExecutionPlan([dict(loop, lowering=dict(lowering, **patch))])
    for patch in (dict(continue_while_nonzero=False), dict(body=[dict(kind="fill32", destination="x", value=1)])):
        with pytest.raises(ValueError):
            ExecutionPlan([dict(loop, **patch)])
    # No schema-level 8-chunk/512-iteration cap from the live JIT slot ring.
    many = [dict(iterations=64, strategy="compact")] * 9
    wide = [dict(iterations=0xFFFFFFFF, strategy="compact"), dict(iterations=2, strategy="compact")]
    for chunks in (many, wide):
        assert (
            json.loads(
                ExecutionPlan(
                    [
                        dict(
                            loop,
                            max_iterations=sum(c["iterations"] for c in chunks),
                            lowering=dict(lowering, chunks=chunks),
                        )
                    ]
                ).to_json()
            )["version"]
            == 4
        )


def test_selected_vulkan_nested_recipe_closure_and_boundaries():
    kernel = dict(kind="kernel", entry="payload", arguments=["x"])
    inner = dict(
        predicate="ip",
        counter="ic",
        status="is",
        condition_begin=1,
        body_begin=2,
        end=3,
        max_iterations=65,
        chunk_size=65,
    )
    lowering = dict(kind="vulkan_nested", version=1, counter="oc", inners=[inner])
    loop = dict(
        kind="while", predicate="op", max_iterations=65, condition=[kernel], body=[kernel] * 3, lowering=lowering
    )
    plan = ExecutionPlan([loop], resources=[dict(kind="ndarray", name="ic", dtype="i32", shape=[])])
    assert plan.argument_names == ("ip", "is", "oc", "op", "x")
    assert plan._replay_argument_names == {"ic", "ip", "is", "oc", "op", "x"}
    document = json.loads(plan.to_json())
    assert document["version"] == 5
    assert ExecutionPlan.from_json(plan.to_json()).to_json() == plan.to_json()
    for version in (3, 4):
        with pytest.raises(ValueError):
            ExecutionPlan.from_json(json.dumps(dict(document, version=version)))
    for patch in (
        dict(condition_begin=0),
        dict(body_begin=1),
        dict(end=4),
        dict(end=True),
        dict(chunk_size=0),
        dict(chunk_size=66),
        dict(max_iterations=1 << 32),
        dict(predicate=""),
        dict(extra=1),
    ):
        with pytest.raises(ValueError):
            ExecutionPlan([dict(loop, lowering=dict(lowering, inners=[dict(inner, **patch)]))])
    for patch in (dict(condition=[]), dict(max_iterations=1 << 32), dict(continue_while_nonzero=False)):
        with pytest.raises(ValueError):
            ExecutionPlan([dict(loop, **patch)])
    for patch in (dict(inners=[]), dict(inners=None), dict(chunks=[]), dict(version=True)):
        with pytest.raises(ValueError):
            ExecutionPlan([dict(loop, lowering=dict(lowering, **patch))])
    zero = dict(inner, max_iterations=0, chunk_size=0)
    ExecutionPlan([dict(loop, max_iterations=0, lowering=dict(lowering, inners=[zero]))])
    many = [
        dict(
            inner,
            predicate=f"p{i}",
            counter=f"c{i}",
            status=f"s{i}",
            condition_begin=1 + 3 * i,
            body_begin=2 + 3 * i,
            end=3 + 3 * i,
        )
        for i in range(9)
    ]
    ExecutionPlan([dict(loop, body=[kernel] * 27, lowering=dict(lowering, inners=many))])
    with pytest.raises(ValueError):
        ExecutionPlan([dict(loop, body=[kernel] * 27, lowering=dict(lowering, inners=list(reversed(many))))])


def test_persistent_control_closure_and_version():
    body = [dict(kind="kernel", entry="advance", arguments=["scratch", "x"])]
    loop = dict(
        kind="while",
        predicate="predicate",
        max_iterations=1 << 33,
        condition=[dict(kind="kernel", entry="check", arguments=["predicate", "limit"])],
        body=body,
    )
    plan = ExecutionPlan(
        [loop],
        resources=[
            dict(kind="ndarray", name="scratch", dtype="f32", shape=[dict(array="shape_only", axis=0)]),
            dict(kind="ndarray", name="predicate", dtype="i32", shape=[1]),
        ],
        initialization=[dict(kind="copy", source="initial", destination="scratch")],
        gradients={"scratch": "ds"},
    )
    body[0]["entry"] = "changed"
    restored = ExecutionPlan.from_json(plan.to_json())
    assert restored.kernel_entries == ("check", "advance")
    assert restored.argument_names == ("ds", "initial", "limit", "shape_only", "x")
    assert restored._replay_argument_names == {"predicate", "limit", "scratch", "x", "ds"}
    document = json.loads(restored.to_json())
    assert document["version"] == 3
    assert document["operations"][0]["max_iterations"] == 1 << 33
    for patch in (
        dict(max_iterations=-1),
        dict(max_iterations=True),
        dict(max_iterations=1 << 63),
        dict(continue_while_nonzero=1),
        dict(body=[]),
        dict(condition=None),
        dict(predicate=""),
    ):
        with pytest.raises(ValueError):
            ExecutionPlan([dict(loop, **patch)])
    document["version"] = 2
    with pytest.raises(ValueError, match="version 3"):
        ExecutionPlan.from_json(json.dumps(document))
    for invalid in (None, {}, "bad", 1):
        with pytest.raises(ValueError, match="operation array"):
            ExecutionPlan.from_json(json.dumps(dict(version=3, operations=invalid)))


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


def test_kernel_labels_are_per_dispatch_metadata():
    long_label = "solver/" + "阶段/" * 128 + "前向"
    step = dict(kind="kernel", entry="shared", arguments=["x"], label=long_label)
    plan = ExecutionPlan([step, dict(step, label="solver/reverse")], initialization=[dict(step, label="initialize")])
    step["label"] = "mutated"
    restored = ExecutionPlan.from_json(plan.to_json())
    assert restored.kernel_entries == ("shared",)
    assert restored.argument_names == ("x",)
    document = json.loads(restored.to_json())
    assert [op["label"] for op in document["operations"]] == [long_label, "solver/reverse"]
    assert document["initialization"][0]["label"] == "initialize"
    for invalid in (None, True, 1, "", "invalid\0label", "bad\nlabel", "bad\rlabel", [], {}):
        with pytest.raises(ValueError):
            ExecutionPlan([dict(step, label=invalid)])
    # Labels are kernel dispatch metadata, not arbitrary provider parameters.
    with pytest.raises(ValueError):
        ExecutionPlan([dict(kind="fill32", destination="x", value=0, label="invalid")])


def test_execution_zero_initialization_is_a_strict_resource_policy():
    resource = dict(kind="ndarray", name="state", dtype="i8", shape=[17], zero_initialize=True)
    operation = dict(kind="copy", source="state", destination="out")
    plan = ExecutionPlan([operation], resources=[resource])
    assert plan.argument_names == ("out",)
    restored = ExecutionPlan.from_json(plan.to_json())
    assert json.loads(restored.to_json())["resources"][0]["zero_initialize"] is True
    for invalid in (1, 0, "true", None, [], {}):
        with pytest.raises(ValueError, match="boolean"):
            ExecutionPlan([operation], resources=[dict(resource, zero_initialize=invalid)])


def test_binding_constraints_snapshot_closure_and_validation():
    operations = [
        dict(kind="copy", source="x", destination="scratch"),
        dict(kind="copy", source="scratch", destination="y"),
    ]
    resources = [dict(kind="ndarray", name="scratch", dtype="f32", shape=[17])]
    layout = dict(kind="array_layout", array="x", minimum_shape=[17], record_bytes=4, alignment=4)
    constraints = [layout, dict(kind="disjoint", left="scratch", right="y")]
    plan = ExecutionPlan(operations, resources=resources, constraints=constraints)
    layout["minimum_shape"][0] = 1
    restored = ExecutionPlan.from_json(plan.to_json())
    assert restored.argument_names == ("x", "y")
    assert restored._replay_argument_names == {"x", "scratch", "y"}
    assert json.loads(restored.to_json())["constraints"][0]["minimum_shape"] == [17]
    for patch in (
        dict(minimum_shape=None),
        dict(minimum_shape=[True]),
        dict(minimum_shape=[-1]),
        dict(minimum_shape=[1 << 63]),
        dict(record_bytes=0),
        dict(record_bytes=True),
        dict(alignment=0),
        dict(element_shape=[0]),
        dict(element_shape="2"),
        dict(unknown=1),
        dict(array="unused"),
    ):
        with pytest.raises(ValueError):
            ExecutionPlan(operations, resources=resources, constraints=[dict(layout, **patch)])
    for bad in (None, {}, [None], [dict(kind="unknown")], [dict(kind="disjoint", left="x", right="unused")]):
        document = json.loads(plan.to_json())
        document["constraints"] = bad
        with pytest.raises(ValueError):
            ExecutionPlan.from_json(json.dumps(document))


def test_exact_shape_and_same_view_constraints_are_explicit():
    operations = [dict(kind="copy", source="x", destination="y")]
    layout = dict(kind="array_layout", array="x", exact_shape=[17], record_bytes=4, alignment=4)
    pair = dict(kind="disjoint", left="x", right="y", allow_same_view=True)
    plan = ExecutionPlan(operations, constraints=[layout, pair])
    restored = ExecutionPlan.from_json(plan.to_json())
    assert json.loads(restored.to_json())["constraints"] == [layout, pair]
    for patch in (dict(minimum_shape=[17]), dict(exact_shape=None), dict(exact_shape=[True]), dict(exact_shape=[-1])):
        with pytest.raises(ValueError):
            ExecutionPlan(operations, constraints=[dict(layout, **patch)])
    missing = {k: v for k, v in layout.items() if k != "exact_shape"}
    with pytest.raises(ValueError):
        ExecutionPlan(operations, constraints=[missing])
    for bad in (1, None, "true", [], {}):
        with pytest.raises(ValueError, match="boolean"):
            ExecutionPlan(operations, constraints=[dict(pair, allow_same_view=bad)])


def test_scalar_resource_bits_roundtrip_and_validation():
    cases = (
        ("i8", -128, "80"),
        ("u8", 255, "ff"),
        ("i16", -32768, "8000"),
        ("u16", 65535, "ffff"),
        ("i32", -(1 << 31), "80000000"),
        ("u32", (1 << 32) - 1, "ffffffff"),
        ("i64", -(1 << 63), "8000000000000000"),
        ("u64", (1 << 64) - 1, "ffffffffffffffff"),
        ("f16", 1.5, "3fc00000"),
        ("f32", -0.0, "80000000"),
        ("f64", -0.0, "8000000000000000"),
    )
    for dtype, value, bits in cases:
        resource = Scalar(dtype, value).resource("constant")
        assert resource["bits"] == bits
        plan = ExecutionPlan([dict(kind="kernel", entry="k", arguments=["constant", "out"])], resources=[resource])
        assert plan.argument_names == ("out",)
        assert json.loads(ExecutionPlan.from_json(plan.to_json()).to_json())["resources"] == [resource]
    operation = dict(kind="kernel", entry="k", arguments=["constant"])
    base = Scalar("i32", 1).resource("constant")
    for invalid in (None, True, 1, "", "1", "000000001", "0x000001", "0000000g", "0000 001", "-0000001"):
        with pytest.raises(ValueError):
            ExecutionPlan([operation], resources=[dict(base, bits=invalid)])
    for patch in (dict(dtype="bool"), dict(value=1), dict(dtype="f16", bits="3e00")):
        with pytest.raises(ValueError):
            ExecutionPlan([operation], resources=[dict(base, **patch)])
    with pytest.raises(ValueError, match="gradient"):
        ExecutionPlan([operation], resources=[base], gradients={"constant": "dx"})


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


def test_initial_data_tracks_only_materialized_array_closure():
    key = "a" * 64
    resources = [
        dict(kind="ndarray", name="weights", dtype="f32", shape=[dict(array="shape_only", axis=0)], initial_data=key),
        dict(kind="ndarray", name="shape_only", dtype="f32", shape=[17], initial_data="b" * 64),
        dict(kind="ndarray", name="unused", dtype="f32", shape=[17], initial_data="c" * 64),
    ]
    operations = [dict(kind="copy", source="weights", destination="out")]
    plan = ExecutionPlan(operations, resources=resources)
    assert plan.argument_names == ("out",)
    assert plan.data_entries == (key,)
    assert ExecutionPlan.from_json(plan.to_json()).data_entries == (key,)
    for bad in (None, 1, "", "../data", "G" * 64, "A" * 64):
        with pytest.raises(ValueError, match="initial_data"):
            ExecutionPlan(operations, resources=[dict(resources[0], initial_data=bad), resources[1]])
    with pytest.raises(ValueError, match="initial_data"):
        ExecutionPlan(
            [dict(kind="copy", source="weights", destination="out")],
            resources=[dict(resources[0], zero_initialize=True), resources[1]],
        )


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
