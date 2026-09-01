import pytest

import taichi_forge as ti
from taichi_forge.graph import _graph as graph_impl
from taichi_forge.graph._native import ProviderOwnedNdarrayBinding
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils


def _scalar_arg(name):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)


@pytest.mark.parametrize(
    ("kind", "invalid_kind"),
    (("if", "view"), ("switch", "dtype"), ("while", "shape")),
)
@test_utils.test(arch=ti.cpu)
def test_control_preflight_rejects_invalid_scalar_before_condition_side_effect(
    kind, invalid_kind
):
    @ti.kernel
    def set_control(
        control: ti.types.ndarray(dtype=ti.i32, ndim=0),
        condition_runs: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        condition_runs[None] += 1
        control[None] = 1

    @ti.kernel
    def increment(value: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        value[None] += 1

    control_arg = _scalar_arg("control")
    condition_runs_arg = _scalar_arg("condition_runs")
    output_arg = _scalar_arg("output")
    builder = ti.graph.GraphBuilder()
    condition = builder.create_sequential()
    condition.dispatch(set_control, control_arg, condition_runs_arg)
    action = builder.create_sequential()
    action.dispatch(increment, output_arg)
    if kind == "if":
        builder.if_then_else(condition, action, predicate=control_arg)
    elif kind == "switch":
        builder.switch(condition, (action,), selector=control_arg)
    else:
        builder.while_loop(
            condition,
            action,
            predicate=control_arg,
            max_iterations=1,
        )
    graph = builder.compile()

    if invalid_kind == "view":
        control = ti.experimental.ndarray_view(ti.ndarray(ti.i32, shape=()))
    elif invalid_kind == "dtype":
        control = ti.ndarray(ti.f32, shape=())
    else:
        control = ti.ndarray(ti.i32, shape=2)
    condition_runs = ti.ndarray(ti.i32, shape=())
    output = ti.ndarray(ti.i32, shape=())
    condition_runs.fill(0)
    output.fill(0)

    with pytest.raises(TaichiRuntimeError, match="Structured Graph control"):
        graph.run(
            {
                "control": control,
                "condition_runs": condition_runs,
                "output": output,
            }
        )

    assert condition_runs.to_numpy()[()] == 0
    assert output.to_numpy()[()] == 0


@test_utils.test(arch=ti.cpu)
def test_flat_while_control_roles_reject_exact_allocation_alias_before_condition():
    @ti.kernel
    def evaluate(
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        condition_runs: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        condition_runs[None] += 1
        predicate[None] = int(counter[None] < 1)
        status[None] = 7

    @ti.kernel
    def step(
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        output: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        counter[None] += 1
        output[None] += 1

    predicate_arg = _scalar_arg("predicate")
    counter_arg = _scalar_arg("counter")
    status_arg = _scalar_arg("status")
    condition_runs_arg = _scalar_arg("condition_runs")
    output_arg = _scalar_arg("output")
    builder = ti.graph.GraphBuilder()
    condition = builder.create_sequential()
    condition.dispatch(
        evaluate,
        counter_arg,
        predicate_arg,
        status_arg,
        condition_runs_arg,
    )
    body = builder.create_sequential()
    body.dispatch(step, counter_arg, output_arg)
    builder.while_loop(
        condition,
        body,
        predicate=predicate_arg,
        control_inputs=(counter_arg,),
        counter=counter_arg,
        status=status_arg,
        max_iterations=2,
    )
    graph = builder.compile()

    predicate = ti.ndarray(ti.i32, shape=())
    counter_and_status = ti.ndarray(ti.i32, shape=())
    condition_runs = ti.ndarray(ti.i32, shape=())
    output = ti.ndarray(ti.i32, shape=())
    for value in (predicate, counter_and_status, condition_runs, output):
        value.fill(0)
    args = {
        "predicate": predicate,
        "counter": ProviderOwnedNdarrayBinding(counter_and_status.arr, object()),
        "status": ProviderOwnedNdarrayBinding(counter_and_status.arr, object()),
        "condition_runs": condition_runs,
        "output": output,
    }

    with pytest.raises(TaichiRuntimeError, match="control resources must not alias"):
        graph.run(args)
    with pytest.raises(TaichiRuntimeError, match="control resources must not alias"):
        graph.bind(args)

    assert condition_runs.to_numpy()[()] == 0
    assert output.to_numpy()[()] == 0


@test_utils.test(arch=ti.cpu)
def test_flat_while_rejects_one_symbolic_resource_for_two_control_roles():
    @ti.kernel
    def stop(control: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        control[None] = 0

    @ti.kernel
    def step(control: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        control[None] += 1

    control = _scalar_arg("control")
    builder = ti.graph.GraphBuilder()
    condition = builder.create_sequential()
    condition.dispatch(stop, control)
    body = builder.create_sequential()
    body.dispatch(step, control)
    builder.while_loop(
        condition,
        body,
        predicate=control,
        counter=control,
        max_iterations=2,
    )

    with pytest.raises(
        TaichiRuntimeError, match="control resources must be independent"
    ):
        builder.compile()


@test_utils.test(arch=ti.cpu)
def test_nested_sibling_controls_reject_exact_allocation_alias_before_root_condition():
    @ti.kernel
    def set_true(
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        condition_runs: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        condition_runs[None] += 1
        predicate[None] = 1

    @ti.kernel
    def increment(value: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        value[None] += 1

    outer_predicate = _scalar_arg("outer_predicate")
    first_predicate = _scalar_arg("first_predicate")
    second_predicate = _scalar_arg("second_predicate")
    condition_runs = _scalar_arg("condition_runs")
    output = _scalar_arg("output")
    builder = ti.graph.GraphBuilder()

    outer_condition = builder.create_sequential()
    outer_condition.dispatch(set_true, outer_predicate, condition_runs)
    outer_then = builder.create_sequential()
    for name, predicate in (
        ("first", first_predicate),
        ("second", second_predicate),
    ):
        inner_condition = builder.create_sequential()
        inner_condition.dispatch(set_true, predicate, condition_runs)
        inner_then = builder.create_sequential()
        inner_then.dispatch(increment, output)
        outer_then.if_then_else(
            inner_condition,
            inner_then,
            predicate=predicate,
            name=name,
        )
    builder.if_then_else(
        outer_condition,
        outer_then,
        predicate=outer_predicate,
        name="outer",
    )
    graph = builder.compile()

    shared_sibling = ti.ndarray(ti.i32, shape=())
    first_sibling = ProviderOwnedNdarrayBinding(shared_sibling.arr, object())
    second_sibling = ProviderOwnedNdarrayBinding(shared_sibling.arr, object())
    condition_runs_value = ti.ndarray(ti.i32, shape=())
    output_value = ti.ndarray(ti.i32, shape=())
    condition_runs_value.fill(0)
    output_value.fill(0)
    with pytest.raises(TaichiRuntimeError, match="control resources must not alias"):
        graph.run(
            {
                "outer_predicate": ti.ndarray(ti.i32, shape=()),
                "first_predicate": first_sibling,
                "second_predicate": second_sibling,
                "condition_runs": condition_runs_value,
                "output": output_value,
            }
        )

    assert condition_runs_value.to_numpy()[()] == 0
    assert output_value.to_numpy()[()] == 0


@test_utils.test(arch=ti.cpu)
def test_stable_control_binding_version_certifies_once_and_keeps_values_dynamic(
    monkeypatch,
):
    @ti.kernel
    def evaluate(
        decision: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        predicate[None] = decision[None]

    @ti.kernel
    def increment(output: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        output[None] += 1

    decision_arg = _scalar_arg("decision")
    predicate_arg = _scalar_arg("predicate")
    output_arg = _scalar_arg("output")
    builder = ti.graph.GraphBuilder()
    condition = builder.create_sequential()
    condition.dispatch(evaluate, decision_arg, predicate_arg)
    then_region = builder.create_sequential()
    then_region.dispatch(increment, output_arg)
    builder.if_then_else(
        condition,
        then_region,
        predicate=predicate_arg,
        control_inputs=(decision_arg,),
    )
    graph = builder.compile()

    decision = ti.ndarray(ti.i32, shape=())
    predicate = ti.ndarray(ti.i32, shape=())
    output = ti.ndarray(ti.i32, shape=())
    output.fill(0)
    args = {"decision": decision, "predicate": predicate, "output": output}

    validation_calls = {"describe": 0, "owner": 0}
    original_describe = graph_impl._describe_parallel_storage
    original_validate_owner = graph_impl.validate_storage_owner

    def counted_describe(value):
        validation_calls["describe"] += 1
        return original_describe(value)

    def counted_validate_owner(description):
        validation_calls["owner"] += 1
        return original_validate_owner(description)

    monkeypatch.setattr(graph_impl, "_describe_parallel_storage", counted_describe)
    monkeypatch.setattr(graph_impl, "validate_storage_owner", counted_validate_owner)
    bindings = graph.bind(args)
    publish_validation_calls = dict(validation_calls)
    assert bindings.fast_path_qualified
    assert bindings.statistics()["control_publish_validated"]
    assert graph.binding_plan()["control_names"] == ("predicate",)
    assert graph.binding_plan()["control_publish_frame_stable"]

    decision.fill(1)
    graph.run(bindings)
    decision.fill(0)
    graph.run(bindings)
    assert output.to_numpy()[()] == 1
    stats = graph.binding_statistics()
    assert validation_calls == publish_validation_calls
    assert stats["control_publish_validations"] == 1
    assert stats["control_replay_validations"] == 0
    assert stats["version_fast_replays"] == 2
    assert stats["flattened_frame_builds"] == 1

    decision.fill(1)
    graph.run(args)
    assert output.to_numpy()[()] == 2
    assert validation_calls["describe"] > publish_validation_calls["describe"]
    assert validation_calls["owner"] > publish_validation_calls["owner"]
    stats = graph.binding_statistics()
    assert stats["control_publish_validations"] == 1
    assert stats["control_replay_validations"] == 1
    assert stats["raw_replay_validations"] == 1
