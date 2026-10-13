"""Fixed sparse weights, shared compression scratch and explicit adjoints."""

from taichi_forge_runtime.aot import ExecutionPlan
from tests.python.aot_cusparselt_probe import (
    M,
    N,
    K,
    check,
    compress,
    inputs,
    multiply,
    plan_resource,
    read,
    reference,
    write,
)
from tests.python.aot_cutensor_probe import raises


def snapshot_plans():
    forward = [
        dict(kind="fill32", destination="loss", value=0),
        multiply("b", "c", "d", alpha=0.75, beta=0.25),
        multiply("b", "c", "e", alpha=-0.5, beta=0.5),
        dict(kind="kernel", entry="energy", arguments=["d", "e", "loss"]),
    ]
    return {
        "fixed_a": ExecutionPlan(forward, initialization=[compress("a")], resources=[plan_resource()]),
        "fixed_a_adjoint": ExecutionPlan(
            forward + [dict(kind="kernel", entry="adjoint", arguments=["saved_a", "b", "d", "e", "da", "db"])],
            initialization=[dict(kind="copy", source="a", destination="saved_a"), compress("saved_a")],
            resources=[plan_resource(), dict(kind="ndarray", name="saved_a", dtype="f16", shape=[M, K])],
        ),
        "init_and_refresh_a": ExecutionPlan(
            [compress("a")] + forward,
            initialization=[compress("initial_a")],
            resources=[plan_resource()],
        ),
    }


def check_snapshots(runtime, module):
    def frame(name, factor):
        plan = module.execution_plan(name)
        shapes = dict(
            a=(M, K), initial_a=(M, K), b=(N, K), c=(M, N), d=(M, N), e=(M, N), da=(M, K), db=(N, K), loss=(1,)
        )
        arrays = {
            key: runtime.ndarray("f32" if key in ("da", "db", "loss") else "f16", shapes[key])
            for key in plan.argument_names
        }
        values = inputs(factor)
        for key, data in zip("abc", values):
            write(arrays[key], data)
        if "initial_a" in arrays:
            write(arrays["initial_a"], inputs(2)[0])
        write(arrays["d"], [-77] * (M * N))
        prepared = plan.prepare(module, arrays)
        assert read(arrays["d"]) == [-77] * (M * N)
        return arrays, prepared

    def verify(arrays, values):
        expected = reference(*values)
        check(arrays, {key: value for key, value in expected.items() if key in arrays})

    left, one = frame("fixed_a", 1)
    right, two = frame("fixed_a", -0.5)
    before = one.info["workspace_bytes"]
    sampled = inputs(0.5, shift=2)
    write(left["a"], sampled[0])
    one.initialize()
    after = one.info["workspace_bytes"]
    assert 0 < after <= before
    assert read(left["d"]) == [-77] * (M * N)
    write(left["a"], inputs(3)[0])
    one.initialize()
    left["a"].close()
    for _ in range(3):
        one()
        two()
    verify(left, sampled)
    verify(right, inputs(-0.5))
    assert two.info["workspace_bytes"] == after
    changed_b = [value * 0.5 for value in sampled[1]]
    write(left["b"], changed_b)
    one()
    verify(left, (sampled[0], changed_b, sampled[2]))
    write(left["b"], sampled[1])
    # Re-preparing samples current values, independently from the old plan.
    write(right["a"], inputs(2)[0])
    replacement = module.prepare_execution("fixed_a", right)
    replacement()
    verify(right, inputs(2))
    replacement.close()
    two()
    verify(right, inputs(-0.5))

    gradients, differentiated = frame("fixed_a_adjoint", 1)
    differentiated.initialize()
    assert differentiated.info["resource_bytes"] == M * K * 2
    assert differentiated.info["workspace_bytes"] == after
    gradients["a"].close()
    differentiated()
    verify(gradients, inputs(1))

    refresh, dynamic = frame("init_and_refresh_a", 1)
    assert dynamic.info["workspace_bytes"] == before
    dynamic.initialize()
    # A replay compression still needs the shared scratch lease.
    assert dynamic.info["workspace_bytes"] == before
    refresh["initial_a"].close()
    dynamic()
    verify(refresh, inputs(1))
    write(refresh["a"], sampled[0])
    dynamic()
    verify(refresh, sampled)
    dynamic()
    refresh["a"].close()
    raises(dynamic)
    one()
    verify(left, sampled)
    # A failed read-before-compression plan cannot damage existing snapshots.
    invalid = ExecutionPlan([multiply("b", "c", "d")], resources=[plan_resource()])
    raises(lambda: invalid.prepare(module, {key: right[key] for key in "bcd"}))
    one()
    verify(left, sampled)
    one()  # Close must wait for this snapshot replay too.
    for prepared in (one, two, differentiated, dynamic):
        prepared.close()
    for arrays in (left, right, gradients, refresh):
        for value in arrays.values():
            value.close()
    tail_arrays, tail = frame("fixed_a", 1)
    tail.initialize()
    tail_arrays["a"].close()
    return tail_arrays, tail
