import gc
import threading
import time

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._submission import _new_submission_lane
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils


class _BlockingCompletion:
    has_backend_work = True

    def __init__(self):
        self._ready = threading.Event()

    def done(self):
        return self._ready.is_set()

    def wait(self):
        if not self._ready.wait(2.0):
            raise RuntimeError("blocking completion timed out")

    def complete(self):
        self._ready.set()


def _wait_until(predicate):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("condition did not become true")


def _scalar_fill_graph():
    @ti.kernel
    def fill(value: ti.i32, out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        out[0] = value

    value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, value, out)
    return builder.compile()


@test_utils.test(arch=ti.cpu)
def test_graph_binding_set_reuses_preflattened_frame_and_keeps_device_data_dynamic():
    @ti.kernel
    def copy_scaled(
        scale: ti.i32,
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        out: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        out[0] = scale * source[0]

    scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.i32)
    source_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1)
    out_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(copy_scaled, scale, source_arg, out_arg)
    graph = builder.compile()
    source = ti.ndarray(ti.i32, shape=1)
    out = ti.ndarray(ti.i32, shape=1)
    source.fill(3)

    bindings = graph.bind({"scale": 2, "source": source, "out": out})
    assert bindings.fast_path_qualified
    assert graph.binding_plan()["slot_order"] == ("out", "scale", "source")

    graph.run(bindings)
    assert out.to_numpy()[0] == 6
    source.fill(5)
    graph.run(bindings)
    assert out.to_numpy()[0] == 10

    before_update = graph.binding_statistics()
    assert before_update["version_fast_replays"] == 2
    assert before_update["raw_replay_validations"] == 0
    assert before_update["flattened_frame_builds"] == 1

    bindings.update(scale=4)
    graph.run(bindings)
    assert out.to_numpy()[0] == 20
    after_update = graph.binding_statistics()
    assert after_update["version_fast_replays"] == 3
    assert after_update["flattened_frame_builds"] == 2

    graph.run({"scale": 7, "source": source, "out": out})
    assert out.to_numpy()[0] == 35
    assert graph.binding_statistics()["raw_replay_validations"] == 1


@test_utils.test(arch=ti.cpu)
def test_graph_binding_set_snapshots_matrix_values_and_updates_atomically():
    matrix_type = ti.types.matrix(2, 2, ti.i32)

    @ti.kernel
    def consume(
        left: ti.i32,
        right: ti.i32,
        value: matrix_type,
        out: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        out[0] = left - right
        out[1] = value[0, 0]

    left = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "left", ti.i32)
    right = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "right", ti.i32)
    value = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "value", matrix_type)
    out_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(consume, left, right, value, out_arg)
    graph = builder.compile()
    out = ti.ndarray(ti.i32, shape=2)
    matrix = ti.Matrix([[3, 0], [0, 0]], ti.i32)
    bindings = graph.bind({"left": 0, "right": 0, "value": matrix, "out": out})

    matrix[0, 0] = 9
    graph.run(bindings)
    np.testing.assert_array_equal(out.to_numpy(), np.array([0, 3], np.int32))
    diagnostic = bindings.snapshot()
    diagnostic["value"][0, 0] = 17
    graph.run(bindings)
    np.testing.assert_array_equal(out.to_numpy(), np.array([0, 3], np.int32))

    errors = []

    def update_pairs():
        try:
            for index in range(1, 80):
                bindings.update({"left": index, "right": index})
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=update_pairs)
    worker.start()
    while worker.is_alive():
        graph.run(bindings)
        assert out.to_numpy()[0] == 0
    worker.join(2.0)
    assert not errors
    assert bindings.revision == 80

    bindings.update(value=matrix)
    graph.run(bindings)
    np.testing.assert_array_equal(out.to_numpy(), np.array([0, 9], np.int32))


@test_utils.test(arch=ti.cpu)
def test_graph_binding_set_samples_after_pacer_wait():
    graph = _scalar_fill_graph()
    out = ti.ndarray(ti.i32, shape=1)
    bindings = graph.bind({"value": 1, "out": out})
    pacer = ti.graph.SubmissionPacer(1)
    blocker = pacer._reserve(
        impl.pytaichi,
        _new_submission_lane("binding-set-test"),
        lane="blocker",
        on_saturation="wait",
    )
    blocker_completion = _BlockingCompletion()
    blocker._attach(blocker_completion)
    errors = []

    def submit():
        try:
            graph.submit(bindings, pacer=pacer, lane="graph").wait()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=submit)
    worker.start()
    _wait_until(lambda: pacer.statistics()["queued"] == 1)
    bindings.update(value=23)
    blocker_completion.complete()
    worker.join(2.0)
    assert not worker.is_alive()
    assert not errors
    assert out.to_numpy()[0] == 23
    assert graph.binding_statistics()["version_fast_replays"] == 1


@test_utils.test(arch=ti.cpu)
def test_graph_binding_set_rejects_alias_update_without_publishing_it():
    @ti.kernel
    def stop(predicate: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        predicate[None] = 0

    @ti.kernel
    def step(counter: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        counter[None] += 1

    def scalar(name):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)

    outer_predicate_arg = scalar("outer_predicate")
    outer_counter_arg = scalar("outer_counter")
    inner_predicate_arg = scalar("inner_predicate")
    inner_counter_arg = scalar("inner_counter")
    builder = ti.graph.GraphBuilder()
    outer_condition = builder.create_sequential()
    outer_condition.dispatch(stop, outer_predicate_arg)
    inner_condition = builder.create_sequential()
    inner_condition.dispatch(stop, inner_predicate_arg)
    inner_body = builder.create_sequential()
    inner_body.dispatch(step, inner_counter_arg)
    outer_body = builder.create_sequential()
    outer_body.while_loop(
        inner_condition,
        inner_body,
        predicate=inner_predicate_arg,
        counter=inner_counter_arg,
        max_iterations=2,
        name="inner",
    )
    outer_body.dispatch(step, outer_counter_arg)
    builder.while_loop(
        outer_condition,
        outer_body,
        predicate=outer_predicate_arg,
        counter=outer_counter_arg,
        max_iterations=2,
        name="outer",
    )
    graph = builder.compile()
    outer_predicate = ti.ndarray(ti.i32, shape=())
    bindings = graph.bind(
        {
            "outer_predicate": outer_predicate,
            "outer_counter": ti.ndarray(ti.i32, shape=()),
            "inner_predicate": ti.ndarray(ti.i32, shape=()),
            "inner_counter": ti.ndarray(ti.i32, shape=()),
        }
    )
    assert bindings.fast_path_qualified
    revision = bindings.revision

    with pytest.raises(TaichiRuntimeError, match="must not alias"):
        bindings.update(inner_predicate=outer_predicate)

    assert bindings.revision == revision
    graph.run(bindings)
    stats = graph.binding_statistics()
    assert stats["version_volatile_replays"] == 0
    assert stats["version_fast_replays"] == 1
    assert stats["control_publish_validations"] == 2
    assert stats["control_replay_validations"] == 0


@test_utils.test(arch=ti.cpu)
def test_graph_binding_set_reclaims_versions_without_a_hard_inflight_cap():
    graph = _scalar_fill_graph()
    out = ti.ndarray(ti.i32, shape=1)
    bindings = graph.bind({"value": 1, "out": out})
    tickets = []
    for value in range(1, 72):
        tickets.append(graph.submit(bindings))
        bindings.update(value=value + 1)

    assert bindings.statistics()["live_retired_versions"] == len(tickets)
    high_water_retired = bindings._retired_versions

    tickets.clear()
    gc.collect()
    assert bindings.statistics()["live_retired_versions"] == 0
    assert bindings._retired_versions == {}
    assert bindings._retired_versions is not high_water_retired
    bindings.update(value=73)
    assert bindings.statistics()["live_retired_versions"] == 0
    graph.run(bindings)
    assert out.to_numpy()[0] == 73


@test_utils.test(arch=ti.cpu)
def test_graph_binding_set_waiting_submission_is_rejected_by_runtime_reset():
    graph = _scalar_fill_graph()
    out = ti.ndarray(ti.i32, shape=1)
    bindings = graph.bind({"value": 5, "out": out})
    pacer = ti.graph.SubmissionPacer(1)
    blocker = pacer._reserve(
        impl.pytaichi,
        _new_submission_lane("binding-set-reset-test"),
        lane="blocker",
        on_saturation="wait",
    )
    blocker._attach(_BlockingCompletion())
    errors = []

    def submit():
        try:
            graph.submit(bindings, pacer=pacer, lane="graph")
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=submit)
    worker.start()
    _wait_until(lambda: pacer.statistics()["queued"] == 1)
    ti.reset()
    worker.join(2.0)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], TaichiRuntimeError)
    with pytest.raises(TaichiRuntimeError, match="runtime reinitialization"):
        bindings.update(value=6)
