import pytest

import taichi_forge as ti
from taichi_forge.graph._graph import _GraphStructuredTelemetryRecorder
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_summary_telemetry_preserves_single_element_control_ranks():
    @ti.kernel
    def evaluate(
        p: ti.types.ndarray(ti.i32, 1), c: ti.types.ndarray(ti.i32, 2), target: ti.i32
    ):
        p[0] = int(c[0, 0] < target)

    @ti.kernel
    def step(p: ti.types.ndarray(ti.i32, 1), c: ti.types.ndarray(ti.i32, 2)):
        if p[0] != 0:
            c[0, 0] += 1

    predicate = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=1)
    counter = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "counter", ti.i32, ndim=2)
    target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)
    builder = ti.graph.GraphBuilder()
    condition = builder.create_sequential()
    condition.dispatch(evaluate, predicate, counter, target)
    body = builder.create_sequential()
    body.dispatch(step, predicate, counter)
    builder.while_loop(
        condition,
        body,
        predicate=predicate,
        counter=counter,
        control_inputs=(counter, target),
        carried_state=(counter,),
        max_iterations=4,
        chunk_size=2,
        masked_execution=True,
        name="ranked_controls",
    )
    graph = builder.compile()
    p, c = ti.ndarray(ti.i32, (1,)), ti.ndarray(ti.i32, (1, 1))
    graph.prepare_telemetry("summary")
    for limit, iterations in ((3, 3), (0, 0), (7, 4)):
        p.fill(0)
        c.fill(0)
        ticket = graph.submit(
            dict(predicate=p, counter=c, target=limit), telemetry="summary"
        )
        report = ticket.telemetry()
        assert c.to_numpy()[0, 0] == iterations
        assert p.to_numpy()[0] == int(iterations < limit)
        assert report.regions[0].logical_iterations == iterations
        assert report.host_readback_bytes == 24
        assert report.gpu_timestamp_status == "disabled_by_mode"
    graph.close()


@test_utils.test(arch=ti.cpu)
def test_telemetry_rejects_non_scalar_or_multiple_control_values():
    class Context:
        def __init__(self, value):
            self.value = value

        def runtime_args(self):
            return dict(value=self.value)

    for value in (
        ti.ndarray(ti.i32, (2,)),
        ti.ndarray(ti.f32, (1,)),
        ti.Vector.ndarray(1, ti.i32, ()),
    ):
        with pytest.raises(TaichiRuntimeError, match="single-element scalar i32"):
            _GraphStructuredTelemetryRecorder._control_value(
                Context(value), "value", "predicate"
            )
