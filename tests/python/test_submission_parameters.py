"""Per-submit metadata uses the selected queue and independent argument storage."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from taichi_forge_runtime.aot import ExecutionPlan, Runtime, Scalar


NAMES = ("epoch", "simulation_step", "timestamp_ns")


def test_float_submission_preserves_scalar_bits():
    import struct
    from taichi_forge_runtime.aot import _c_api as abi
    from taichi_forge_runtime.aot._parameters import SubmissionParameters

    for dtype, fmt in (("f32", "<f"), ("f64", "<d")):
        params = SubmissionParameters(("alpha",))
        params.configure([abi.NamedArgument(b"alpha", Scalar(dtype, 0)._argument())])
        for value in (0.0, -0.0, 0.1, -1.5, float("inf"), float("nan")):
            values = {"alpha": value}
            bits = params.snapshot(values)
            values["alpha"] = 7.0
            assert bits == [int.from_bytes(struct.pack(fmt, value), "little")]
            assert list(params.bind(bits)) == bits
        for invalid in (True, "0.5", b"0.5", None):
            with pytest.raises((ValueError, TypeError)):
                params.snapshot({"alpha": invalid})


@ti.kernel
def publish_floats(out: ti.types.ndarray(ti.u64, 2), index: ti.i32,
                   alpha: ti.f32, guard: ti.i32, weight: ti.f64):
    out[index, 0] = ti.cast(ti.bit_cast(alpha, ti.u32), ti.u64)
    out[index, 1] = ti.cast(guard, ti.u64)
    out[index, 2] = ti.bit_cast(weight, ti.u64)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_jit_aot_float_submission_abi_and_instances(tmp_path):
    import struct
    from contextlib import ExitStack
    from taichi_forge_runtime import runtime_contract_manifest

    assert runtime_contract_manifest()["features"]["cuda_submission_parameters_float"]
    builder = ti.graph.GraphBuilder()
    names = ("out", "index", "alpha", "guard", "weight")
    builder.dispatch(publish_floats,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.u64, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "index", ti.i32),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "alpha", ti.f32),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "guard", ti.i32),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "weight", ti.f64))
    source = builder.compile()
    artifact = ti.aot.Module()
    artifact.add_kernel(publish_floats)
    artifact.add_execution_plan("publish", ExecutionPlan([
        dict(kind="kernel", entry="publish_floats", arguments=list(names))]))
    artifact.save(tmp_path / "float")
    mutable = ("index", "alpha", "weight")
    reference = ti.ndarray(ti.u64, (16, 3))
    expected = [np.zeros((16, 3), np.uint64) for _ in range(2)]
    for lane in range(2):
        for index in range(16):
            alpha = (-0.0, 0.125, -1.5, 32.75)[index % 4]
            weight = index / 7.0 + lane
            source.run(dict(out=reference, index=index, alpha=alpha, guard=12345, weight=weight))
            expected[lane][index] = (int.from_bytes(struct.pack('<f', alpha), 'little'),
                                    12345, int.from_bytes(struct.pack('<d', weight), 'little'))
        np.testing.assert_array_equal(reference.to_numpy(), expected[lane])
    with ExitStack() as scope:
        runtime = scope.enter_context(Runtime("cuda"))
        module = scope.enter_context(runtime.load_module(tmp_path / "float"))
        aot_domain = scope.enter_context(runtime.execution_domain(capacity=2))
        aot_out = runtime.ndarray("u64", (16, 3))
        jit_domain = scope.enter_context(ti.graph.execution_domain(capacity=2))
        jit_out = ti.ndarray(ti.u64, (16, 3))
        jit = scope.enter_context(source.prepare(dict(out=jit_out, index=0, alpha=0., guard=12345, weight=0.),
            execution_domain=jit_domain, mutable_parameters=mutable))
        aot = scope.enter_context(module.prepare_execution("publish", dict(out=aot_out, index=0,
            alpha=Scalar("f32", 0), guard=12345, weight=Scalar("f64", 0)),
            execution_domain=aot_domain, mutable_parameters=mutable))
        for call in (jit, aot):
            call.initialize()
        initial = [call.info.copy() for call in (jit, aot)]
        for index in range(16):
            tickets=[]
            for lane, call in enumerate((jit, aot)):
                values=dict(index=index, alpha=(-0.0, .125, -1.5, 32.75)[index % 4], weight=index/7.+lane)
                ticket=call.submit(values=values)
                assert ticket is not None
                values.update(index=0, alpha=999., weight=999.)
                tickets.append(ticket)
            for ticket in tickets:
                ticket.wait()
                ticket.close()
        np.testing.assert_array_equal(jit_out.to_numpy(), expected[0])
        np.testing.assert_array_equal(np.frombuffer(aot_out.read(),np.uint64).reshape(16,3), expected[1])
        for call, before in zip((jit,aot),initial):
            assert call.info['argument_bytes'] == before['argument_bytes']
            assert call.info['workspace_bytes'] == before['workspace_bytes']
        # Replay-only is still required; floating initialization inputs cannot mutate.
        plan=ExecutionPlan([dict(kind='kernel',entry='publish_floats',arguments=list(names))],
                           initialization=[dict(kind='kernel',entry='publish_floats',arguments=list(names))])
        with pytest.raises((ValueError,RuntimeError), match='replay-only'):
            module.prepare_execution(plan, dict(out=aot_out,index=0,alpha=Scalar('f32',0),guard=12345,
                weight=Scalar('f64',0)),execution_domain=aot_domain,mutable_parameters=mutable)


@ti.kernel
def publish(out: ti.types.ndarray(ti.u64, 2), epoch: ti.u64,
            simulation_step: ti.u64, timestamp_ns: ti.u64):
    index = ti.cast(epoch & ti.u64(1023), ti.i32)
    out[index, 0] = epoch
    out[index, 1] = simulation_step
    out[index, 2] = timestamp_ns


def graph():
    builder = ti.graph.GraphBuilder()
    builder.dispatch(publish, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.u64, ndim=2),
                     *(ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, ti.u64) for name in NAMES))
    return builder.compile()


def values(index, lane):
    return dict(zip(NAMES, ((1 << 63) + (lane << 40) + index,
                           (1 << 62) + index * 17 + lane,
                           (1 << 64) - 1 - index * 31 - lane)))


def exercise(calls):
    expected = [np.zeros((1000, 3), dtype=np.uint64) for _ in calls]
    for index in range(1000):
        tickets = []
        for lane, call in enumerate(calls):
            mapping = values(index, lane)
            expected[lane][index] = tuple(mapping.values())
            ticket = call.submit(values=mapping)
            assert ticket is not None
            mapping.update(dict.fromkeys(NAMES, 0))
            tickets.append(ticket)
        for ticket in tickets:
            ticket.wait()
            ticket.close()
    return expected


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_jit_submission_u64_interleaving():
    source = graph()
    outputs = [ti.ndarray(ti.u64, (1000, 3)) for _ in range(2)]
    domains = [ti.graph.execution_domain(capacity=2) for _ in range(2)]
    calls = [source.prepare(dict(out=out, **values(0, lane)), execution_domain=domain,
                            mutable_parameters=NAMES)
             for lane, (out, domain) in enumerate(zip(outputs, domains))]
    for call in calls:
        call.initialize()
    source.close()
    for out, expected in zip(outputs, exercise(calls)):
        np.testing.assert_array_equal(out.to_numpy(), expected)
    for invalid in (None, {}, dict(values(0, 0), extra=1), dict(values(0, 0), epoch=-1),
                    dict(values(0, 0), epoch=1 << 64), dict(values(0, 0), epoch=True)):
        with pytest.raises((ValueError, TypeError)):
            calls[0].submit(values=invalid)
    for call in calls:
        call.close()
    for domain in domains:
        domain.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_submission_u64_and_cold_rejection(tmp_path):
    artifact = ti.aot.Module()
    artifact.add_kernel(publish)
    artifact.add_execution_plan("publish", ExecutionPlan([
        dict(kind="kernel", entry="publish", arguments=["out", *NAMES])]))
    artifact.save(tmp_path / "module")
    with Runtime("cuda") as runtime:
        module = runtime.load_module(tmp_path / "module")
        domains = [runtime.execution_domain(capacity=2) for _ in range(2)]
        outputs = [runtime.ndarray("u64", (1000, 3)) for _ in domains]
        arguments = [dict(out=out, **{name: Scalar("u64", value) for name, value in values(0, lane).items()})
                     for lane, out in enumerate(outputs)]
        for names in (("out",), ("missing",), ("epoch", "epoch")):
            with pytest.raises((ValueError, RuntimeError)):
                module.prepare_execution("publish", arguments[0], execution_domain=domains[0],
                                         mutable_parameters=names)
        calls = [module.prepare_execution("publish", args, execution_domain=domain,
                                          mutable_parameters=NAMES)
                 for args, domain in zip(arguments, domains)]
        for call in calls:
            call.initialize()
        for out, expected in zip(outputs, exercise(calls)):
            np.testing.assert_array_equal(np.frombuffer(out.read(), dtype=np.uint64).reshape(1000, 3), expected)
        for call in calls:
            call.close()
        for domain in domains:
            domain.close()
