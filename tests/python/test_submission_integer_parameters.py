"""Typed per-submit integers retain ABI width and fixed replay ownership."""

from contextlib import ExitStack
import ctypes
import os

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import ExecutionPlan, Runtime, Scalar
from tests import test_utils


TYPES = dict(index="i32", left="i32", right="u32", epoch="i64", stamp="u64", count="i32")
NAMES = tuple(TYPES)
GUARD = 0x12345678
MASK = (1 << 64) - 1
ROWS = 16


@ti.kernel
def publish(out: ti.types.ndarray(ti.u64, 2), index: ti.i32, guard: ti.i32,
            left: ti.i32, right: ti.u32, epoch: ti.i64, stamp: ti.u64, count: ti.i32):
    out[index, 0] = ti.cast(left, ti.u64)
    out[index, 1] = guard
    out[index, 2] = right
    out[index, 3] = ti.cast(epoch, ti.u64)
    out[index, 4] = stamp
    out[index, 5] = count
    out[index, 6] = 0
    for j in range(count):
        out[index, 6] += ti.cast(j + 1, ti.u64)


def graph():
    builder = ti.graph.GraphBuilder()
    slots = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.u64, ndim=2)]
    for name in ("index", "guard", "left", "right", "epoch", "stamp", "count"):
        slots.append(ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, getattr(ti, TYPES.get(name, "i32"))))
    builder.dispatch(publish, *slots)
    return builder.compile()


def values(index, lane):
    return dict(index=index, left=(-(1 << 31), (1 << 31) - 1, -1, 0)[index % 4],
                right=((1 << 32) - 1 - lane - index),
                epoch=(-(1 << 63), (1 << 63) - 1, -(1 << 40) - lane, 0)[index % 4],
                stamp=MASK - index - lane, count=(0, 1, 7, 3)[index % 4])


def exercise(calls):
    expected = [np.zeros((ROWS, 7), dtype=np.uint64) for _ in calls]
    for index in range(ROWS):
        tickets = []
        for lane, call in enumerate(calls):
            mapping = values(index, lane)
            count = mapping["count"]
            expected[lane][index] = [mapping["left"] & MASK, GUARD, mapping["right"],
                                     mapping["epoch"] & MASK, mapping["stamp"], count,
                                     count * (count + 1) // 2]
            ticket = call.submit(values=mapping)
            assert ticket is not None
            mapping.update(dict.fromkeys(NAMES, 0))
            tickets.append(ticket)
        for ticket in tickets:
            ticket.wait()
            ticket.close()
    return expected


def reject_invalid(call):
    before = call.info["launches"]
    for name, invalid in (("left", -(1 << 31) - 1), ("left", 1 << 31),
                          ("right", -1), ("right", 1 << 32),
                          ("epoch", -(1 << 63) - 1), ("epoch", 1 << 63),
                          ("stamp", -1), ("stamp", 1 << 64),
                          ("count", True), ("count", 1.5)):
        with pytest.raises((ValueError, TypeError)):
            call.submit(values=dict(values(0, 0), **{name: invalid}))
    # Native consumers bypass Python range validation. Reject a noncanonical
    # i32 bit pattern before admission and preserve the next valid submission.
    from taichi_forge_runtime.aot import _c_api as abi
    call = getattr(call, "_prepared", call)
    runtime = call._runtime
    snapshot = call._submission_parameters.snapshot(values(0, 0))
    snapshot[NAMES.index("left")] |= 1 << 32
    raw = (ctypes.c_uint64 * len(snapshot))(*snapshot)
    completion = abi.Handle()
    with runtime._lock, pytest.raises(RuntimeError, match="zero upper 32 bits"):
        runtime._api.call(runtime._api.ti_try_submit_prepared_parameters_ext,
                          runtime._handle, call._handle, 0, None, ctypes.byref(completion),
                          False, len(raw), raw)
    assert not completion.value
    assert call.info["launches"] == before


def test_submission_integer_schema():
    from taichi_forge_runtime.aot import _c_api as abi
    from taichi_forge_runtime.aot._parameters import SubmissionParameters

    for dtype, lower, upper in (("i32", -(1 << 31), (1 << 31) - 1),
                               ("u32", 0, (1 << 32) - 1),
                               ("i64", -(1 << 63), (1 << 63) - 1),
                               ("u64", 0, MASK)):
        parameters = SubmissionParameters(("value",))
        parameters.configure([abi.NamedArgument(b"value", Scalar(dtype, 0)._argument())])
        mapping = {"value": lower}
        frozen = parameters.snapshot(mapping)
        mapping["value"] = upper
        assert list(parameters.bind(frozen)) == [lower & ((1 << int(dtype[1:])) - 1)]
        assert parameters.snapshot(mapping) == [upper]
        for invalid in (lower - 1, upper + 1, True, 1.0):
            with pytest.raises((ValueError, TypeError)):
                parameters.snapshot({"value": invalid})
    parameters = SubmissionParameters(("value",))
    for dtype in ("f32", "i16", "u8"):
        with pytest.raises(ValueError):
            parameters.configure([abi.NamedArgument(b"value", Scalar(dtype, 0)._argument())])
    with pytest.raises(ValueError):
        parameters.configure([])


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_jit_mixed_integer_submission():
    from taichi_forge_runtime import runtime_contract_manifest
    backend = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "vulkan"
    assert runtime_contract_manifest()["features"][backend + "_submission_parameters_integer"]
    source = graph()
    with ExitStack() as scope:
        outputs = [ti.ndarray(ti.u64, (ROWS, 7)) for _ in range(2)]
        calls = []
        for lane, out in enumerate(outputs):
            domain = scope.enter_context(ti.graph.execution_domain(capacity=2, instance_capacity=1))
            call = scope.enter_context(source.prepare(dict(out=out, guard=GUARD, **values(0, lane)),
                execution_domain=domain, mutable_parameters=NAMES))
            call.initialize()
            calls.append(call)
        source.close()
        reject_invalid(calls[0])
        before = [call.info.copy() for call in calls]
        for out, expected in zip(outputs, exercise(calls)):
            np.testing.assert_array_equal(out.to_numpy(), expected)
        for call, original in zip(calls, before):
            assert call.info["argument_bytes"] == original["argument_bytes"]
            assert call.info["workspace_bytes"] == original["workspace_bytes"]


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_mixed_integer_submission(tmp_path):
    vulkan = ti.lang.impl.current_cfg().arch == ti.vulkan
    artifact = ti.aot.Module(caps=["spirv_has_int64", "spirv_has_atomic_int64"] if vulkan else [])
    artifact.add_kernel(publish)
    artifact.add_execution_plan("publish", ExecutionPlan([
        dict(kind="kernel", entry="publish", arguments=["out", "index", "guard", "left", "right", "epoch", "stamp", "count"])]))
    artifact.save(tmp_path / "module")
    with Runtime("cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "vulkan",
                 device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))) as runtime:
        module = runtime.load_module(tmp_path / "module")
        with ExitStack() as scope:
            outputs = [runtime.ndarray("u64", (ROWS, 7)) for _ in range(2)]
            calls = []
            for lane, out in enumerate(outputs):
                domain = scope.enter_context(runtime.execution_domain(capacity=2, instance_capacity=1))
                args = dict(out=out, guard=GUARD, **{name: Scalar(TYPES[name], value)
                            for name, value in values(0, lane).items()})
                # Reject unsupported/missing types before any native instance is published.
                with pytest.raises(ValueError):
                    module.prepare_execution("publish", dict(args, left=Scalar("f32", 1)),
                                             execution_domain=domain, mutable_parameters=NAMES)
                call = scope.enter_context(module.prepare_execution("publish", args,
                    execution_domain=domain, mutable_parameters=NAMES))
                call.initialize()
                calls.append(call)
            reject_invalid(calls[0])
            for out, expected in zip(outputs, exercise(calls)):
                np.testing.assert_array_equal(np.frombuffer(out.read(), dtype=np.uint64).reshape(ROWS, 7), expected)
