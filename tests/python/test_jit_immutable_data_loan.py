"""JIT initialization bytes use the AOT integrity and private-resource contract."""

import hashlib
import struct

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import ExecutionPlan, Scalar
from taichi_forge_runtime.aot._runtime import Module
from tests import test_utils
from tests.python.test_jit_execution_code_loan import _graph


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_immutable_jit_payloads_freeze_and_initialize_independent_instances():
    @ti.kernel
    def update(state: ti.types.ndarray(ti.u32, 1), output: ti.types.ndarray(ti.u32, 1), delta: ti.u32):
        for i in state:
            state[i] += delta
            output[i] = state[i]

    graph, selected = _graph(
        update,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state", ti.u32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.u32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "delta", ti.u32),
    )
    values = np.array([0, 7, 0x80000000, 0xFFFFFF00], np.uint32)
    payload = struct.pack("<4I", *values)
    key = hashlib.sha256(payload).hexdigest()
    imports = [("update", selected, 0, False)]
    plan = ExecutionPlan(
        [dict(kind="kernel", entry="update", arguments=["state", "output", "delta"])],
        resources=[dict(name="state", kind="ndarray", dtype="u32", shape=[4], initial_data=key)],
    )
    with ti.aot.borrow_runtime() as runtime:
        for invalid in ({"bad": payload}, {key: bytes(len(payload))}):
            with pytest.raises((ValueError, RuntimeError), match="immutable JIT data"):
                runtime._borrower.import_jit_kernels(imports, invalid)
        data = {key: payload}
        with Module(runtime, runtime._borrower.import_jit_kernels(imports, data)) as module:
            data.clear()
            graph.close()
            with runtime.execution_domain(instance_capacity=2) as domain:
                outputs = [runtime.ndarray("u32", 4) for _ in range(2)]
                calls = [
                    module.prepare_execution(plan, dict(output=out, delta=Scalar("u32", lane + 1)), execution_domain=domain)
                    for lane, out in enumerate(outputs)
                ]
                for call in calls:
                    call.initialize()
                for repeat in range(1, 4):
                    for lane, call in enumerate(calls):
                        ticket = call.submit()
                        ticket.wait()
                        ticket.close()
                        np.testing.assert_array_equal(
                            np.frombuffer(outputs[lane].read(), np.uint32), values + repeat * (lane + 1)
                        )
                calls[0].close()
                calls[1].submit().wait()
                np.testing.assert_array_equal(np.frombuffer(outputs[1].read(), np.uint32), values + 8)
                calls[1].close()
                for output in outputs:
                    output.close()
