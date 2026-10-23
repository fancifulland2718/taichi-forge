"""Runtime-only Vulkan domains: exact parameters, timing, AD and retirement."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge._lib import utils as runtime_utils
from tests import test_utils


CONSUMER = r'''
import array
import os
import sys

class NoFrontend:
    def find_spec(self, fullname, *args):
        if fullname == "taichi_forge" or fullname.startswith("taichi_forge."):
            raise AssertionError("AOT consumer imported the compiler: " + fullname)

sys.meta_path.insert(0, NoFrontend())
from taichi_forge_runtime.aot import (
    CompletionSet, ExecutionDomain, Runtime, Scalar, VulkanExecutionDomain,
)

def read(value, code):
    result = array.array(code)
    result.frombytes(value.read())
    return list(result)

names = ("epoch", "simulation_step", "timestamp_ns")
kept = []
with Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))) as runtime:
    module = runtime.load_module(sys.argv[1])
    domains = []
    try:
        for _ in range(2):
            domains.append(runtime.execution_domain(capacity=2, instance_capacity=1, gpu_timing=True))
    except RuntimeError as error:
        # Hardware qualification only; do not suppress preparation/execution failures.
        if "No independent Vulkan compute queue is available" not in str(error) and "timeline support" not in str(error):
            raise
        print("VULKAN_DOMAIN_HARDWARE_UNAVAILABLE:", error)
        sys.exit(77)
    assert all(isinstance(d, (ExecutionDomain, VulkanExecutionDomain)) for d in domains)
    assert domains[0].info["queue_index"] != domains[1].info["queue_index"]
    outputs = [runtime.ndarray("u64", (1000, 3)) for _ in domains]
    arguments = [dict(out=out, **dict.fromkeys(names, Scalar("u64", 0))) for out in outputs]
    calls = [module.prepare_execution("publish", args, execution_domain=domain,
                                     mutable_parameters=names)
             for args, domain in zip(arguments, domains)]
    for call in calls:
        call.initialize()
    expected = [[], []]
    for index in range(1000):
        tickets = []
        for lane, call in enumerate(calls):
            values = dict(zip(names, ((1 << 63) + (lane << 40) + index,
                                     (1 << 62) + index * 17 + lane,
                                     (1 << 64) - 1 - index * 31 - lane)))
            expected[lane].extend(values.values())
            ticket = call.submit(values=values, sample_gpu=index in (0, 999))
            assert ticket is not None
            values.update(dict.fromkeys(names, 0))
            tickets.append(ticket)
        if index == 0:
            with CompletionSet(tickets) as completions:
                assert completions.wait(timeout=5).status == "ready"
        for ticket in tickets:
            ticket.wait()
            timing = ticket.gpu_timing
            assert timing["owner"] == ticket.info["owner"]
            assert timing["sequence"] == ticket.info["sequence"]
            assert timing["scope"] == "compute_envelope"
            if index in (0, 999):
                assert timing["status"] == "ready" and timing["duration_ns"] > 0
                kept.append((ticket, timing))
            else:
                assert timing["status"] == "not_sampled" and timing["duration_ns"] is None
                ticket.close()
    for output, want in zip(outputs, expected):
        assert read(output, "Q") == want
    calls[0].close()
    # The closed instance's slot can prepare a different recipe; the peer stays live.
    x, y, dx, dy = [runtime.ndarray("f32", 17) for _ in range(4)]
    x.write(array.array("f", [3] * 17))
    adjoint = module.prepare_execution("square", dict(x=x, y=y, dx=dx, dy=dy),
                                      gradients=dict(x=dx, y=dy), execution_domain=domains[0])
    adjoint.initialize()
    ready = calls[1].submit(values=dict(zip(names, (1 << 63, 7, 9))))
    ready.wait()
    done = adjoint.submit(wait_for=(ready,), ready_only=True, sample_gpu=True)
    assert done is not None
    done.wait()
    assert read(y, "f") == [9] * 17
    assert read(dx, "f") == [12] * 17
    done.close()
    ready.close()
    adjoint.close()
    calls[1].close()
    for domain in domains:
        domain.close()
for ticket, timing in kept:
    assert ticket.done() and ticket.gpu_timing == timing
    ticket.close()
assert not any(name == "taichi_forge" or name.startswith("taichi_forge.") for name in sys.modules)
print("VULKAN_DOMAIN_SOURCE_FREE_OK")
'''


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_source_free_vulkan_domain_parameters_timing_and_ad(tmp_path):
    @ti.kernel
    def publish(out: ti.types.ndarray(ti.u64, 2), epoch: ti.u64,
                simulation_step: ti.u64, timestamp_ns: ti.u64):
        index = ti.cast(epoch & ti.u64(1023), ti.i32)
        out[index, 0] = epoch
        out[index, 1] = simulation_step
        out[index, 2] = timestamp_ns

    @ti.kernel
    def square(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            y[i] = x[i] * x[i]

    artifact = ti.aot.Module(caps=["spirv_has_int64"])
    artifact.add_kernel(publish)
    artifact.add_kernel(square)
    artifact.add_kernel(square.grad, name="backward")
    artifact.add_execution_plan("publish", ti.aot.ExecutionPlan([
        dict(kind="kernel", entry="publish", arguments=["out", "epoch", "simulation_step", "timestamp_ns"])]))
    artifact.add_execution_plan("square", ti.aot.ExecutionPlan([
        dict(kind="fill32", destination="dx", value=0),
        dict(kind="fill32", destination="dy", value=1073741824),
        dict(kind="kernel", entry="square", arguments=["x", "y"]),
        dict(kind="kernel", entry="backward", arguments=["x", "y"])]))
    artifact.save(tmp_path / "module")
    native = runtime_utils._loaded_native_runtime_path
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(sys.path), TAICHI_NATIVE_RUNTIME_DIR=str(Path(native).parent),
               TAICHI_RUNTIME_DIR=runtime_utils._runtime_bitcode_dir(), TI_ENABLE_VERSION_CHECK="0")
    result = subprocess.run([sys.executable, "-S", "-c", CONSUMER, str(tmp_path / "module")],
                            env=env, capture_output=True, text=True, timeout=60)
    if result.returncode == 77 and "VULKAN_DOMAIN_HARDWARE_UNAVAILABLE:" in result.stdout:
        pytest.skip(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VUID-" not in result.stderr and "SYNC-HAZARD" not in result.stderr, result.stderr
    assert "VULKAN_DOMAIN_SOURCE_FREE_OK" in result.stdout
