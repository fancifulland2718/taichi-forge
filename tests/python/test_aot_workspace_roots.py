"""Workspace leases isolate results, while Field storage belongs to a module."""

import struct

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import ExecutionPlan, Runtime, WorkspaceBusyError
from tests import test_utils
from tests.python.test_aot_storage_groups import _square


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_workspace_roots_keep_module_identity_and_owned_results(tmp_path):
    state = ti.field(ti.i32, shape=1)

    @ti.kernel
    def advance(
        alias: ti.types.ndarray(ti.i32, 1),
        delta: ti.types.ndarray(ti.i32, 1),
        history: ti.types.ndarray(ti.i32, 1),
        visits: ti.types.ndarray(ti.i32, 1),
        x: ti.types.ndarray(ti.f32, 1),
    ):
        state[0] += delta[0]
        history[0] += 1
        visits[0] = history[0]
        x[0] = state[0] + alias[0]

    exported = ti.aot.Module()
    exported.add_field("state", state, as_array=True)
    exported.add_kernel(advance)
    _square(exported)
    exported.add_execution_plan(
        "step",
        ExecutionPlan(
            [
                dict(
                    kind="kernel",
                    entry="advance",
                    arguments=["alias", "delta", "history", "visits", "x"],
                ),
                dict(kind="fill32", destination="dx", value=0),
                dict(kind="fill32", destination="dy", value=0x3F800000),
                dict(kind="kernel", entry="square", arguments=["x", "y"]),
                dict(kind="kernel", entry="reverse", arguments=["x", "y"]),
            ],
            resources=[
                dict(kind="module_array", name="alias", view="state"),
                dict(
                    kind="ndarray",
                    name="history",
                    dtype="i32",
                    shape=[1],
                    zero_initialize=True,
                ),
            ],
            gradients=dict(x="dx", y="dy"),
        ),
    )
    artifact = tmp_path / "workspace-roots.tcm"
    exported.archive(str(artifact))
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    arrays = {
        name: dict(dtype="i32" if name in ("delta", "visits") else "f32", shape=1)
        for name in ("delta", "visits", "x", "y", "dx", "dy")
    }

    def submit(lease, delta):
        lease.arrays["delta"].write(struct.pack("<i", delta))
        return lease.submit()

    def check(lease, current, visits):
        lease.wait()
        assert struct.unpack("<i", lease.arrays["visits"].read()) == (visits,)
        assert struct.unpack("<f", lease.arrays["x"].read()) == (2 * current,)
        assert struct.unpack("<f", lease.arrays["y"].read()) == ((2 * current) ** 2,)
        assert struct.unpack("<f", lease.arrays["dx"].read()) == (4 * current,)

    for factory in (lambda: Runtime(arch), ti.aot.borrow_runtime):
        with factory() as runtime:
            first = runtime.load_module(artifact)
            second = runtime.load_module(artifact)
            with first.prepare_workspace_pool("step", 2, arrays=arrays) as pool:
                with second.prepare_workspace_pool("step", 1, arrays=arrays) as other:
                    a = submit(pool.acquire(), 1)
                    b = submit(pool.acquire(), 10)
                    c = submit(other.acquire(), 7)
                    check(c, 7, 1)
                    check(b, 11, 1)
                    check(a, 1, 1)
                    with pytest.raises(WorkspaceBusyError):
                        pool.acquire(wait=True)
                    a.release()
                    a = submit(pool.acquire(), 4)
                    check(a, 15, 2)
                    # A later submission changed the shared Field, not b's held
                    # result. A separate module owns a separate Field root.
                    check(b, 11, 1)
                    c.release()
                    c = submit(other.acquire(), 2)
                    check(c, 9, 2)
                    a.release()
                    b.release()
                    pending = submit(pool.acquire(), 1)
                    first.close()
                    pending.wait()
                    pending.release()
                    c.release()
                    c = submit(other.acquire(), 3)
                    check(c, 12, 3)
                    c.release()
            second.close()
