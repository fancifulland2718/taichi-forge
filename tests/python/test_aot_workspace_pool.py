"""Workspace consumption and device completion are independent obligations."""

from concurrent.futures import ThreadPoolExecutor
import json
import struct
import threading
import time
from zipfile import ZipFile

import pytest
import taichi_forge as ti

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from taichi_forge_runtime.aot._workspace import WorkspaceBusyError, WorkspacePool
from tests import test_utils


class _Array:
    nbytes = 4

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Ticket:
    def __init__(self):
        self.complete = threading.Event()
        self.waiting = threading.Event()
        self.error = None
        self.closed = False
        self.queries = 0

    def done(self):
        assert not self.closed
        self.queries += 1
        if self.error:
            raise self.error
        return self.complete.is_set()

    def wait(self):
        assert not self.closed
        self.waiting.set()
        if self.error:
            raise self.error
        assert self.complete.wait(3), "Test did not release its completion"

    def close(self):
        self.closed = True


class _Call:
    def __init__(self):
        self.ticket = _Ticket()
        self.error = None
        self.closed = False

    def submit(self):
        if self.error:
            raise self.error
        return self.ticket

    def close(self):
        self.closed = True


class _Module:
    def __init__(self, fail_prepare=None):
        self._runtime = self
        self.arrays, self.calls = [], []
        self.fail_prepare = fail_prepare

    def ndarray(self, **_):
        array = _Array()
        self.arrays.append(array)
        return array

    def prepare_execution(self, *_args, **_kwargs):
        if len(self.calls) == self.fail_prepare:
            raise ValueError("injected prepare failure")
        call = _Call()
        self.calls.append(call)
        return call


def _pool(capacity=3, module=None):
    module = module or _Module()
    plan = ExecutionPlan([dict(kind="fill32", destination="out", value=0)])
    return (
        WorkspacePool(
            module, plan, capacity, arrays=dict(out=dict(dtype="i32", shape=1))
        ),
        module,
    )


def test_completed_results_remain_leased_and_stale_leases_cannot_submit():
    pool, module = _pool()
    leases = [pool.acquire() for _ in range(3)]
    for lease, call in zip(leases, module.calls):
        lease.submit()
        call.ticket.complete.set()
        lease.wait()
    for wait in (False, True):
        with pytest.raises(WorkspaceBusyError, match="results are leased"):
            pool.acquire(wait=wait)
    leases[1].release()
    replacement = pool.acquire()
    assert replacement.arrays is not leases[0].arrays
    with pytest.raises(RuntimeError, match="released"):
        leases[1].submit()
    leases[1].release()  # Cannot release the new generation.
    with pytest.raises(WorkspaceBusyError):
        pool.acquire()
    replacement.release()
    for lease in leases:
        lease.release()
    pool.close()
    assert all(call.closed for call in module.calls)
    assert all(array.closed for array in module.arrays)


def test_discard_wait_leaves_submission_and_other_slots_available():
    pool, module = _pool(2)
    first, second = pool.acquire(), pool.acquire()
    first.submit().discard()
    with pytest.raises(WorkspaceBusyError, match="in flight"):
        pool.acquire()
    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(pool.acquire, wait=True)
        try:
            assert module.calls[0].ticket.waiting.wait(2)
            second.submit()
            module.calls[1].ticket.complete.set()
            second.release()
            available = pool.acquire()
            assert available._index == 1
            with pytest.raises(WorkspaceBusyError, match="reclaiming"):
                pool.acquire()
        finally:
            module.calls[0].ticket.complete.set()
        reclaimed = waiting.result(timeout=2)
    assert reclaimed._index == 0
    assert module.calls[0].ticket.queries == 1
    available.release()
    reclaimed.release()
    pool.close()


def test_prepare_failure_retires_created_calls_and_partial_arrays():
    module = _Module(fail_prepare=2)
    with pytest.raises(ValueError, match="injected prepare"):
        _pool(3, module)
    assert len(module.calls) == 2 and len(module.arrays) == 3
    assert all(call.closed for call in module.calls)
    assert all(array.closed for array in module.arrays)


def test_submission_failure_stops_admission_without_recycling_partial_work():
    pool, module = _pool(2)
    lease = pool.acquire()
    failure = ValueError("partial submission failed")
    module.calls[0].error = failure
    with pytest.raises(ValueError, match="partial submission"):
        lease.submit()
    lease.release()
    with pytest.raises(RuntimeError, match="pool failed") as error:
        pool.acquire()
    assert error.value.__cause__ is failure
    pool.close()
    assert all(call.closed for call in module.calls)
    assert all(array.closed for array in module.arrays)


def test_completion_failure_preserves_first_error_and_closes_other_slots():
    pool, module = _pool(2)
    first, second = pool.acquire(), pool.acquire()
    first.submit()
    second.submit()
    module.calls[0].ticket.error = ValueError("first device failure")
    module.calls[1].ticket.complete.set()
    with pytest.raises(ValueError, match="first device"):
        first.wait()
    with pytest.raises(RuntimeError) as error:
        pool.acquire()
    assert error.value.__cause__ is module.calls[0].ticket.error
    with pytest.raises(ValueError, match="first device"):
        pool.close()
    assert all(call.closed for call in module.calls)
    assert all(array.closed for array in module.arrays)
    assert all(call.ticket.closed for call in module.calls)
    pool.close()  # Closed completion handles are not queried again on retry.


def test_workspace_capacity_is_not_limited_to_a_fixed_small_ring():
    pool, module = _pool(65)
    leases = [pool.acquire() for _ in range(65)]
    assert len({id(lease.arrays["out"]) for lease in leases}) == 65
    for lease in reversed(leases):
        lease.release()
    assert all(call.ticket.queries == 0 for call in module.calls)
    pool.close()


def _archive(path, arch):
    @ti.kernel
    def advance(
        state: ti.types.ndarray(ti.f32, ndim=1), x: ti.types.ndarray(ti.f32, ndim=1)
    ):
        for i in x:
            state[i] += 1
            x[i] = state[i] + i

    @ti.kernel
    def square(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in x:
            y[i] = x[i] * x[i]

    @ti.kernel
    def condition(
        count: ti.types.ndarray(ti.i32, ndim=1),
        predicate: ti.types.ndarray(ti.i32, ndim=1),
        target: ti.types.ndarray(ti.i32, ndim=1),
    ):
        predicate[0] = ti.cast(count[0] < target[0], ti.i32)

    @ti.kernel
    def increment(count: ti.types.ndarray(ti.i32, ndim=1)):
        count[0] += 1

    exported = ti.aot.Module()
    for kernel in (advance, square, condition, increment):
        exported.add_kernel(kernel)
    exported.add_kernel(square.grad, name="reverse")
    exported.add_execution_plan(
        "step",
        ExecutionPlan(
            [
                dict(kind="kernel", entry="advance", arguments=["state", "x"]),
                dict(kind="fill32", destination="dx", value=0),
                dict(kind="fill32", destination="dy", value=0x3F800000),
                dict(kind="kernel", entry="square", arguments=["x", "y"]),
                dict(kind="kernel", entry="reverse", arguments=["x", "y"]),
            ],
            resources=[
                dict(
                    kind="ndarray",
                    name="state",
                    dtype="f32",
                    shape=[dict(array="x", axis=0)],
                    zero_initialize=True,
                )
            ],
            gradients=dict(x="dx", y="dy"),
        ),
    )
    loop = dict(
        kind="while",
        predicate="predicate",
        max_iterations=4,
        condition=[
            dict(
                kind="kernel",
                entry="condition",
                arguments=["count", "predicate", "target"],
            )
        ],
        body=[dict(kind="kernel", entry="increment", arguments=["count"])],
    )
    if arch == "vulkan":
        loop["lowering"] = dict(
            kind="vulkan_structured",
            version=1,
            counter="count",
            chunks=[dict(iterations=4, strategy="compact")],
        )
    exported.add_execution_plan(
        "terminal",
        ExecutionPlan(
            [
                dict(kind="fill32", destination="count", value=0),
                loop,
                dict(kind="copy", source="count", destination="terminal"),
            ],
            resources=[
                dict(kind="ndarray", name=name, dtype="i32", shape=[1])
                for name in ("count", "predicate")
            ],
        ),
    )
    exported.archive(str(path))


def _values(lease, name):
    array = lease.arrays[name]
    return list(
        struct.unpack(
            "<" + ("f" if array.dtype == "f32" else "i") * array.shape[0], array.read()
        )
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_complete_workspaces_keep_private_state_gradients_and_terminal(tmp_path):
    arch = (
        "cpu"
        if ti.lang.impl.current_cfg().arch == ti.cpu
        else str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    )
    artifact = tmp_path / "workspaces.tcm"
    started = time.perf_counter_ns()
    _archive(artifact, arch)
    export_ms = (time.perf_counter_ns() - started) / 1e6
    with ZipFile(artifact) as archive:
        code = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.endswith((".ll", ".bc", ".spv", ".cubin", ".ptx"))
        }
    measurements = []
    for mode, factory in (
        ("standalone", lambda: Runtime(arch)),
        ("borrowed", ti.aot.borrow_runtime),
    ):
        with factory() as runtime:
            module = runtime.load_module(artifact)
            shared_kernel = module.kernel("square")._handle
            specs = {
                name: dict(dtype="f32", shape=257) for name in ("x", "y", "dx", "dy")
            }
            started = time.perf_counter_ns()
            pool = module.prepare_workspace_pool("step", 3, arrays=specs)
            prepare_ms = (time.perf_counter_ns() - started) / 1e6
            pool.initialize()
            info = pool.info
            assert info["capacity"] == 3 and info["array_bytes"] == 3 * 4 * 257 * 4
            assert module.kernel("square")._handle == shared_kernel
            held = [pool.acquire() for _ in range(3)]
            for lease in held:
                lease.submit()
            for lease in reversed(held):
                lease.wait()
                assert _values(lease, "x") == [i + 1 for i in range(257)]
                assert _values(lease, "y") == [(i + 1) ** 2 for i in range(257)]
                assert _values(lease, "dx") == [2 * (i + 1) for i in range(257)]
            with pytest.raises(WorkspaceBusyError):
                pool.acquire(wait=True)
            counts = [1, 1, 1]
            samples = []
            for old in (held[2], held[0], held[1]):
                old.release()
                lease = pool.acquire()
                counts[lease.index] += 1
                began = time.perf_counter_ns()
                lease.submit()
                submitted = time.perf_counter_ns()
                lease.wait()
                samples.append(
                    dict(
                        host_us=(submitted - began) / 1e3,
                        complete_us=(time.perf_counter_ns() - began) / 1e3,
                    )
                )
                assert _values(lease, "x")[0] == counts[lease.index]
                lease.discard()
            # Acquire reclaims discarded work without scanning live results.
            lease = pool.acquire(wait=True)
            counts[lease.index] += 1
            lease.submit().wait()
            assert _values(lease, "dx")[0] == 2 * counts[lease.index]
            lease.release()
            old = pool.acquire().submit()
            pool.close()
            with pytest.raises(RuntimeError, match="closed"):
                old.wait()
            old.release()
            pool.close()

            # A provider owns private prepared sort workspace in each slot.
            sort = ExecutionPlan([dict(kind="sort", keys="keys", values="values")])
            with module.prepare_workspace_pool(
                sort,
                3,
                arrays={
                    name: dict(dtype="i32", shape=257) for name in ("keys", "values")
                },
            ) as sorting:
                leases = [sorting.acquire() for _ in range(3)]
                for ordinal, lease in enumerate(leases):
                    lease.arrays["keys"].write(
                        struct.pack("<257i", *[256 - i + ordinal for i in range(257)])
                    )
                    lease.arrays["values"].write(struct.pack("<257i", *range(257)))
                    lease.submit()
                for ordinal, lease in reversed(list(enumerate(leases))):
                    lease.wait()
                    assert _values(lease, "keys") == list(range(ordinal, 257 + ordinal))
                    assert _values(lease, "values") == list(reversed(range(257)))
                    lease.release()

            # Zero iterations, early termination and exhausted budget retain
            # separate terminal results until their consumers return them.
            with module.prepare_workspace_pool(
                "terminal",
                3,
                arrays={
                    name: dict(dtype="i32", shape=1) for name in ("target", "terminal")
                },
            ) as control:
                leases = [control.acquire() for _ in range(3)]
                for lease, target in zip(leases, (0, 2, 7)):
                    lease.arrays["target"].write(struct.pack("<i", target))
                    lease.submit()
                for lease, expected in reversed(list(zip(leases, (0, 2, 4)))):
                    lease.wait()
                    assert _values(lease, "terminal") == [expected]
                    lease.release()
            measurements.append(
                dict(mode=mode, prepare_ms=prepare_ms, info=info, samples=samples)
            )
            if arch == "cuda" and mode == "standalone":
                tiny = {name: dict(dtype="f32", shape=1) for name in specs}
                with module.prepare_workspace_pool("step", 65, arrays=tiny) as wide:
                    held = [wide.acquire() for _ in range(65)]
                    for lease in held:
                        lease.submit()
                    for lease in reversed(held):
                        lease.wait()
                        assert _values(lease, "dx") == [2]
                        lease.release()
                    assert wide.info["array_bytes"] == 65 * 4 * 4
                    assert module.kernel("square")._handle == shared_kernel
            # Closing a module invalidates its prepared work, preserving safe
            # cleanup of a pool whose slots have already been acquired.
            pool = module.prepare_workspace_pool("step", 1, arrays=specs)
            pending = pool.acquire().submit()
            module.close()
            pending.wait()
            pending.release()
            with pytest.raises(RuntimeError):
                pool.acquire().submit()
            pool.close()
    # Program reset retires the borrowed native resources before late leases.
    runtime = ti.aot.borrow_runtime()
    module = runtime.load_module(artifact)
    pool = module.prepare_workspace_pool("step", 3, arrays=specs)
    pending = pool.acquire().submit()
    ti.reset()
    pending.wait()
    pool.close()
    pending.release()
    runtime.close()
    (tmp_path / "measurements.json").write_text(
        json.dumps(
            dict(
                arch=arch,
                export_ms=export_ms,
                artifact_bytes=artifact.stat().st_size,
                code_entries=len(code),
                code_bytes=sum(map(len, code.values())),
                modes=measurements,
            )
        )
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_workspace_failure_retires_all_slots_after_owner_reset(tmp_path):
    artifact = tmp_path / "fault.tcm"
    _archive(artifact, "cuda")
    runtime = ti.aot.borrow_runtime()
    module = runtime.load_module(artifact)
    specs = {name: dict(dtype="f32", shape=1) for name in ("x", "y", "dx", "dy")}
    pool = module.prepare_workspace_pool("step", 3, arrays=specs)
    lease = pool.acquire().submit()
    runtime.wait()
    ti.lang.impl.get_runtime().prog._debug_inject_runtime_fault(
        -1, "workspace-test", "workspace owner failure"
    )
    ti.reset()
    with pytest.raises(RuntimeError, match="workspace owner failure"):
        lease.wait()
    with pytest.raises(RuntimeError, match="pool failed") as error:
        pool.acquire()
    assert "workspace owner failure" in str(error.value.__cause__)
    with pytest.raises(RuntimeError, match="workspace owner failure"):
        pool.close()
    pool.close()
    lease.release()
    module.close()
    runtime.close()
