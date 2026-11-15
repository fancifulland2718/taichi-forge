import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_fault_snapshot_shares_first_fault_without_waiting(monkeypatch):
    import ctypes as C
    from dataclasses import asdict, FrozenInstanceError

    runtime = ti.aot.borrow_runtime()
    healthy = runtime.fault_snapshot()
    assert asdict(healthy) == asdict(ti.runtime.stats().fault)
    before = ti.runtime.stats().synchronization
    with monkeypatch.context() as patch:
        patch.setattr(runtime, 'wait', lambda: pytest.fail('Fault query must not wait'))
        assert runtime.fault_snapshot() == healthy
    assert ti.runtime.stats().synchronization == before

    # The C ABI reports the required capacity without partially overwriting a
    # short buffer. Invalid pointers reject without changing fault state.
    runtime._api.load_runtime_fault_snapshot()
    query = runtime._api.ti_get_runtime_fault_snapshot_ext
    size = C.c_uint64(1)
    buffer = C.create_string_buffer(b'Z')
    runtime._api.call(query, runtime._handle, C.byref(size), buffer)
    assert size.value > 1 and buffer.value == b'Z'
    with pytest.raises(RuntimeError, match='size'):
        runtime._api.call(query, runtime._handle, None, None)
    assert runtime.fault_snapshot() == healthy

    program = impl.get_runtime().prog
    code = 700 if ti.cfg.arch == ti.cuda else -1
    message = 'quoted "message" and unicode 关闭\\newline\n' * 50
    program._debug_inject_runtime_fault(code, 'snapshot_test', message)
    first = runtime.fault_snapshot()
    assert asdict(first) == asdict(ti.runtime.fault_snapshot())
    assert first.state == 'faulted' and first.first_fatal_faults == 1
    assert first.first_fault.message == message
    # Check the same record through the independent C ABI, including its resize
    # path and JSON escaping. The borrowed adapter still owns this handle here.
    import json
    size = C.c_uint64(0)
    runtime._api.call(query, runtime._handle, C.byref(size), None)
    assert size.value > 512
    buffer = C.create_string_buffer(size.value)
    runtime._api.call(query, runtime._handle, C.byref(size), buffer)
    raw = json.loads(buffer.value)
    assert raw.pop('schema_version') == 1 and raw == asdict(first)
    assert healthy.state == 'healthy' and healthy.first_fault is None
    with pytest.raises(FrozenInstanceError):
        first.state = 'healthy'
    program._debug_inject_runtime_fault(code, 'second_fault', 'must not replace the first record')
    with monkeypatch.context() as patch:
        patch.setattr(runtime, 'wait', lambda: pytest.fail('Fault query must not wait'))
        assert runtime.fault_snapshot() == first
    runtime.close()
    ti.reset()
    with pytest.raises(RuntimeError, match='closed'):
        runtime.fault_snapshot()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_faulted_runtime_rejects_work_and_reset_creates_healthy_program():
    arch = impl.current_cfg().arch
    value = ti.ndarray(ti.i32, shape=1)

    @ti.kernel
    def advance(dst: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        dst[0] += 1

    arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance, arg)
    graph = builder.compile()

    advance(value)
    ti.sync()
    graph.run({"dst": value})
    ti.sync()

    prog = impl.get_runtime().prog
    injected_code = 700 if arch == ti.cuda else -4 if arch == ti.vulkan else -1
    prog._debug_inject_runtime_fault(
        injected_code, "injected_failure", "injected fatal backend failure"
    )

    state = prog._debug_runtime_fault_state()
    assert state["state"] == "faulted"
    assert state["backend_code"] == injected_code
    assert state["operation"] == "injected_failure"
    assert state["message"] == "injected fatal backend failure"

    for operation in (
        lambda: advance(value),
        lambda: graph.run({"dst": value}),
        lambda: graph.submit({"dst": value}),
        lambda: graph.submit({"dst": value}, wait_mode="blocking"),
        ti.sync,
        prog._record_runtime_completion,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            operation()
        message = str(exc_info.value)
        assert "Runtime is faulted" in message
        assert "injected_failure" in message
        assert "injected fatal backend failure" in message

    rejected = prog._debug_runtime_fault_state()["rejected_submissions"]
    assert rejected >= 5

    # reset() must not re-enter a backend wait after fatal state. The old
    # Device/context is not recovered in place; a new Program owns the next
    # healthy submission domain.
    ti.reset()
    ti.init(arch=arch)
    fresh = ti.field(ti.i32, shape=())

    @ti.kernel
    def initialize_fresh_program():
        fresh[None] = 42

    initialize_fresh_program()
    ti.sync()
    assert fresh[None] == 42
    fresh_state = impl.get_runtime().prog._debug_runtime_fault_state()
    assert fresh_state["state"] == "healthy"
    assert fresh_state["message"] is None


@test_utils.test(arch=ti.vulkan)
def test_faulted_vulkan_ggui_destroy_skips_backend_waits():
    window = ti.ui.Window(
        "faulted Vulkan GGUI teardown",
        (64, 64),
        show_window=False,
        vsync=False,
    )
    canvas = window.get_canvas()
    canvas.set_image(np.zeros((64, 64, 4), dtype=np.uint8))
    window.show()

    prog = impl.get_runtime().prog
    prog._debug_inject_runtime_fault(
        -4, "injected_show_failure", "injected Vulkan device loss"
    )
    with pytest.raises(RuntimeError, match="injected_show_failure"):
        window.show()

    # Window and Renderer destructors must abandon unsafe backend waits while
    # still releasing host-side owners. Neither destroy() nor reset() may
    # terminate the process or replace the first fault.
    window.destroy()
    ti.reset()
