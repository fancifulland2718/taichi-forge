import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


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
