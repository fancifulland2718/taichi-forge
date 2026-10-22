"""Check automatic owner retirement before Python/native module teardown."""

import atexit
from pathlib import Path
import sys

state = {}
output = Path(sys.argv[3])


def check_retired():
    # Registered before importing Forge, so this runs after its exit handler.
    # Keep strong references to live wrappers: GC must not be the mechanism
    # that makes the test pass.
    assert state["runtime"].prog is None
    assert impl.get_runtime() is state["runtime"]  # No replacement PyTaichi.
    if "borrower" in state:
        assert state["borrower"]._borrower.closed
        assert state["borrower"]._handle is None
        assert state["value"].arr is None
        assert state["program"]._debug_runtime_fault_state()["state"] == "finalized"
    assert state["user_exit_ran"]
    output.write_text("retired", encoding="utf-8")


atexit.register(check_retired)

import taichi_forge as ti
from taichi_forge.lang import impl


def check_user_exit():
    # Application callbacks registered after import can still use the owner.
    if sys.argv[2] == "live":
        assert state["runtime"].prog is state["program"]
        assert not state["borrower"]._borrower.closed
    state["user_exit_ran"] = True


atexit.register(check_user_exit)


if sys.argv[2] != "uninitialized":
    ti.init(arch=getattr(ti, sys.argv[1]), offline_cache=False)
    value = ti.ndarray(ti.i32, shape=4)
    value.fill(0)

    @ti.kernel
    def advance(dst: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in dst:
            dst[i] += i + 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.i32, ndim=1))
    graph = builder.compile()
    borrower = ti.aot.borrow_runtime()
    borrowed_value = borrower.import_ndarray(value)
    advance(value)
    assert value.to_numpy().tolist() == [1, 2, 3, 4]
    completion = graph.submit({"dst": value})  # Leave submission/owners live.
    state.update(value=value, borrower=borrower, program=impl.get_runtime().prog)
    if sys.argv[2] == "reset":
        ti.reset()

state["runtime"] = impl.get_runtime()
