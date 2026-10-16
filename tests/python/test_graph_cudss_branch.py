"""Qualified native providers inside CUDA IF/SWITCH share capture ownership."""

import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.hardware._cudss_capture import CudssCaptureRecording
from tests import test_utils
from tests.python.test_graph_cudss_capture import _forbidden, _owner


def build_branch(owner, kind):
    @ti.kernel
    def seed(solution: ti.types.ndarray(ti.f32, 1)):
        for i in solution:
            solution[i] = -91

    @ti.kernel
    def condition(choice: ti.i32, selected: ti.types.ndarray(ti.i32, 0)):
        selected[None] = choice

    @ti.kernel
    def shift(solution: ti.types.ndarray(ti.f32, 1), delta: ti.f32):
        for i in solution:
            solution[i] += delta

    builder = ti.graph.GraphBuilder()
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "solution", ti.f32, ndim=1)
    choice = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "choice", ti.i32)
    selected = builder.private_ndarray("selected", ti.i32, ())
    parent, cond, first, second, default = (builder.create_sequential() for _ in range(5))
    parent.dispatch(seed, output)
    cond.dispatch(condition, choice, selected)
    first.append_native(CudssCaptureRecording(owner, phase="refactor_solve", values="values0"))
    # Different physical counts also exercise the provider/kernel boundary map.
    second.dispatch(shift, output, second._bind_internal_scalar("unused_seed", ti.f32, 7))
    second.append_native(CudssCaptureRecording(owner, phase="refactor_solve", values="values1"))
    second.dispatch(shift, output, second._bind_internal_scalar("selected_shift", ti.f32, 2))
    default.dispatch(shift, output, default._bind_internal_scalar("default_shift", ti.f32, 8))
    if kind == "if":
        parent.if_then_else(
            cond, first, predicate=selected, else_region=second, lowering_mode="native_required"
        )
    else:
        parent.switch(
            cond, [first, second], selector=selected, default_region=default, lowering_mode="native_required"
        )
    parent.dispatch(shift, output, parent._bind_internal_scalar("tail_shift", ti.f32, 100))
    builder.append(parent)
    return builder.compile()


@pytest.mark.parametrize("kind", ["if", "switch"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_native_branches_rebind_and_retire(kind, monkeypatch):
    monkeypatch.delenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", raising=False)
    owner, _, dense = _owner()
    graph = build_branch(owner, kind)
    values = np.array([4, -1, -1, 4, -1, -1, 4, -1, -1, 3], np.float32)
    frames = []
    for scale in (1, 3):
        arrays = {
            name: ti.ndarray(ti.f32, size)
            for name, size in (("rhs", 4), ("solution", 4), ("values0", 10), ("values1", 10))
        }
        arrays["rhs"].from_numpy(np.arange(1, 5, dtype=np.float32) * (scale + 1))
        arrays["values0"].from_numpy(values * scale)
        arrays["values1"].from_numpy(values * scale * 2)
        arrays["solution"].fill(71)
        frames.append((arrays, scale))
    # No host execution/fallback may replace the selected provider command.
    with monkeypatch.context() as replay:
        replay.setattr(CudssCaptureRecording, "execute", _forbidden)
        for choice in (-1, 0, 1, 2, 0):
            for arrays, scale in frames:
                graph.submit(dict(arrays, choice=choice)).wait()
                selected = (0 if choice else 1) if kind == "if" else choice
                if selected in (0, 1):
                    expected = np.linalg.solve(dense * scale * (selected + 1), np.arange(1, 5) * (scale + 1))
                    expected += 100 + 2 * selected
                else:
                    expected = np.full(4, 17)
                np.testing.assert_allclose(arrays["solution"].to_numpy(), expected, rtol=2e-6, atol=2e-5)
        # Live device values change without reconstructing the binding.
        arrays, scale = frames[0]
        arrays["values0"].from_numpy(values * 4)
        graph.submit(dict(arrays, choice=1 if kind == "if" else 0)).wait()
        np.testing.assert_allclose(
            arrays["solution"].to_numpy(), np.linalg.solve(dense * 4, np.arange(1, 5) * 2) + 100, rtol=2e-6
        )
    with pytest.raises(RuntimeError, match="retained by a Graph"):
        owner.close()
    pending = graph.submit(dict(frames[1][0], choice=0))
    graph.close()
    pending.wait()
    del pending, graph
    gc.collect()
    assert owner._capture_leases == 0
    owner.close()
