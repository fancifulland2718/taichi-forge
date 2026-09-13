import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from tests import test_utils


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.cuda])
def test_graph_borrowed_display_storage_retention_and_retirement():
    window = ti.ui.Window("graph display", (12, 20), show_window=False)
    canvas = window.get_canvas()
    program = impl.get_runtime().prog
    baseline = program._debug_external_dense_storage_stats()

    @ti.kernel
    def paint(target: ti.types.ndarray(dtype=ti.u32, ndim=2), value: ti.u32):
        for i, j in target:
            target[i, j] = value

    builder = ti.graph.GraphBuilder()
    target = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "target", ti.u32, ndim=2)
    value_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.u32)
    builder.dispatch(paint, target, value_arg)
    builder.dispatch(paint, target, value_arg)
    graph = builder.compile()
    saved = None
    try:
        for value in (0xFF0000FF, 0xFF00FF00, 0xFF332211):
            with canvas.acquire_frame(12, 20) as frame:
                bound = graph.bind(dict(target=frame.pixels, value=value))
                # Both dispatches share one external producer epoch; no global
                # inflight lease may prevent display-consumer retirement.
                graph.run(bound)
                assert canvas.submit_frame(frame, track_source=True)
            frame.source_completion.wait()
            assert window.show()
            frame.completion.wait()
            saved = bound
        canvas.repeat_frame()
        image = window.get_image_buffer_as_numpy()
        np.testing.assert_allclose(
            image, np.broadcast_to(np.array([17, 34, 51, 255]) / 255, image.shape), atol=1 / 255 + 1e-5
        )
        # A cancelled epoch also has to release the Graph's local leases.
        with canvas.acquire_frame(12, 20) as cancelled:
            graph.run(graph.bind(dict(target=cancelled.pixels, value=0)))
        assert cancelled.completion.status == "cancelled"
        window.destroy()
        with pytest.raises(RuntimeError, match="retired|stale|closed|invalid"):
            graph.run(saved)
    finally:
        graph.close()
        window.destroy()
    ti.sync()
    final = program._debug_external_dense_storage_stats()
    for key in ("live", "retiring", "leases"):
        assert final[key] == baseline[key]
