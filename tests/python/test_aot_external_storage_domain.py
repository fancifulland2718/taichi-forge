"""Public JIT/AOT recipes own one CUDA/Vulkan epoch across all dispatches."""

from contextlib import ExitStack

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as _ti_core
from tests import test_utils


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI not available")
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_external_epoch_through_public_recipe(tmp_path, jit, selected):
    @ti.kernel
    def paint(target: ti.types.ndarray(ti.u32, 2)):
        for i, j in target:
            target[i, j] = ti.u32(0xFF332211)

    window = ti.ui.Window("prepared external epoch", (12, 20), show_window=False)
    canvas = window.get_canvas()
    calls = {}
    loans = []
    with ExitStack() as scope:
        if jit:
            domain = ti.graph.execution_domain(instance_capacity=16)
            builder = ti.graph.GraphBuilder()
            left = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "left", ti.u32, ndim=2)
            right = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "right", ti.u32, ndim=2)
            builder.dispatch(paint, left)
            builder.dispatch(paint, right)
            graph = builder.compile()
        else:
            exported = ti.aot.Module()
            exported.add_kernel(paint)
            exported.add_execution_plan(
                "paint_twice",
                ti.aot.ExecutionPlan(
                    [
                        dict(kind="kernel", entry="paint", arguments=["left"]),
                        dict(kind="kernel", entry="paint", arguments=["right"]),
                    ]
                ),
            )
            artifact = tmp_path / "external.tcm"
            exported.archive(str(artifact))
            runtime = scope.enter_context(ti.aot.borrow_runtime())
            domain = runtime.execution_domain(instance_capacity=16)
            module = runtime.load_module(artifact)
        if selected:
            canvas.set_execution_domain(domain)
            assert len(canvas.prepare_frame_pool(12, 20, capacity=3)) == 3
        try:
            for _ in range(8):
                frame = canvas.acquire_frame(12, 20)
                assert frame is not None
                key = id(frame.pixels.descriptor)
                if key not in calls:
                    if jit:
                        call = graph.prepare(
                            dict(left=frame.pixels, right=frame.pixels),
                            execution_domain=domain,
                        )
                    else:
                        # Distinct loans of one physical owner still consume
                        # the binary semaphore exactly once for the recipe.
                        left = runtime.import_dense_view(frame.pixels)
                        right = runtime.import_dense_view(frame.pixels)
                        loans.extend([left, right])
                        with pytest.raises(RuntimeError, match="external storage"):
                            module.kernel("paint")(left)
                        call = module.prepare_execution(
                            "paint_twice",
                            dict(left=left, right=right),
                            execution_domain=domain,
                        )
                    call.initialize()
                    calls[key] = call
                ticket = calls[key].submit()
                assert ticket is not None
                if selected:
                    # Tracking needs this exact producer, not the current tail.
                    with pytest.raises(RuntimeError, match="exact producer"):
                        canvas.submit_frame(frame, track_source=True)
                    assert canvas.submit_frame(
                        frame, track_source=True, source_completion=ticket
                    )
                    assert frame.source_completion.sequence == ticket.info["sequence"]
                else:
                    assert canvas.submit_frame(frame)
                # Readback proves actual Vulkan consumption of both dispatches.
                image = window.get_image_buffer_as_numpy()
                np.testing.assert_allclose(
                    image,
                    np.broadcast_to(np.array([17, 34, 51, 255]) / 255, image.shape),
                    atol=1 / 255,
                )
                ticket.wait()
                if selected:
                    frame.source_completion.wait()
                    assert canvas.repeat_frame() is not None
                    repeated = window.get_image_buffer_as_numpy()
                    np.testing.assert_allclose(repeated, image, atol=1 / 255)
            assert 1 <= len(calls) <= 3
            if selected:
                assert len(calls) == 3
            window.destroy()
            assert domain.info["closed"]
            with pytest.raises(RuntimeError, match="retired|closed|live"):
                next(iter(calls.values())).submit()
        finally:
            for call in calls.values():
                call.close()
            domain.close()
            for loan in loans:
                loan.close()
            if jit:
                graph.close()
            else:
                module.close()
            window.destroy()
