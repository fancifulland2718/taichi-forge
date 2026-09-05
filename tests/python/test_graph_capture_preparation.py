"""Capture preparation must not execute a recurrent Graph ahead of its first run."""

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize(
    "operation", ("fft", "spmv", "spmm", "spsv", "spsm", "bsr_spmv")
)
@pytest.mark.parametrize("with_kernel", (False, True))
def test_capture_preparation_preserves_feedback_state(operation, with_kernel):
    plan = None
    if operation == "fft":
        if not ti.hardware.fft.is_available():
            pytest.skip("the optional cuFFT runtime is unavailable")
        plan = ti.hardware.fft.CufftPlan1D(16)
        shape, factor = (16, 2), 16.0
        recordings = (
            plan.record(input="state", output="scratch"),
            plan.record(direction="inverse", input="scratch", output="state"),
        )
    else:
        available = getattr(
            ti.hardware.linalg,
            {
                "spmv": "cusparse_is_available",
                "bsr_spmv": "cusparse_is_available",
                "spmm": "cusparse_spmm_is_available",
                "spsv": "cusparse_spsv_is_available",
                "spsm": "cusparse_spsm_is_available",
            }[operation],
        )
        if not available():
            pytest.skip("the optional cuSPARSE execution capability is unavailable")
        if operation == "bsr_spmv":
            offsets, columns, values = [0, 1, 2], [0, 1], [2, 0, 0, 2] * 2
        else:
            offsets, columns, values = list(range(5)), list(range(4)), [2] * 4
        arrays = []
        for host, dtype in ((offsets, ti.i32), (columns, ti.i32), (values, ti.f32)):
            array = ti.ndarray(dtype, len(host))
            array.from_numpy(
                np.asarray(host, np.float32 if dtype == ti.f32 else np.int32)
            )
            arrays.append(array)
        if operation == "bsr_spmv":
            pattern = ti.linalg.SparsePattern.bsr(2, 2, 2, *arrays[:2])
        else:
            pattern = ti.linalg.SparsePattern.csr(4, 4, *arrays[:2])
        matrix = pattern.matrix(arrays[2])
        constructor = getattr(
            ti.hardware.linalg,
            {
                "spmv": "CusparseSpmvRecording",
                "bsr_spmv": "CusparseSpmvRecording",
                "spmm": "CusparseSpmmRecording",
                "spsv": "CusparseSpsvRecording",
                "spsm": "CusparseSpsmRecording",
            }[operation],
        )
        multiple = operation in ("spmm", "spsm")
        arguments = (matrix, 2) if multiple else (matrix,)
        shape = (4, 2) if multiple else (4,)
        factor = 0.25 if operation in ("spsv", "spsm") else 4.0
        recordings = tuple(
            constructor(*arguments, input=source, output=target)
            for source, target in (("state", "scratch"), ("scratch", "state"))
        )
    bindings = {name: ti.ndarray(ti.f32, shape) for name in ("state", "scratch")}
    initial = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 32 + 0.25
    bindings["state"].from_numpy(initial)
    builder = ti.graph.GraphBuilder()
    for recording in recordings:
        builder.append_native(recording, admission="auto")
    if with_kernel:
        bindings["observed"] = ti.ndarray(ti.f32, shape)

        @ti.kernel
        def observe(
            state: ti.types.ndarray(dtype=ti.f32),
            output: ti.types.ndarray(dtype=ti.f32),
        ):
            for index in ti.grouped(state):
                output[index] = state[index]

        builder.dispatch(
            observe,
            *(
                ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=len(shape))
                for name in ("state", "observed")
            ),
        )
    graph = builder.compile()
    bound = graph.bind(bindings)
    try:
        for iteration in range(1, 4):
            graph.run(bound)
            expected = initial * factor**iteration
            np.testing.assert_allclose(
                bindings["state"].to_numpy(), expected, rtol=1e-5, atol=1e-5
            )
            if with_kernel:
                np.testing.assert_allclose(
                    bindings["observed"].to_numpy(), expected, rtol=1e-5, atol=1e-5
                )
        assert (
            graph._graph_stats[0]["last_path"] == "cuda_exact_replay"
        ), graph._graph_stats
    finally:
        if plan is not None:
            plan.close()
