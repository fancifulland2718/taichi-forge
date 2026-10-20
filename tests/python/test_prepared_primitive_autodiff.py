"""Prepared discrete operations must reject automatic AD before any writes."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("kind", ("sort", "compact", "prefix", "unique"))
@pytest.mark.parametrize("mode", ("reverse", "forward"))
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, vulkan_sparse_experimental=False)
def test_prepared_primitive_rejects_automatic_ad_before_writing(kind, mode):
    if ti.cfg.arch == ti.cpu and kind != "sort":
        pytest.skip("This prepared primitive currently requires CUDA/Vulkan")
    parameter = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)
    keys, flags, output_keys = (ti.ndarray(ti.i32, 5) for _ in range(3))
    values, output_values = (ti.ndarray(ti.f32, 5) for _ in range(2))
    count = ti.ndarray(ti.i32, 1)
    keys.from_numpy(np.array([2, 2, 1, 1, 0], np.int32))
    flags.from_numpy(np.array([1, 0, 1, 0, 1], np.int32))
    values.from_numpy(np.array([9, 8, 7, 6, 5], np.float32))
    output_keys.fill(-77)
    output_values.fill(-77)
    count.fill(3 if kind == "prefix" else -77)
    if kind == "sort":
        plan = ti.algorithms.prepare_sort(keys, values)
    elif kind == "compact":
        plan = ti.algorithms.prepare_compact(values, flags, output_values, count)
    elif kind == "prefix":
        plan = ti.algorithms.prepare_sort(keys, values, count=count)
    else:
        plan = ti.algorithms.prepare_unique_by_key(keys, values, output_keys, output_values, count)
    arrays = [keys, flags, values, output_keys, output_values, count]
    if kind == "prefix":
        plan.status.fill(-77)
        arrays.append(plan.status)
    recording = plan.record()
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    graph = builder.compile()
    executions = [plan.run, lambda: graph.run({}), lambda: graph.submit({})]
    if kind in ("sort", "compact"):
        executions += [lambda: recording.execute({}), recording.prepare_graph_execute({})]
    expected = [array.to_numpy().tobytes() for array in arrays]
    for execute in executions:
        context = ti.ad.Tape(loss) if mode == "reverse" else ti.ad.FwdMode(loss=loss, param=parameter, seed=[1])
        with context:
            with pytest.raises(RuntimeError, match="ti.ad.Tape|ti.ad.FwdMode"):
                execute()
        assert [array.to_numpy().tobytes() for array in arrays] == expected
        assert ti.lang.impl.get_runtime().target_tape is None
        assert ti.lang.impl.get_runtime().fwd_mode_manager is None
    # A rejected automatic-AD call must leave the same plan reusable in primal execution.
    plan.run()
    if kind == "sort":
        np.testing.assert_array_equal(values.to_numpy(), [5, 7, 6, 9, 8])
    elif kind == "prefix":
        np.testing.assert_array_equal(values.to_numpy(), [7, 9, 8, 6, 5])
        np.testing.assert_array_equal(plan.status.to_numpy(), [3, 0])
    else:
        assert count.to_numpy()[0] == 3
        np.testing.assert_array_equal(output_values.to_numpy()[:3], [9, 7, 5])
    graph.close()
    plan.close()
