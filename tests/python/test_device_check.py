import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _skip_if_native_check_unavailable():
    arch = impl.current_cfg().arch
    prog = impl.get_runtime().prog
    if arch == ti.cuda:
        if not (
            hasattr(prog, "cuda_cub_check_count_available")
            and prog.cuda_cub_check_count_available()
        ):
            pytest.skip("CUDA native check_count is unavailable.")
        return
    if arch == ti.vulkan:
        if not (
            hasattr(prog, "vulkan_check_count_available")
            and prog.vulkan_check_count_available()
        ):
            pytest.skip("Vulkan native check_count is unavailable.")
        if (
            hasattr(prog, "vulkan_check_count_value_type_available")
            and not prog.vulkan_check_count_value_type_available(1)
        ):
            pytest.skip("Vulkan native check_count f32 is unavailable.")
        return
    if not (
        hasattr(prog, "cpu_check_count_available")
        and prog.cpu_check_count_available()
    ):
        pytest.skip("CPU native check_count is unavailable.")


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_check_predicates():
    _skip_if_native_check_unavailable()
    flags = ti.ndarray(ti.i32, shape=5)
    flags.from_numpy(np.array([0, 1, 2, 0, -3], dtype=np.int32))
    workspace = ti.algorithms.CheckWorkspace(max_items=5)

    count = ti.algorithms.count_if(flags, workspace=workspace)
    assert count.to_int() == 3
    assert ti.algorithms.any_if(flags, workspace=workspace).to_bool()
    assert not ti.algorithms.all_if(flags, workspace=workspace).to_bool()

    flags.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    assert ti.algorithms.all_if(flags, workspace=workspace).to_bool()
    assert workspace.workspace_bytes_peak >= 4


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_check_floating_values():
    _skip_if_native_check_unavailable()
    values = ti.ndarray(ti.f32, shape=6)
    values.from_numpy(
        np.array([0.0, np.nan, np.inf, -np.inf, 4.0, -5.0], dtype=np.float32)
    )
    workspace = ti.algorithms.CheckWorkspace(max_items=6)

    assert ti.algorithms.nan_count(values, workspace=workspace).to_int() == 1
    assert ti.algorithms.inf_count(values, workspace=workspace).to_int() == 2
    finite = ti.algorithms.all_finite(values, workspace=workspace)
    assert not finite.to_bool()
    assert not finite.ok()

    values.from_numpy(np.array([0.0, 1.0, -2.0, 3.5, 4.0, -5.0], dtype=np.float32))
    finite = ti.algorithms.all_finite(values, workspace=workspace)
    assert finite.to_bool()
    assert finite.ok()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_check_index_bounds():
    _skip_if_native_check_unavailable()
    indices = ti.ndarray(ti.i32, shape=5)
    indices.from_numpy(np.array([-1, 0, 2, 4, 7], dtype=np.int32))
    workspace = ti.algorithms.CheckWorkspace(max_items=5)

    invalid = ti.algorithms.index_bounds_check(
        indices, lower=0, upper=4, workspace=workspace
    )
    assert invalid.to_int() == 3
    assert not invalid.ok()

    indices.from_numpy(np.array([0, 1, 2, 3, 0], dtype=np.int32))
    invalid = ti.algorithms.index_bounds_check(
        indices, lower=0, upper=4, workspace=workspace
    )
    assert invalid.to_int() == 0
    assert invalid.ok()
