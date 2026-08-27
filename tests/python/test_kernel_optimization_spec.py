from dataclasses import FrozenInstanceError

import pytest

from taichi_forge.lang._kernel_optimization import (
    _ArtifactOptions,
    _BackendCodegenOptions,
    _IrOptimizationOptions,
    _KernelOptimizationSpec,
    _LaunchOptions,
)
from taichi_forge.lang.task_launch import TaskLaunchPolicy


def test_kernel_optimization_spec_is_immutable_and_deterministic():
    baseline = _KernelOptimizationSpec()
    assert baseline.is_baseline
    assert baseline.identity == ""

    first = _KernelOptimizationSpec.from_task_launch_policy(
        TaskLaunchPolicy.block(256, mode="require")
    )
    second = _KernelOptimizationSpec(
        backend=_BackendCodegenOptions(workgroup_size=256),
        launch=_LaunchOptions(block_mode="require"),
    )
    assert first == second
    assert first.identity == second.identity
    assert first.identity.startswith("kos1:")
    assert first.specialization_key == second.specialization_key
    with pytest.raises(FrozenInstanceError):
        first.launch = _LaunchOptions()


def test_kernel_optimization_spec_separates_all_bounded_axes():
    baseline = _KernelOptimizationSpec.from_task_launch_policy(
        TaskLaunchPolicy.block(128)
    )
    variants = (
        _KernelOptimizationSpec(
            backend=_BackendCodegenOptions(workgroup_size=256),
            launch=_LaunchOptions(block_mode="hint"),
        ),
        _KernelOptimizationSpec(
            backend=_BackendCodegenOptions(workgroup_size=128),
            launch=_LaunchOptions(block_mode="require"),
        ),
        _KernelOptimizationSpec(
            ir=_IrOptimizationOptions(thread_local="off"),
            backend=_BackendCodegenOptions(workgroup_size=128),
            launch=_LaunchOptions(block_mode="hint"),
        ),
        _KernelOptimizationSpec(
            artifact=_ArtifactOptions(provider_mode="apply_explicit_acf"),
            backend=_BackendCodegenOptions(workgroup_size=128),
            launch=_LaunchOptions(block_mode="hint"),
        ),
        _KernelOptimizationSpec(
            backend=_BackendCodegenOptions(workgroup_size=128),
            launch=_LaunchOptions(block_mode="hint", grid_residency_waves=2),
        ),
    )
    assert len({baseline.identity, *(variant.identity for variant in variants)}) == 6


@pytest.mark.parametrize(
    "factory",
    (
        lambda: _KernelOptimizationSpec(
            backend=_BackendCodegenOptions(workgroup_size=128)
        ),
        lambda: _LaunchOptions(grid_residency_waves=3),
        lambda: _IrOptimizationOptions(thread_local="sometimes"),
        lambda: _ArtifactOptions(provider_mode="recursive"),
        lambda: _ArtifactOptions(max_registers=0),
    ),
)
def test_kernel_optimization_spec_rejects_invalid_contracts(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()
