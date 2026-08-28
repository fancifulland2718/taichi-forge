from dataclasses import FrozenInstanceError

import pytest

from taichi_forge.lang._kernel_optimization import (
    _ArtifactOptions,
    _BackendCodegenOptions,
    _IrOptimizationOptions,
    _KernelOptimizationSpec,
    _LaunchOptions,
    _bind_kernel_optimization_spec,
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


def test_grid_residency_has_full_identity_but_shares_compilation_identity():
    variants = tuple(
        _KernelOptimizationSpec(
            backend=_BackendCodegenOptions(workgroup_size=256),
            launch=_LaunchOptions(block_mode="require", grid_residency_waves=waves),
        )
        for waves in (1, 2, 4)
    )
    assert len({variant.identity for variant in variants}) == 3
    assert len({variant.compilation_identity for variant in variants}) == 1
    assert len({variant.compilation_specialization_key for variant in variants}) == 1
    assert all(variant.compilation_identity.startswith("kos1:") for variant in variants)


def test_private_binding_rejects_recursive_provider_tuning():
    spec = _KernelOptimizationSpec(
        backend=_BackendCodegenOptions(workgroup_size=256),
        artifact=_ArtifactOptions(provider_mode="request_tuning"),
        launch=_LaunchOptions(block_mode="require"),
    )
    with pytest.raises(ValueError, match="recursively request"):
        _bind_kernel_optimization_spec(object(), spec)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: _KernelOptimizationSpec(
            backend=_BackendCodegenOptions(workgroup_size=128)
        ),
        lambda: _LaunchOptions(grid_residency_waves=3),
        lambda: _IrOptimizationOptions(thread_local="sometimes"),
        lambda: _ArtifactOptions(provider_mode="recursive"),
        # A per-kernel register cap is intentionally unavailable until it has
        # physical lowering; identity-only options must fail closed.
        lambda: _ArtifactOptions(max_registers=64),
    ),
)
def test_kernel_optimization_spec_rejects_invalid_contracts(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()
