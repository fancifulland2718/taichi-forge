from dataclasses import FrozenInstanceError

import pytest

from taichi_forge.lang import _kernel_optimization as kernel_optimization
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


def test_kernel_optimization_spec_caches_immutable_payloads_and_identities(
    monkeypatch,
):
    spec = _KernelOptimizationSpec(
        ir=_IrOptimizationOptions(thread_local="off", compile_tier="full"),
        backend=_BackendCodegenOptions(workgroup_size=256, cuda_min_blocks_per_sm=1),
        artifact=_ArtifactOptions(cuda_max_registers=64),
        launch=_LaunchOptions(
            block_mode="require",
            grid_residency_waves=2,
            range_work_per_thread_target=4,
        ),
    )
    expected = (
        spec.stable_payload,
        spec.compilation_payload,
        spec.identity,
        spec.compilation_identity,
    )

    def unexpected_asdict(_):
        raise AssertionError("immutable optimization payload was recomputed")

    monkeypatch.setattr(kernel_optimization, "asdict", unexpected_asdict)
    for _ in range(100):
        assert (
            spec.stable_payload,
            spec.compilation_payload,
            spec.identity,
            spec.compilation_identity,
        ) == expected


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
        _KernelOptimizationSpec(
            backend=_BackendCodegenOptions(workgroup_size=128),
            launch=_LaunchOptions(block_mode="hint", range_work_per_thread_target=4),
        ),
        _KernelOptimizationSpec(
            ir=_IrOptimizationOptions(compile_tier="full"),
            backend=_BackendCodegenOptions(workgroup_size=128),
            launch=_LaunchOptions(block_mode="hint"),
        ),
        _KernelOptimizationSpec(
            backend=_BackendCodegenOptions(
                workgroup_size=128, cuda_min_blocks_per_sm=1
            ),
            launch=_LaunchOptions(block_mode="hint"),
        ),
        _KernelOptimizationSpec(
            artifact=_ArtifactOptions(cuda_max_registers=64),
            backend=_BackendCodegenOptions(workgroup_size=128),
            launch=_LaunchOptions(block_mode="hint"),
        ),
    )
    assert len({baseline.identity, *(variant.identity for variant in variants)}) == 10


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


def test_range_work_per_thread_has_launch_identity_only():
    variants = tuple(
        _KernelOptimizationSpec(
            backend=_BackendCodegenOptions(workgroup_size=256),
            launch=_LaunchOptions(
                block_mode="require", range_work_per_thread_target=target
            ),
        )
        for target in (1, 2, 4, 8)
    )
    assert len({variant.identity for variant in variants}) == 4
    assert len({variant.compilation_identity for variant in variants}) == 1


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
        lambda: _LaunchOptions(range_work_per_thread_target=3),
        lambda: _IrOptimizationOptions(thread_local="sometimes"),
        lambda: _IrOptimizationOptions(compile_tier="aggressive"),
        lambda: _BackendCodegenOptions(cuda_min_blocks_per_sm=3),
        lambda: _ArtifactOptions(cuda_max_registers=8),
        lambda: _ArtifactOptions(provider_mode="recursive"),
        # The old backend-neutral spelling remains unavailable; CUDA uses the
        # explicit cuda_max_registers artifact contract above.
        lambda: _ArtifactOptions(max_registers=64),
    ),
)
def test_kernel_optimization_spec_rejects_invalid_contracts(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()
