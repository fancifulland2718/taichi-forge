from dataclasses import replace
import json

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from taichi_forge.lang import _task_launch_tuning as tuning
from tests import test_utils


def _profile():
    return tuning._TaskLaunchWorkloadProfile(
        workload_id="test:vulkan-update",
        input_distribution_id="ascending-v1",
        shape=(4099,),
        ir_identity="test:update-v1",
        spirv_identity="test:spirv-update-v1",
        oracle_identity="numpy-exact-v1",
        replay_identity="overwrite-output-v1",
    )


def _evidence():
    # Synthetic admission evidence tests the protocol, not a speedup claim.
    return dict(
        correctness_passed=True,
        independent_abba_blocks=5,
        worst_candidate_over_baseline_ratio=0.9,
        cost_metric="completion_time",
        completion_waited=True,
    )


@pytest.mark.parametrize("block", [32, 64, 128])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_exact_qualified_record_is_consumed(tmp_path, monkeypatch, block):
    monkeypatch.setattr(tuning, "_cache_root_from_config", lambda config: tmp_path)
    values = ti.ndarray(ti.i32, 4099)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(out.shape[0]):
            out[i] = i * 3 + 5

    binding = tuning._bind_workload_profile(
        fill.with_launch_policy(ti.TaskLaunchPolicy.auto()), _profile()
    )
    before = impl.get_runtime().prog._runtime_statistics_snapshot()["submission"]
    assert binding.report(values).status == "auto"
    assert (
        impl.get_runtime().prog._runtime_statistics_snapshot()["submission"] == before
    )
    decision = binding._last_auto_decision
    assert block in decision.candidates
    scope = json.loads(decision.record_scope_json)
    assert scope["hardware"]["backend"] == "vulkan"
    assert scope["hardware"]["device_uuid"].strip("0")
    assert scope["hardware"]["subgroup_size"] > 0
    assert all(
        len(code) == 64 for code in scope["kernel"]["compiled_artifact_identity"]
    )
    assert "ptx_identity" not in scope["kernel"]
    for bad in (dict(cost_metric="enqueue_time"), dict(completion_waited=False)):
        tuning._publish_qualified_record(
            decision=decision,
            cache_root=tmp_path,
            block_dim=block,
            evidence={**_evidence(), **bad},
        )
        assert binding.report(values).status == "auto"
    path = tuning._publish_qualified_record(
        decision=decision, cache_root=tmp_path, block_dim=block, evidence=_evidence()
    )
    report = binding.report(values)
    assert report.status == "auto_qualified"
    task = next(task for task in report.tasks if task.task_type == "range_for")
    assert task.actual_block_size == block
    binding(values)
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(4099, dtype=np.int32) * 3 + 5
    )
    # Device, driver, shader and workload changes cannot reuse this record.
    assert tuning._validated_record(path, scope, decision.record_id)
    for section, key, value in (
        ("hardware", "device_uuid", "f" * 32),
        ("hardware", "driver_version", -1),
        ("hardware", "subgroup_size", 1),
        ("kernel", "compiled_artifact_identity", ["0" * 64]),
        ("workload", "input_distribution_id", "changed"),
        ("workload", "topology_stability", "dynamic"),
    ):
        changed = json.loads(decision.record_scope_json)
        changed[section][key] = value
        assert tuning._validated_record(path, changed, decision.record_id) is None


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_shared_workgroup_is_not_rewritten(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "_cache_root_from_config", lambda config: tmp_path)
    values = ti.ndarray(ti.i32, 256)

    @ti.kernel
    def shared(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        ti.loop_config(block_dim=64)
        for i in range(256):
            scratch = ti.simt.block.SharedArray((64,), ti.i32)
            scratch[i % 64] = i + 5
            ti.simt.block.sync()
            out[i] = scratch[i % 64]

    binding = tuning._bind_workload_profile(
        shared.with_launch_policy(ti.TaskLaunchPolicy.auto()), _profile()
    )
    assert binding.report(values).status == "auto"
    assert set(binding._last_auto_decision.candidates) <= {64}
    binding(values)
    np.testing.assert_array_equal(values.to_numpy(), np.arange(256, dtype=np.int32) + 5)


def test_cuda_profile_identity_stays_compatible():
    profile = replace(_profile(), spirv_identity="", ptx_identity="ptx-v1")
    assert "spirv_identity" not in profile.stable_scope
    with pytest.raises(ValueError, match="ptx_identity or spirv_identity"):
        replace(profile, ptx_identity="")


@pytest.mark.parametrize("entry", ["report", "call"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_rejected_specialization_keeps_reason(tmp_path, monkeypatch, entry):
    from taichi_forge.lang.exception import TaichiRuntimeError

    monkeypatch.setattr(tuning, "_cache_root_from_config", lambda config: tmp_path)
    values = ti.ndarray(ti.i32, 4099)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(out.shape[0]):
            out[i] = i + 7

    binding = tuning._bind_workload_profile(
        fill.with_launch_policy(ti.TaskLaunchPolicy.auto()), _profile()
    )
    binding.report(values)
    tuning._publish_qualified_record(
        decision=binding._last_auto_decision,
        cache_root=tmp_path,
        block_dim=32,
        evidence=_evidence(),
    )
    method = (
        "_task_launch_report"
        if entry == "report"
        else "_ensure_compiled_with_task_launch_policy"
    )
    original = getattr(binding._kernel, method)
    rejected = []

    def reject(policy, *args, **kwargs):
        if policy.mode == "require":
            rejected.append(True)
            raise TaichiRuntimeError("test specialization resource rejection")
        return original(policy, *args, **kwargs)

    monkeypatch.setattr(binding._kernel, method, reject)
    if entry == "report":
        binding.report(values)
    else:
        binding(values)
    for _ in range(2):
        report = binding.report(values)
        assert report.status == "auto"
        assert "test specialization resource rejection" in report.reason
        binding(values)
    assert rejected == [True]
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(4099, dtype=np.int32) + 7
    )
