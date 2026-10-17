import hashlib
import json

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@pytest.mark.parametrize("validate", [False, True])
@test_utils.test(arch=ti.vulkan, offline_cache=False, compile_tier="fast")
def test_compile_profile_separates_pipeline_preparation(validate, monkeypatch):
    monkeypatch.delenv("TI_VULKAN_DIAGNOSTICS_DIR", raising=False)
    monkeypatch.setenv("TI_VULKAN_VALIDATE_SPIRV", "1" if validate else "0")
    values = ti.ndarray(ti.i32, 37)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(out.shape[0]):
            out[i] = i * i + 7

    with ti.compile_profile() as profile:
        fill(values)
        ti.sync()
    paths = [row["path"].split("/")[-1] for row in profile.records(include_python=False)]
    assert "cpp.compile.spirv.emit_task" in paths
    assert "cpp.vulkan.descriptor_reflection_and_layout" in paths
    assert "cpp.vulkan.create_shader_modules" in paths
    assert "cpp.vulkan.driver_create_compute_pipeline" in paths
    assert ("cpp.vulkan.validate_spirv" in paths) == validate
    assert "cpp.vulkan.diagnostic_reflection" not in paths
    np.testing.assert_array_equal(values.to_numpy(), np.arange(37, dtype=np.int32) ** 2 + 7)


@test_utils.test(arch=ti.vulkan, offline_cache=False, compile_tier="balanced")
def test_export_matches_actual_jit_pipeline_without_submission(tmp_path, monkeypatch):
    captures = tmp_path / "pipelines"
    monkeypatch.setenv("TI_VULKAN_DIAGNOSTICS_DIR", str(captures))
    monkeypatch.setenv("TI_VULKAN_VALIDATE_SPIRV", "1")
    values = ti.ndarray(ti.i32, 137)

    @ti.kernel(opt_level="fast")
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(out.shape[0]):
            if i % 3 == 0:
                continue
            out[i] = i + 11

    values.fill(-1)
    ti.sync()
    before = impl.get_runtime().prog._runtime_statistics_snapshot()["submission"]
    manifest = ti.tools.export_vulkan_kernel(
        fill, values, directory=tmp_path / "jit", validate=True
    )
    after = impl.get_runtime().prog._runtime_statistics_snapshot()["submission"]
    assert before == after
    assert not manifest["pipeline_registered"]
    assert manifest["resolved_config"]["compile_tier"] == "fast"
    assert manifest["metadata"]["kernel_attribs"]["tasks_attribs"]
    np.testing.assert_array_equal(values.to_numpy(), -np.ones(137, dtype=np.int32))
    for shader in manifest["shaders"]:
        binary = (tmp_path / "jit" / shader["file"]).read_bytes()
        assert hashlib.sha256(binary).hexdigest() == shader["sha256"]
        assert shader["validation"]["returncode"] == 0
    fill(values)
    ti.sync()
    expected = np.arange(137, dtype=np.int32) + 11
    expected[::3] = -1
    np.testing.assert_array_equal(values.to_numpy(), expected)
    pipelines = [
        json.loads(path.read_text()) for path in captures.glob("*/pipeline.json")
    ]
    exported = {item["sha256"] for item in manifest["shaders"]}
    matched = [item for item in pipelines if item["shaders"][0]["sha256"] in exported]
    assert len(matched) == len(exported)
    for item in matched:
        assert item["shaders"][0]["validation_passed"]
        assert item["shaders"][0]["descriptor_bindings"]
        assert item["shaders"][0]["local_size"][0] > 0
        assert item["specialization_constants"] == []
    for path in captures.glob("*/events.jsonl"):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert events[-2]["phase"] == "pipeline_call_started"
        assert events[-2]["vk_result"] is None
        assert events[-1]["vk_result"] == 0
    with pytest.raises(FileExistsError):
        ti.tools.export_vulkan_kernel(fill, values, directory=tmp_path / "jit")


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_pipeline_cache_policy_is_captured_per_runtime(tmp_path, monkeypatch):
    path = tmp_path / "独立缓存" / "pipeline.bin"
    redirect = tmp_path / "should-not-exist.bin"
    ti.reset()
    monkeypatch.setenv("TI_VULKAN_PIPELINE_CACHE_PATH", str(path))
    monkeypatch.setenv("TI_VULKAN_PIPELINE_CACHE", "1")
    ti.init(arch=ti.vulkan, offline_cache=False)
    data = ti.tools.vulkan_runtime_diagnostics()
    assert data["pipeline_cache"]["load_status"] == "not_found"
    assert data["device"]["device_uuid"].strip("0")
    values = ti.ndarray(ti.i32, 17)
    values.fill(7)
    ti.sync()
    monkeypatch.setenv("TI_VULKAN_PIPELINE_CACHE_PATH", str(redirect))
    ti.reset()
    assert path.is_file() and path.stat().st_size > 0
    assert not redirect.exists()
    binary = path.read_bytes()
    monkeypatch.setenv("TI_VULKAN_PIPELINE_CACHE_PATH", str(path))
    ti.init(arch=ti.vulkan, offline_cache=False)
    cache = ti.tools.vulkan_runtime_diagnostics()["pipeline_cache"]
    assert cache["load_status"] == "supplied_to_backend"
    assert cache["input_sha256"] == hashlib.sha256(binary).hexdigest()
    ti.reset()
    baseline = path.read_bytes()
    monkeypatch.setenv("TI_VULKAN_PIPELINE_CACHE", "0")
    ti.init(arch=ti.vulkan, offline_cache=False)
    cache = ti.tools.vulkan_runtime_diagnostics()["pipeline_cache"]
    assert cache["load_status"] == "disabled"
    ti.ndarray(ti.i32, 17).fill(9)
    ti.reset()
    assert path.read_bytes() == baseline


@test_utils.test(arch=ti.cpu)
def test_vulkan_diagnostics_reject_non_vulkan():
    with pytest.raises(RuntimeError, match="Vulkan diagnostics require"):
        ti.tools.vulkan_runtime_diagnostics()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_export_bound_kernel_and_retain_validation_failure(tmp_path, monkeypatch):
    from taichi_forge.tools import vulkan_diagnostics
    from types import SimpleNamespace

    @ti.data_oriented
    class Owner:
        def __init__(self):
            self.values = ti.field(ti.i32, shape=17)

        @ti.kernel
        def fill(self):
            for i in self.values:
                self.values[i] = i + 1

    owner = Owner()
    first = ti.tools.export_vulkan_kernel(owner.fill, directory=tmp_path / "bound")
    assert first["shaders"]
    monkeypatch.setattr(vulkan_diagnostics.shutil, "which", lambda _: "spirv-val")
    monkeypatch.setattr(
        vulkan_diagnostics.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="invalid test artifact"
        ),
    )
    with pytest.raises(RuntimeError, match="artifacts preserved"):
        ti.tools.export_vulkan_kernel(
            owner.fill, directory=tmp_path / "failed", validate=True
        )
    failed = json.loads((tmp_path / "failed" / "manifest.json").read_text())
    assert failed["shaders"][0]["validation"]["returncode"] == 1
    assert (tmp_path / "failed" / "shader-0.spv").read_bytes() == (
        tmp_path / "bound" / "shader-0.spv"
    ).read_bytes()
    np.testing.assert_array_equal(owner.values.to_numpy(), np.zeros(17, dtype=np.int32))
