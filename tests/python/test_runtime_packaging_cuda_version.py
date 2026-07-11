import os
from pathlib import Path
import re
from zipfile import ZipFile

import pytest

from scripts import repair_runtime_wheel
from scripts import validate_installed_runtime
from scripts import validate_runtime_wheel
from taichi_forge._lib import utils as runtime_utils


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_runtime_wheel(
    wheel: Path,
    *,
    platform: str,
    version: str,
    cuda_major: int,
    extra_cudart_major: int | None = None,
) -> None:
    dist_info = f"taichi_forge_runtime-{version}.dist-info"
    native = "taichi_forge_runtime/_lib/runtime_native"
    if platform == "windows":
        runtime_name = "taichi_runtime.dll"
        cudart_name = f"cudart64_{cuda_major}.dll"
    else:
        runtime_name = "libtaichi_runtime.so"
        cudart_name = f"libcudart-deadbeef.so.{cuda_major}"
    with ZipFile(wheel, "w") as zf:
        zf.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.1\nName: taichi-forge-runtime\nVersion: {version}\n",
        )
        zf.writestr(f"{dist_info}/RECORD", "")
        zf.writestr(f"{native}/cuda_runtime_major.txt", f"{cuda_major}\n")
        zf.writestr(f"{native}/{runtime_name}", b"runtime")
        zf.writestr(f"{native}/{cudart_name}", b"cudart")
        if platform == "windows":
            zf.writestr(f"{native}/taichi_runtime.lib", b"import library")
        if extra_cudart_major is not None:
            if platform == "windows":
                extra_name = f"cudart64_{extra_cudart_major}.dll"
            else:
                extra_name = f"libcudart-other.so.{extra_cudart_major}"
            zf.writestr(f"{native}/{extra_name}", b"stale cudart")


def test_runtime_repair_discovers_versioned_windows_cudart(tmp_path):
    runtime = tmp_path / "taichi_runtime.dll"
    runtime.write_bytes(b"prefix\0cudart64_11.dll\0suffix")

    assert repair_runtime_wheel._windows_cudart_name(runtime) == "cudart64_11.dll"


def test_runtime_repair_discovers_versioned_linux_cudart(tmp_path):
    runtime = tmp_path / "libtaichi_runtime.so"
    runtime.write_bytes(b"libcudart.so.12\0libcudart.so.12.8.90\0")

    assert (
        repair_runtime_wheel._linux_cudart_name(runtime)
        == "libcudart.so.12.8.90"
    )


@pytest.mark.parametrize(
    ("system", "name", "major"),
    [
        ("windows", "cudart64_11.dll", 11),
        ("windows", "cudart64_12.dll", 12),
        ("linux", "libcudart.so.12.8.90", 12),
    ],
)
def test_runtime_repair_derives_cuda_runtime_major(system, name, major):
    assert repair_runtime_wheel._cuda_runtime_major_from_name(system, name) == major


def test_runtime_repair_uses_only_selected_build_cache(tmp_path):
    old = tmp_path / "old"
    current = tmp_path / "current"
    source = tmp_path / "source"
    runtime_output = source / "runtimes"
    old.mkdir()
    current.mkdir()
    runtime_output.mkdir(parents=True)
    (old / "CMakeCache.txt").write_text(
        "CUDAToolkit_VERSION_MAJOR:STRING=13\n", encoding="utf-8"
    )
    (current / "CMakeCache.txt").write_text(
        "CUDAToolkit_VERSION_MAJOR:STRING=11\n"
        f"CMAKE_HOME_DIRECTORY:INTERNAL={source.as_posix()}\n",
        encoding="utf-8",
    )

    assert repair_runtime_wheel._artifact_roots(current, "windows") == [
        runtime_output
    ]
    assert repair_runtime_wheel._artifact_roots(current, "linux") == [current]
    assert repair_runtime_wheel._cmake_cache_values([current]) == {
        "CUDAToolkit_VERSION_MAJOR": "11",
        "CMAKE_HOME_DIRECTORY": source.as_posix(),
    }


@pytest.mark.parametrize(
    ("system", "name", "major"),
    [
        ("Windows", "cudart64_11.dll", 11),
        ("Windows", "cudart64_13.dll", 13),
        ("Linux", "libcudart.so.12", 12),
        ("Linux", "libcudart.so.12.8.90", 12),
        ("Linux", "libcudart-deadbeef.so.13", 13),
    ],
)
def test_installed_runtime_validator_derives_cudart_major(
    monkeypatch, system, name, major
):
    monkeypatch.setattr(validate_installed_runtime.platform, "system", lambda: system)

    assert validate_installed_runtime._packaged_cuda_runtime_major(Path(name)) == major


def test_installed_runtime_validator_rejects_unversioned_cudart(monkeypatch):
    monkeypatch.setattr(
        validate_installed_runtime.platform, "system", lambda: "Linux"
    )

    with pytest.raises(RuntimeError, match="unrecognized bundled CUDART"):
        validate_installed_runtime._packaged_cuda_runtime_major(
            Path("libcudart.so")
        )


def test_shared_wheel_validator_accepts_windows_and_manylinux_pair(tmp_path):
    windows = tmp_path / "taichi_forge_runtime-0.4.2-py3-none-win_amd64.whl"
    linux = (
        tmp_path
        / "taichi_forge_runtime-0.4.2-py3-none-manylinux_2_35_x86_64.whl"
    )
    _write_runtime_wheel(
        windows, platform="windows", version="0.4.2", cuda_major=12
    )
    _write_runtime_wheel(
        linux, platform="manylinux", version="0.4.2", cuda_major=12
    )

    infos = validate_runtime_wheel.validate_runtime_wheels(
        tmp_path, "pair", expected_cuda_major=12
    )

    assert {info.platform for info in infos} == {"windows", "manylinux"}


def test_shared_wheel_validator_rejects_stale_second_cudart(tmp_path):
    wheel = tmp_path / "taichi_forge_runtime-0.4.2-py3-none-win_amd64.whl"
    _write_runtime_wheel(
        wheel,
        platform="windows",
        version="0.4.2",
        cuda_major=12,
        extra_cudart_major=11,
    )

    with pytest.raises(RuntimeError, match="Expected one CUDART"):
        validate_runtime_wheel.inspect_runtime_wheel(wheel)


def test_shared_wheel_validator_rejects_pair_with_different_majors(tmp_path):
    windows = tmp_path / "taichi_forge_runtime-0.4.2-py3-none-win_amd64.whl"
    linux = (
        tmp_path
        / "taichi_forge_runtime-0.4.2-py3-none-manylinux_2_35_x86_64.whl"
    )
    _write_runtime_wheel(
        windows, platform="windows", version="0.4.2", cuda_major=12
    )
    _write_runtime_wheel(
        linux, platform="manylinux", version="0.4.2", cuda_major=13
    )

    with pytest.raises(RuntimeError, match="CUDART majors differ"):
        validate_runtime_wheel.validate_runtime_wheels(tmp_path, "pair")


def test_shared_wheel_validator_rejects_cuda_versioned_release(tmp_path):
    wheel = tmp_path / "taichi_forge_runtime-0.4.2+cu12-py3-none-win_amd64.whl"
    _write_runtime_wheel(
        wheel, platform="windows", version="0.4.2+cu12", cuda_major=12
    )

    with pytest.raises(RuntimeError, match="CUDA-versioned runtime wheel"):
        validate_runtime_wheel.inspect_runtime_wheel(wheel)


@pytest.mark.parametrize(
    ("system", "library_name", "major"),
    [
        ("win", "cudart64_11.dll", 11),
        ("win", "cudart64_12.dll", 12),
        ("linux", "libcudart-deadbeef.so.12", 12),
        ("linux", "libcudart.so.13.2.0", 13),
    ],
)
def test_shim_uses_single_runtime_manifest_major(
    monkeypatch, tmp_path, system, library_name, major
):
    (tmp_path / "cuda_runtime_major.txt").write_text(
        f"{major}\n", encoding="ascii"
    )
    runtime = tmp_path / library_name
    runtime.write_bytes(b"")
    monkeypatch.setattr(runtime_utils, "get_os_name", lambda: system)
    path_var = "TI_CUDA_CUB_SORT_BUNDLED_CUDART_PATH"
    major_var = "TI_CUDA_CUB_SORT_BUNDLED_CUDART_MAJOR"
    monkeypatch.setenv(path_var, os.environ.get(path_var, ""))
    monkeypatch.setenv(major_var, os.environ.get(major_var, ""))

    runtime_utils._prepare_bundled_cuda_runtime([str(tmp_path)])

    assert Path(os.environ[path_var]) == runtime
    assert os.environ[major_var] == str(major)


def test_shim_rejects_conflicting_runtime_manifests(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "cuda_runtime_major.txt").write_text("11\n", encoding="ascii")
    (second / "cuda_runtime_major.txt").write_text("12\n", encoding="ascii")
    monkeypatch.setattr(runtime_utils, "get_os_name", lambda: "win")

    with pytest.raises(RuntimeError, match="Conflicting bundled CUDA runtime"):
        runtime_utils._bundled_cuda_runtime_major([str(first), str(second)])


def test_shim_rejects_library_major_conflicting_with_manifest(
    monkeypatch, tmp_path
):
    (tmp_path / "cuda_runtime_major.txt").write_text("12\n", encoding="ascii")
    (tmp_path / "cudart64_11.dll").write_bytes(b"")
    monkeypatch.setattr(runtime_utils, "get_os_name", lambda: "win")

    with pytest.raises(RuntimeError, match="conflict with manifest major 12"):
        runtime_utils._bundled_cuda_runtime_major([str(tmp_path)])


def test_runtime_distribution_is_platform_only_not_cuda_versioned():
    runtime_pyproject = (
        REPO_ROOT / "packaging" / "runtime" / "pyproject.toml"
    ).read_text(encoding="utf-8")
    root_pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    release_version = (
        (REPO_ROOT / "version.txt")
        .read_text(encoding="utf-8")
        .strip()
        .removeprefix("v")
    )

    assert re.search(
        r'^name\s*=\s*"taichi-forge-runtime"\s*$',
        runtime_pyproject,
        re.MULTILINE,
    )
    assert re.search(
        r'^wheel\.py-api\s*=\s*"py3"\s*$',
        runtime_pyproject,
        re.MULTILINE,
    )
    runtime_dependencies = re.findall(
        r'^\s*"(taichi-forge-runtime[^"\n]*)",?\s*$',
        root_pyproject,
        re.MULTILINE,
    )
    assert runtime_dependencies == [
        f"taichi-forge-runtime=={release_version}"
    ]
    assert not re.search(
        r"taichi[-_]forge[-_]runtime[-_](?:cu|cuda)\d+",
        runtime_pyproject + root_pyproject,
        re.IGNORECASE,
    )


def test_runtime_publish_workflow_has_no_cuda_wheel_matrix():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "publish_runtime_pypi.yml"
    ).read_text(encoding="utf-8")

    assert "build_linux_runtime:" in workflow
    assert "build_windows_runtime:" in workflow
    assert "matrix:" not in workflow
    assert workflow.count("scripts/validate_runtime_wheel.py") == 4
    assert "--wheel-dir dist-runtime --platform linux" in workflow
    assert "--wheel-dir wheelhouse-runtime --platform manylinux" in workflow
    assert "--wheel-dir dist-runtime --platform windows" in workflow
    assert "--wheel-dir dist --platform pair" in workflow
    assert not re.search(
        r"taichi[-_]forge[-_]runtime[-_](?:cu|cuda)\d+",
        workflow,
        re.IGNORECASE,
    )
