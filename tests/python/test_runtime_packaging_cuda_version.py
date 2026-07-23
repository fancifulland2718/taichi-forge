import os
from pathlib import Path
import re
from zipfile import ZipFile

import pytest

from scripts import repair_runtime_wheel
from scripts import validate_installed_runtime
from scripts import validate_runtime_wheel
from scripts import validate_shim_wheel
from taichi_forge._lib import utils as runtime_utils


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_runtime_wheel(
    wheel: Path,
    *,
    platform: str,
    version: str,
    cuda_major: int,
    extra_cudart_major: int | None = None,
    hashed_runtime: bool = False,
    misplaced_cudart: bool = False,
    auditwheel_layout: bool = False,
    duplicate_raw_cudart: bool = False,
    dependency_class: str = "toolkit-reference",
) -> None:
    dist_info = f"taichi_forge_runtime-{version}.dist-info"
    native = "taichi_forge_runtime/_lib/runtime_native"
    if platform == "windows":
        runtime_name = "taichi_runtime.dll"
        cudart_name = f"cudart64_{cuda_major}.dll"
    else:
        runtime_name = (
            "libtaichi_runtime-deadbeef.so"
            if hashed_runtime
            else "libtaichi_runtime.so"
        )
        cudart_name = (
            f"libcudart-deadbeef.so.{cuda_major}.2.75"
            if auditwheel_layout
            else f"libcudart.so.{cuda_major}"
        )
    with ZipFile(wheel, "w") as zf:
        zf.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.1\nName: taichi-forge-runtime\nVersion: {version}\n",
        )
        zf.writestr(f"{dist_info}/RECORD", "")
        zf.writestr(f"{native}/{runtime_name}", b"runtime")
        if dependency_class == "toolkit-reference":
            zf.writestr(f"{native}/cuda_runtime_major.txt", f"{cuda_major}\n")
            if misplaced_cudart:
                cudart_dir = "unrelated_package"
            elif auditwheel_layout:
                cudart_dir = "taichi_forge_runtime.libs"
            else:
                cudart_dir = native
            zf.writestr(f"{cudart_dir}/{cudart_name}", b"cudart")
            if duplicate_raw_cudart:
                zf.writestr(
                    f"{native}/libcudart.so.{cuda_major}",
                    b"redundant raw cudart",
                )
        elif dependency_class != "driver-only":
            raise ValueError(f"unknown dependency class: {dependency_class}")
        if platform == "windows":
            zf.writestr(f"{native}/taichi_runtime.lib", b"import library")
        if extra_cudart_major is not None:
            if platform == "windows":
                extra_name = f"cudart64_{extra_cudart_major}.dll"
            else:
                extra_name = f"libcudart-other.so.{extra_cudart_major}"
            zf.writestr(f"{native}/{extra_name}", b"stale cudart")


def _write_shim_wheel(
    wheel: Path,
    *,
    platform: str,
    version: str,
    runtime_version: str | None = None,
    duplicate_runtime: bool = False,
    missing_dependency: str | None = None,
    llvm_abi_sentinel: bool = False,
) -> None:
    dist_info = f"taichi_forge-{version}.dist-info"
    extension = "taichi_python.pyd" if platform == "windows" else "taichi_python.so"
    requirement_version = runtime_version or version
    requirements = [
        "numpy>=1.23; python_version < '3.14'",
        "numpy>=2.1; python_version >= '3.14'",
        "colorama",
        "dill",
        "rich",
        f"taichi-forge-runtime=={requirement_version}",
    ]
    if missing_dependency is not None:
        requirements = [
            requirement
            for requirement in requirements
            if validate_shim_wheel._requirement_project(requirement)
            != missing_dependency
        ]
    requirement_metadata = "".join(
        f"Requires-Dist: {requirement}\n" for requirement in requirements
    )
    with ZipFile(wheel, "w") as zf:
        zf.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\n"
            "Name: taichi-forge\n"
            f"Version: {version}\n"
            f"{requirement_metadata}",
        )
        zf.writestr(f"{dist_info}/RECORD", "")
        extension_payload = b"shim"
        if llvm_abi_sentinel:
            extension_payload += b"\0_ZN4llvm24DisableABIBreakingChecksE\0"
        zf.writestr(f"taichi_forge/_lib/core/{extension}", extension_payload)
        if duplicate_runtime:
            zf.writestr(
                "taichi_forge/_lib/runtime/taichi_runtime.dll",
                b"duplicated runtime",
            )


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


def test_installed_runtime_validator_requires_matching_distribution_versions(
    monkeypatch,
):
    versions = {
        "taichi-forge": "1.2.3",
        "taichi-forge-runtime": "1.2.4",
    }
    monkeypatch.setattr(
        validate_installed_runtime.metadata,
        "version",
        versions.__getitem__,
    )

    with pytest.raises(RuntimeError, match="version mismatch"):
        validate_installed_runtime._validate_distribution_versions()


def test_installed_runtime_validator_accepts_driver_only_package(
    monkeypatch, tmp_path
):
    package = tmp_path / "taichi_forge_runtime"
    package.mkdir()
    monkeypatch.setattr(
        validate_installed_runtime, "_runtime_package_dirs", lambda: [package]
    )
    monkeypatch.setattr(
        validate_installed_runtime.platform, "system", lambda: "Linux"
    )
    monkeypatch.delenv("TI_CUDA_CUB_SORT_BUNDLED_CUDART_PATH", raising=False)

    assert validate_installed_runtime._validate_packaged_cuda_runtime() == (
        None,
        None,
    )


def test_installed_runtime_validator_rejects_unannounced_packaged_cudart(
    monkeypatch, tmp_path
):
    package = tmp_path / "taichi_forge_runtime"
    package.mkdir()
    libs = tmp_path / "taichi_forge_runtime.libs"
    libs.mkdir()
    (libs / "libcudart-deadbeef.so.13.2.75").write_bytes(b"")
    monkeypatch.setattr(
        validate_installed_runtime, "_runtime_package_dirs", lambda: [package]
    )
    monkeypatch.setattr(
        validate_installed_runtime.platform, "system", lambda: "Linux"
    )
    monkeypatch.delenv("TI_CUDA_CUB_SORT_BUNDLED_CUDART_PATH", raising=False)

    with pytest.raises(RuntimeError, match="undiscovered CUDART"):
        validate_installed_runtime._validate_packaged_cuda_runtime()


def test_shared_wheel_validator_accepts_windows_and_manylinux_pair(tmp_path):
    windows = tmp_path / "taichi_forge_runtime-0.4.3-py3-none-win_amd64.whl"
    linux = (
        tmp_path
        / "taichi_forge_runtime-0.4.3-py3-none-manylinux_2_35_x86_64.whl"
    )
    _write_runtime_wheel(
        windows, platform="windows", version="0.4.3", cuda_major=12
    )
    _write_runtime_wheel(
        linux,
        platform="manylinux",
        version="0.4.3",
        cuda_major=12,
        hashed_runtime=True,
        auditwheel_layout=True,
    )

    infos = validate_runtime_wheel.validate_runtime_wheels(
        tmp_path, "pair", expected_cuda_major=12
    )

    assert {info.platform for info in infos} == {"windows", "manylinux"}


def test_shared_wheel_validator_accepts_driver_only_pair(tmp_path):
    windows = tmp_path / "taichi_forge_runtime-0.5.1-py3-none-win_amd64.whl"
    linux = (
        tmp_path
        / "taichi_forge_runtime-0.5.1-py3-none-manylinux_2_35_x86_64.whl"
    )
    _write_runtime_wheel(
        windows,
        platform="windows",
        version="0.5.1",
        cuda_major=0,
        dependency_class="driver-only",
    )
    _write_runtime_wheel(
        linux,
        platform="manylinux",
        version="0.5.1",
        cuda_major=0,
        hashed_runtime=True,
        dependency_class="driver-only",
    )

    infos = validate_runtime_wheel.validate_runtime_wheels(
        tmp_path,
        "pair",
        expected_dependency_class="driver-only",
    )

    assert {info.dependency_class for info in infos} == {"driver-only"}
    assert {info.cuda_major for info in infos} == {None}


def test_shared_wheel_validator_rejects_reference_when_driver_only_required(
    tmp_path,
):
    wheel = tmp_path / "taichi_forge_runtime-0.5.1-py3-none-win_amd64.whl"
    _write_runtime_wheel(
        wheel, platform="windows", version="0.5.1", cuda_major=13
    )

    with pytest.raises(RuntimeError, match="dependency class mismatch"):
        validate_runtime_wheel.inspect_runtime_wheel(
            wheel, expected_dependency_class="driver-only"
        )


def test_manylinux_normalizer_accepts_driver_only_wheel(tmp_path):
    wheel = (
        tmp_path
        / "taichi_forge_runtime-0.5.1-py3-none-manylinux_2_35_x86_64.whl"
    )
    _write_runtime_wheel(
        wheel,
        platform="manylinux",
        version="0.5.1",
        cuda_major=0,
        dependency_class="driver-only",
    )

    repair_runtime_wheel.normalize_manylinux_wheel(wheel)

    info = validate_runtime_wheel.inspect_runtime_wheel(
        wheel, expected_dependency_class="driver-only"
    )
    assert info.cuda_major is None


def test_manylinux_normalizer_prunes_raw_cudart_and_rewrites_record(tmp_path):
    wheel = (
        tmp_path
        / "taichi_forge_runtime-0.4.3-py3-none-manylinux_2_35_x86_64.whl"
    )
    _write_runtime_wheel(
        wheel,
        platform="manylinux",
        version="0.4.3",
        cuda_major=13,
        auditwheel_layout=True,
        duplicate_raw_cudart=True,
    )

    with pytest.raises(RuntimeError, match="Expected one CUDART"):
        validate_runtime_wheel.inspect_runtime_wheel(wheel)

    repair_runtime_wheel.normalize_manylinux_wheel(wheel)
    info = validate_runtime_wheel.inspect_runtime_wheel(wheel)

    assert info.cuda_major == 13
    with ZipFile(wheel) as zf:
        names = zf.namelist()
        assert (
            "taichi_forge_runtime/_lib/runtime_native/libcudart.so.13"
            not in names
        )
        hashed = [
            name
            for name in names
            if name.startswith("taichi_forge_runtime.libs/libcudart-")
        ]
        assert len(hashed) == 1
        record_name = next(
            name for name in names if name.endswith(".dist-info/RECORD")
        )
        record = zf.read(record_name).decode("utf-8")
        assert hashed[0] in record
        assert "runtime_native/libcudart.so.13" not in record


def test_shared_wheel_validator_rejects_stale_second_cudart(tmp_path):
    wheel = tmp_path / "taichi_forge_runtime-0.4.3-py3-none-win_amd64.whl"
    _write_runtime_wheel(
        wheel,
        platform="windows",
        version="0.4.3",
        cuda_major=12,
        extra_cudart_major=11,
    )

    with pytest.raises(RuntimeError, match="Expected one CUDART"):
        validate_runtime_wheel.inspect_runtime_wheel(wheel)


def test_shared_wheel_validator_rejects_cudart_outside_runtime_package(tmp_path):
    wheel = tmp_path / "taichi_forge_runtime-0.4.3-py3-none-win_amd64.whl"
    _write_runtime_wheel(
        wheel,
        platform="windows",
        version="0.4.3",
        cuda_major=12,
        misplaced_cudart=True,
    )

    with pytest.raises(RuntimeError, match="outside the runtime package"):
        validate_runtime_wheel.inspect_runtime_wheel(wheel)


def test_shared_wheel_validator_rejects_pair_with_different_majors(tmp_path):
    windows = tmp_path / "taichi_forge_runtime-0.4.3-py3-none-win_amd64.whl"
    linux = (
        tmp_path
        / "taichi_forge_runtime-0.4.3-py3-none-manylinux_2_35_x86_64.whl"
    )
    _write_runtime_wheel(
        windows, platform="windows", version="0.4.3", cuda_major=12
    )
    _write_runtime_wheel(
        linux, platform="manylinux", version="0.4.3", cuda_major=13
    )

    with pytest.raises(RuntimeError, match="CUDART majors differ"):
        validate_runtime_wheel.validate_runtime_wheels(tmp_path, "pair")


def test_shared_wheel_validator_rejects_cuda_versioned_release(tmp_path):
    wheel = tmp_path / "taichi_forge_runtime-0.4.3+cu12-py3-none-win_amd64.whl"
    _write_runtime_wheel(
        wheel, platform="windows", version="0.4.3+cu12", cuda_major=12
    )

    with pytest.raises(RuntimeError, match="CUDA-versioned runtime wheel"):
        validate_runtime_wheel.inspect_runtime_wheel(wheel)


@pytest.mark.parametrize(
    ("platform", "tag"),
    [
        ("windows", "cp310-cp310-win_amd64"),
        ("manylinux", "cp310-cp310-manylinux_2_35_x86_64"),
    ],
)
def test_shim_wheel_validator_accepts_runtime_free_wheel(
    tmp_path, platform, tag
):
    wheel = tmp_path / f"taichi_forge-0.4.3-{tag}.whl"
    _write_shim_wheel(wheel, platform=platform, version="0.4.3")

    assert validate_shim_wheel.validate_shim_wheel(wheel, platform) == "0.4.3"


def test_shim_wheel_validator_rejects_duplicate_runtime(tmp_path):
    wheel = tmp_path / "taichi_forge-0.4.3-cp310-cp310-win_amd64.whl"
    _write_shim_wheel(
        wheel,
        platform="windows",
        version="0.4.3",
        duplicate_runtime=True,
    )

    with pytest.raises(RuntimeError, match="duplicates runtime artifacts"):
        validate_shim_wheel.validate_shim_wheel(wheel, "windows")


def test_shim_wheel_validator_rejects_mismatched_runtime_version(tmp_path):
    wheel = tmp_path / "taichi_forge-0.4.3-cp310-cp310-win_amd64.whl"
    _write_shim_wheel(
        wheel,
        platform="windows",
        version="0.4.3",
        runtime_version="0.4.1",
    )

    with pytest.raises(RuntimeError, match="Expected runtime dependency"):
        validate_shim_wheel.validate_shim_wheel(wheel, "windows")


@pytest.mark.parametrize("dependency", ["colorama", "dill", "numpy", "rich"])
def test_shim_wheel_validator_rejects_missing_python_dependency(
    tmp_path, dependency
):
    wheel = tmp_path / "taichi_forge-0.4.3-cp310-cp310-win_amd64.whl"
    _write_shim_wheel(
        wheel,
        platform="windows",
        version="0.4.3",
        missing_dependency=dependency,
    )

    with pytest.raises(RuntimeError, match=rf"dependencies.*{dependency}"):
        validate_shim_wheel.validate_shim_wheel(wheel, "windows")


def test_manylinux_shim_rejects_llvm_abi_link_sentinel(tmp_path):
    wheel = (
        tmp_path
        / "taichi_forge-0.4.3-cp310-cp310-manylinux_2_35_x86_64.whl"
    )
    _write_shim_wheel(
        wheel,
        platform="manylinux",
        version="0.4.3",
        llvm_abi_sentinel=True,
    )

    with pytest.raises(RuntimeError, match="LLVM ABI link sentinels"):
        validate_shim_wheel.validate_shim_wheel(wheel, "manylinux")


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


def test_shim_discovers_auditwheel_cudart_separate_from_manifest(
    monkeypatch, tmp_path
):
    native = tmp_path / "taichi_forge_runtime" / "_lib" / "runtime_native"
    auditwheel_libs = tmp_path / "taichi_forge_runtime.libs"
    native.mkdir(parents=True)
    auditwheel_libs.mkdir()
    (native / "cuda_runtime_major.txt").write_text("13\n", encoding="ascii")
    cudart = auditwheel_libs / "libcudart-deadbeef.so.13.2.75"
    cudart.write_bytes(b"")
    monkeypatch.setattr(runtime_utils, "get_os_name", lambda: "linux")

    runtime_utils._prepare_bundled_cuda_runtime(
        [str(native), str(auditwheel_libs)]
    )

    assert Path(
        os.environ["TI_CUDA_CUB_SORT_BUNDLED_CUDART_PATH"]
    ) == cudart
    assert os.environ["TI_CUDA_CUB_SORT_BUNDLED_CUDART_MAJOR"] == "13"


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


def test_prebuilt_shim_configures_libdevice_version_without_installing_assets():
    cmake = (REPO_ROOT / "cmake" / "TaichiCore.cmake").read_text(
        encoding="utf-8"
    )
    guarded_runtime_assets = re.search(
        r"if\s*\(NOT TI_WITH_PREBUILT_PYTHON_RUNTIME\)"
        r".*install\(FILES[^)]*_ti_cuda_libdevice_filename[^)]*"
        r"COMPONENT runtime\)",
        cmake,
        re.DOTALL,
    )

    version_discovery = cmake.index("file(GLOB _ti_cuda_libdevice_files")
    prebuilt_source_guard = cmake.index(
        "if(NOT TI_WITH_PREBUILT_PYTHON_RUNTIME)", version_discovery
    )
    assert version_discovery < prebuilt_source_guard
    assert guarded_runtime_assets


def test_prebuilt_linux_shim_disables_llvm_abi_link_sentinel():
    cmake = (REPO_ROOT / "cmake" / "TaichiCore.cmake").read_text(
        encoding="utf-8"
    )
    prebuilt_llvm = cmake[cmake.index("if(TI_WITH_PREBUILT_PYTHON_RUNTIME)") :]

    assert "if(LINUX)" in prebuilt_llvm
    assert (
        "PRIVATE LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1"
        in prebuilt_llvm
    )


def test_shim_publish_workflow_validates_wheel_boundaries():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "publish_pypi.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count("scripts/validate_runtime_wheel.py") == 2
    assert workflow.count("scripts/validate_shim_wheel.py") == 2
    assert "--wheel-dir wheelhouse --platform manylinux" in workflow
    assert "--wheel-dir dist --platform windows" in workflow
    assert "Reject implicit CUDA Toolkit DLL imports" not in workflow
    install_commands = re.findall(
        r"^\s*python -m pip install --force-reinstall.*$", workflow, re.MULTILINE
    )
    assert len(install_commands) == 2
    assert all("--no-deps" not in command for command in install_commands)
    assert all("--only-binary=:all:" in command for command in install_commands)
    assert workflow.count("python -m pip check") == 2


def test_runtime_publish_workflow_has_no_cuda_wheel_matrix():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "publish_runtime_pypi.yml"
    ).read_text(encoding="utf-8")

    assert "build_linux_runtime:" in workflow
    assert "build_windows_runtime:" in workflow
    assert "matrix:" not in workflow
    assert workflow.count("scripts/validate_runtime_wheel.py") == 4
    assert "--wheel-dir dist-runtime --platform linux" in workflow
    assert (
        workflow.count("--wheel-dir wheelhouse-runtime --platform manylinux")
        == 2
    )
    assert "--wheel-dir dist-runtime --platform windows" in workflow
    assert "--wheel-dir dist --platform pair" in workflow
    assert "auditwheel show wheelhouse-runtime/*.whl" in workflow
    assert workflow.count("--dependency-class driver-only") == 4
    assert workflow.count("TI_WITH_CUDA_TOOLKIT:BOOL=OFF") == 4
    assert (
        workflow.count(
            "TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE:BOOL=OFF"
        )
        == 4
    )
    assert "CUDA_TOOLKIT_VERSION" not in workflow
    assert "Jimver/cuda-toolkit" not in workflow
    assert "CUDAToolkit_NVCC_EXECUTABLE" not in workflow
    assert "Reject implicit CUDA Toolkit shared-library imports" in workflow
    assert "Reject implicit CUDA Toolkit DLL imports" in workflow
    assert "cudart64_|cupti64_|nvrtc64_" in workflow
    assert not re.search(
        r"taichi[-_]forge[-_]runtime[-_](?:cu|cuda)\d+",
        workflow,
        re.IGNORECASE,
    )


def test_runtime_project_defaults_to_driver_only():
    runtime_project = (
        REPO_ROOT / "packaging" / "runtime" / "pyproject.toml"
    ).read_text(encoding="utf-8")

    assert 'TI_WITH_CUDA_TOOLKIT = "OFF"' in runtime_project
    assert 'TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE = "OFF"' in runtime_project
    assert 'TI_WITH_CUPTI = "OFF"' in runtime_project


def test_cuda_toolkit_reference_workflow_is_non_publishing_and_separate():
    workflow = (
        REPO_ROOT
        / ".github"
        / "workflows"
        / "test_cuda_toolkit_reference.yml"
    ).read_text(encoding="utf-8")

    assert "TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE:BOOL=ON" in workflow
    assert "--dependency-class toolkit-reference" in workflow
    assert "Jimver/cuda-toolkit" in workflow
    assert "publish_runtime_pypi.yml" in workflow
    assert "gh-action-pypi-publish" not in workflow
    assert "actions/upload-artifact" not in workflow


def test_dynamic_cudart_requirement_prefers_primitive_reference_switch():
    assert repair_runtime_wheel._dynamic_cuda_runtime_required(
        {
            "TI_WITH_CUDA_TOOLKIT": "OFF",
            "TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE": "ON",
            "TI_CUDA_CUB_SORT_DYNAMIC_CUDART": "ON",
        }
    )
    assert not repair_runtime_wheel._dynamic_cuda_runtime_required(
        {
            "TI_WITH_CUDA_TOOLKIT": "ON",
            "TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE": "OFF",
            "TI_CUDA_CUB_SORT_DYNAMIC_CUDART": "ON",
        }
    )


def test_dynamic_cudart_requirement_keeps_legacy_cache_compatibility():
    assert repair_runtime_wheel._dynamic_cuda_runtime_required(
        {
            "TI_WITH_CUDA_TOOLKIT": "ON",
            "TI_CUDA_CUB_SORT_DYNAMIC_CUDART": "ON",
        }
    )
