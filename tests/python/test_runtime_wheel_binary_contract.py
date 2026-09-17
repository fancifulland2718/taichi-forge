import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from scripts import validate_runtime_wheel
from scripts import validate_shim_wheel
from scripts import repair_runtime_wheel


@pytest.mark.parametrize("case", ("linux", "windows", "missing_static", "cached_shared"))
def test_vkfft_compiler_library_discovery_is_static_and_scoped(tmp_path, case):
    """Exercise the real CMake module without requiring a GPU or C++ compiler."""
    cmake = shutil.which("cmake")
    if not cmake or not shutil.which("ninja"):
        pytest.skip("CMake and Ninja are required for the configure-only contract")
    sdk = tmp_path / "sdk"
    libraries = ("glslang", "SPIRV-Tools-opt", "SPIRV-Tools")
    sdk_lib = sdk / ("Lib" if case == "windows" else "lib")
    sdk_lib.mkdir(parents=True)
    for name in libraries:
        if case == "windows":
            (sdk_lib / f"{name}.lib").touch()
        else:
            (sdk_lib / f"lib{name}.so").touch()
            if case != "missing_static" or name != "glslang":
                (sdk_lib / f"lib{name}.a").touch()
    include = sdk / "include/glslang/Include"
    include.mkdir(parents=True)
    (include / "glslang_c_interface.h").touch()
    source = tmp_path / "source"
    source.mkdir()
    (source / "vkFFT").mkdir()
    (source / "vkFFT/vkFFT.h").write_text("return 10304;\n", encoding="utf-8")
    (source / "FindVulkan.cmake").write_text(
        'set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/include")\n'
        'add_library(Vulkan::Vulkan INTERFACE IMPORTED)\n',
        encoding="utf-8",
    )
    (source / "FindThreads.cmake").write_text(
        "add_library(Threads::Threads INTERFACE IMPORTED)\n", encoding="utf-8"
    )
    (source / "CMakeLists.txt").write_text(
        '''
cmake_minimum_required(VERSION 3.20)
project(vkfft_library_discovery LANGUAGES NONE)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CXX_COMPILE_FEATURES cxx_std_17)
set(CMAKE_CXX_CREATE_SHARED_LIBRARY "unused")
set(TI_BUILD_VKFFT_PROVIDER ON CACHE BOOL "")
set(TI_VKFFT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "")
set(APPLE FALSE)
if(CASE STREQUAL "windows")
    set(UNIX FALSE)
    set(CMAKE_FIND_LIBRARY_PREFIXES "")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
else()
    set(UNIX TRUE)
    set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so;.a")
endif()
set(original_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
# Confine all library lookup to the fixture; no host SDK can mask missing .a.
set(CMAKE_IGNORE_PREFIX_PATH "")
set(CMAKE_FIND_USE_CMAKE_PATH FALSE)
set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)
set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)
set(CMAKE_FIND_USE_INSTALL_PREFIX FALSE)
include("${PROVIDER_MODULE}")
if(NOT "${CMAKE_FIND_LIBRARY_SUFFIXES}" STREQUAL "${original_suffixes}")
    message(FATAL_ERROR "VkFFT changed other targets' library search policy")
endif()
foreach(variable TI_VKFFT_GLSLANG_STATIC TI_VKFFT_SPIRV_OPT_STATIC TI_VKFFT_SPIRV_STATIC)
    if(CASE STREQUAL "windows")
        set(suffix ".lib")
    else()
        set(suffix ".a")
    endif()
    get_filename_component(extension "${${variable}}" LAST_EXT)
    if(NOT extension STREQUAL suffix)
        message(FATAL_ERROR "Wrong compiler library: ${${variable}}")
    endif()
endforeach()
set_target_properties(taichi_forge_vkfft_provider PROPERTIES LINKER_LANGUAGE CXX)
''',
        encoding="utf-8",
    )
    module = Path(__file__).resolve().parents[2] / "cmake/TaichiVkfftProvider.cmake"
    command = [
        cmake, "-S", str(source), "-B", str(tmp_path / "build"), "-G", "Ninja",
        f"-DCASE={case}", f"-DPROVIDER_MODULE={module.as_posix()}",
    ]
    if case == "cached_shared":
        command.append(f"-DTI_VKFFT_GLSLANG_STATIC={(sdk_lib / 'libglslang.so').as_posix()}")
    result = subprocess.run(
        command, env={**os.environ, "VULKAN_SDK": sdk.as_posix()},
        capture_output=True, text=True, timeout=30,
    )
    output = result.stdout + result.stderr
    if case in {"linux", "windows"}:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0, output
        assert "TI_VKFFT_GLSLANG_STATIC" in output
        assert ("static archive" if case == "cached_shared" else "Could not find") in output


def _wheel_with_member(path: Path, member: str) -> ZipFile:
    with ZipFile(path, "w") as zf:
        zf.writestr(member, b"native binary placeholder")
    return ZipFile(path)


def test_strict_elf_export_audit_matches_final_binary(monkeypatch, tmp_path):
    wheel = _wheel_with_member(
        tmp_path / "runtime.whl",
        "taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so",
    )
    monkeypatch.setattr(validate_runtime_wheel.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        validate_runtime_wheel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "TAICHI_FORGE_RUNTIME_PRIVATE_1 A 0 0\n"
                "taichi_runtime_anchor@@TAICHI_FORGE_RUNTIME_PRIVATE_1 T 10 4\n"
            ),
        ),
    )
    try:
        validate_runtime_wheel._strict_binary_exports(
            wheel,
            "taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so",
            "manylinux",
            ["TAICHI_FORGE_RUNTIME_PRIVATE_1", "taichi_runtime_anchor"],
        )
    finally:
        wheel.close()


@pytest.mark.parametrize("provider", ("cudss", "vkfft"))
def test_windows_provider_optional_exports_are_owner_scoped(
    provider, monkeypatch, tmp_path
):
    required = f"taichi_forge_{provider}_provider_query"
    stem = (
        validate_runtime_wheel.VKFFT_ADAPTER_STEM
        if provider == "vkfft"
        else "taichi_forge_cudss_provider_abi1_cudss080"
    )
    member = f"taichi_forge_runtime/_lib/hardware_providers/{stem}.dll"
    monkeypatch.setattr(validate_runtime_wheel.shutil, "which", lambda name: name)
    exports = set()
    monkeypatch.setattr(
        validate_runtime_wheel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                f"    {index} 0 00001000 {symbol}"
                for index, symbol in enumerate(sorted(exports), 1)
            ),
        ),
    )
    with _wheel_with_member(tmp_path / "runtime.whl", member) as wheel:
        def audit():
            validate_runtime_wheel._strict_provider_exports(
                wheel, {member: required}, "windows"
            )

        exports.add(required)
        audit()  # Previous adapters do not need the new extension symbols.
        if provider == "cudss":
            # A real adapter export must be checked independently of the allowlist.
            exports.add("taichi_forge_cudss_factor_statistics_query")
            audit()
        extensions = validate_runtime_wheel.OPTIONAL_PROVIDER_EXPORTS[required]
        for extension in sorted(extensions):
            exports.add(extension)
            audit()
        exports.remove(required)
        with pytest.raises(RuntimeError, match="requires"):
            audit()
        exports.add(required)
        exports.add("taichi_forge_other_provider_query")
        with pytest.raises(RuntimeError, match="permits only"):
            audit()
        if provider == "vkfft":
            exports.remove("taichi_forge_other_provider_query")
            exports.add("unintended_shader_compiler_export")
            with pytest.raises(RuntimeError, match="permits only"):
                audit()


def test_strict_elf_export_audit_rejects_repair_drift(monkeypatch, tmp_path):
    wheel = _wheel_with_member(
        tmp_path / "runtime.whl",
        "taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so",
    )
    monkeypatch.setattr(validate_runtime_wheel.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        validate_runtime_wheel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="taichi_runtime_anchor T 10 4\nLLVMContextCreate T 20 4\n",
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="differ from the audited manifest"):
            validate_runtime_wheel._strict_binary_exports(
                wheel,
                "taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so",
                "manylinux",
                ["taichi_runtime_anchor"],
            )
    finally:
        wheel.close()


def test_strict_shim_audit_requires_dependency_and_relative_runpath(
    monkeypatch, tmp_path
):
    wheel = _wheel_with_member(
        tmp_path / "shim.whl",
        "taichi_forge/_lib/core/taichi_python.so",
    )
    monkeypatch.setattr(validate_shim_wheel.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        validate_shim_wheel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "(NEEDED) Shared library: [libtaichi_runtime.so]\n"
                "(RUNPATH) Library runpath: [$ORIGIN/../../../"
                "taichi_forge_runtime/_lib/runtime_native]\n"
            ),
        ),
    )
    try:
        validate_shim_wheel._strict_dynamic_contract(
            wheel,
            "taichi_forge/_lib/core/taichi_python.so",
            "manylinux",
        )
    finally:
        wheel.close()


def test_manylinux_normalization_requires_canonical_primary_runtime(tmp_path):
    canonical = tmp_path / "canonical.whl"
    with ZipFile(canonical, "w") as zf:
        zf.writestr(
            "taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so",
            b"runtime",
        )
    repair_runtime_wheel.normalize_manylinux_wheel(canonical)

    hashed = tmp_path / "hashed.whl"
    with ZipFile(hashed, "w") as zf:
        zf.writestr(
            "taichi_forge_runtime/_lib/runtime_native/"
            "libtaichi_runtime-deadbeef.so",
            b"runtime",
        )
    with pytest.raises(SystemExit, match="must preserve.*primary runtime"):
        repair_runtime_wheel.normalize_manylinux_wheel(hashed)
