"""Isolated SDK component inventories, without GPU or vendor build dependencies."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_provider_sdk_variants_coexist_without_changing_plain_components(tmp_path):
    cmake, ninja = shutil.which("cmake"), shutil.which("ninja")
    if not cmake or not ninja:
        pytest.skip("CMake and Ninja required for the SDK install contract")
    if os.name == "nt" and not shutil.which("cl") and not os.environ.get("CC"):
        pytest.skip("SDK install contract requires a configured Windows C/C++ compiler environment")
    root = Path(__file__).resolve().parents[2]
    build, sdk, default = (tmp_path / name for name in ("build", "sdk", "default"))
    (tmp_path / "adapter.c").write_text("int forge_install_fixture(void) { return 0; }\n")
    for variant in ("first", "second"):
        folder = tmp_path / variant
        folder.mkdir()
        (folder / "LICENSE.txt").write_text(f"{variant} fixture license\n")
    probe = "#include <taichi/forge_optix_runtime_provider.h>\nint abi_size(void) { return sizeof(TiForgeOptixExecutionApi); }\n"
    (tmp_path / "probe.c").write_text(probe)
    (tmp_path / "probe.cpp").write_text(probe)
    (tmp_path / "CMakeLists.txt").write_text(f"""cmake_minimum_required(VERSION 3.22)
project(ProviderInstallContract LANGUAGES C CXX)
include("{root.as_posix()}/cmake/TaichiProviderSdk.cmake")
foreach(name first second plain)
    add_library(${{name}} SHARED adapter.c)
endforeach()
foreach(name first second)
    ti_install_provider_sdk(${{name}} optix VARIANT ${{name}}
        PUBLIC_HEADERS "{root.as_posix()}/taichi/optix/forge_optix_provider.h"
            "{root.as_posix()}/taichi/optix/forge_optix_runtime_provider.h"
        LICENSE_FILES "${{CMAKE_CURRENT_SOURCE_DIR}}/${{name}}/LICENSE.txt"
        EXTERNAL_REQUIREMENTS "application-managed driver")
endforeach()
ti_install_provider_sdk(plain legacy)
add_library(probe_c OBJECT EXCLUDE_FROM_ALL probe.c)
add_library(probe_cpp OBJECT EXCLUDE_FROM_ALL probe.cpp)
target_include_directories(probe_c PRIVATE "{sdk.as_posix()}/include")
target_include_directories(probe_cpp PRIVATE "{sdk.as_posix()}/include")
target_compile_features(probe_cpp PRIVATE cxx_std_17)
""")

    def run(*args):
        result = subprocess.run([cmake, *map(str, args)], capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr

    run("-S", tmp_path, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release")
    run("--build", build, "--target", "first", "second", "plain")
    run("--install", build, "--prefix", default)
    assert not default.exists() or not any(path.is_file() for path in default.rglob("*"))
    sentinel = sdk / "sdk_manifest.json"
    sdk.mkdir()
    sentinel.write_bytes(b'{"base":"unchanged"}')
    manifests = {}
    for key in ("optix_first", "optix_second", "legacy"):
        run("--install", build, "--prefix", sdk, "--component", f"TaichiSDKProvider_{key}")
        manifest_file = sdk / f"share/taichi/providers/{key}.json"
        manifests[key] = manifest_file.read_bytes()
        manifest = json.loads(manifests[key])
        assert manifest["component"] == f"TaichiSDKProvider_{key}"
        assert manifest["provider_id"] == ("legacy" if key == "legacy" else "optix")
        assert manifest.get("variant") == (None if key == "legacy" else key.removeprefix("optix_"))
        paths = [entry["path"] for entry in manifest["files"]]
        assert len(paths) == len(set(paths))
        assert manifest["adapter"] in paths
        if key != "legacy":
            assert f"licenses/{key}/LICENSE.txt" in paths
            assert manifest["external_requirements"] == ["application-managed driver"]
        assert sum(entry["bytes"] for entry in manifest["files"]) == manifest["installed_payload_bytes"]
        for entry in manifest["files"]:
            data = (sdk / entry["path"]).read_bytes()
            assert len(data) == entry["bytes"]
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        for previous, frozen in manifests.items():
            assert (sdk / f"share/taichi/providers/{previous}.json").read_bytes() == frozen
        assert sentinel.read_bytes() == b'{"base":"unchanged"}'
    run("--install", build, "--prefix", sdk, "--component", "TaichiSDKProvider_optix_first")
    assert (sdk / "share/taichi/providers/optix_first.json").read_bytes() == manifests["optix_first"]
    assert (sdk / "licenses/optix_first/LICENSE.txt").read_text() == "first fixture license\n"
    assert (sdk / "licenses/optix_second/LICENSE.txt").read_text() == "second fixture license\n"
    run("--build", build, "--target", "probe_c", "probe_cpp")
