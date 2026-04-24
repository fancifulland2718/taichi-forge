# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform

# -- third party --
# -- own --
from .bootstrap import get_cache_home
from .cmake import cmake_args
from .dep import download_dep
from .misc import banner, get_cache_home, is_manylinux2014


# -- code --
# LLVM 20.1.x preferred, LLVM 19.1.x fallback (both stock upstream, built
# with specific CMake flags — no custom source patches).
#
# Recommended flow:
#
#   1. Run `scripts/build_llvm20_local.ps1` (or `build_llvm19_local.ps1`)
#      once to produce `dist/taichi-llvm-20/` (or `taichi-llvm-19/`).
#   2. Export `LLVM_DIR=<repo>/dist/taichi-llvm-20/lib/cmake/llvm`
#      (or just the install prefix — `setup_llvm` accepts either).
#   3. Run `python build.py ...` normally.
#
# Auto-detection order (first hit wins):
#   $LLVM_DIR (if already set and exists)
#   dist/taichi-llvm-20/  (via _find_local_llvm(20))
#   dist/taichi-llvm-19/  (via _find_local_llvm(19))
#
# Alternatively, set `LLVM20_WIN_URL` / `LLVM20_LINUX_URL` / etc. to a zip
# URL you have hosted yourself. If none of the above is set, `setup_llvm`
# falls back to LLVM 19 URLs (LLVM19_*_URL) and finally to the legacy
# LLVM 15 prebuilts on Linux/macOS (raises a clear error on Windows).

_LLVM20_WIN_URL = os.environ.get("LLVM20_WIN_URL", "")
_LLVM20_LINUX_URL = os.environ.get("LLVM20_LINUX_URL", "")
_LLVM20_LINUX_MANYLINUX_URL = os.environ.get("LLVM20_LINUX_MANYLINUX_URL", "")
_LLVM20_LINUX_AMDGPU_URL = os.environ.get("LLVM20_LINUX_AMDGPU_URL", "")
_LLVM20_MAC_ARM64_URL = os.environ.get("LLVM20_MAC_ARM64_URL", "")
_LLVM20_MAC_X64_URL = os.environ.get("LLVM20_MAC_X64_URL", "")

_LLVM19_WIN_URL = os.environ.get("LLVM19_WIN_URL", "")
_LLVM19_LINUX_URL = os.environ.get("LLVM19_LINUX_URL", "")
_LLVM19_LINUX_MANYLINUX_URL = os.environ.get("LLVM19_LINUX_MANYLINUX_URL", "")
_LLVM19_LINUX_AMDGPU_URL = os.environ.get("LLVM19_LINUX_AMDGPU_URL", "")
_LLVM19_MAC_ARM64_URL = os.environ.get("LLVM19_MAC_ARM64_URL", "")
_LLVM19_MAC_X64_URL = os.environ.get("LLVM19_MAC_X64_URL", "")


def _repo_root() -> str:
    """Return the absolute path to the Taichi repo root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _find_local_llvm(major: int) -> str:
    """Look for a local LLVM install produced by build_llvm{major}_local.ps1.

    Returns the path to the lib/cmake/llvm subdirectory because CMake's
    find_package(LLVM CONFIG) expects LLVM_DIR to point directly at the
    directory containing LLVMConfig.cmake.
    """
    prefix = os.path.join(_repo_root(), "dist", f"taichi-llvm-{major}")
    cmake_dir = os.path.join(prefix, "lib", "cmake", "llvm")
    if os.path.isfile(os.path.join(cmake_dir, "LLVMConfig.cmake")):
        return cmake_dir
    return ""


def _pick_url(u, major: int):
    """Return (url, out_dir_name) for the current platform + LLVM major version.

    The returned URL may be empty, in which case the caller should fall
    through to the next fallback tier.
    """
    prefix = f"llvm{major}"
    if u.system == "Linux":
        if cmake_args.get_effective("TI_WITH_AMDGPU"):
            url = globals().get(f"_LLVM{major}_LINUX_AMDGPU_URL", "")
            return url, f"{prefix}-amdgpu"
        if is_manylinux2014():
            url = globals().get(f"_LLVM{major}_LINUX_MANYLINUX_URL", "")
            return url, f"{prefix}-manylinux2014"
        url = globals().get(f"_LLVM{major}_LINUX_URL", "")
        return url, prefix
    if (u.system, u.machine) == ("Darwin", "arm64"):
        return globals().get(f"_LLVM{major}_MAC_ARM64_URL", ""), f"{prefix}-m1"
    if (u.system, u.machine) == ("Darwin", "x86_64"):
        return globals().get(f"_LLVM{major}_MAC_X64_URL", ""), f"{prefix}-mac"
    if (u.system, u.machine) == ("Windows", "AMD64"):
        return globals().get(f"_LLVM{major}_WIN_URL", ""), prefix
    return "", ""


@banner("Setup LLVM")
def setup_llvm() -> None:
    """
    Download and install LLVM, preferring 20 → 19 → legacy 15 (where no
    modern prebuilt is available).

    Respects the LLVM_DIR environment variable: if it is set to an existing
    directory this function is a no-op, so developers with a local LLVM
    install can skip the download entirely.
    """
    existing = os.environ.get("LLVM_DIR", "")
    if existing and os.path.isdir(existing):
        return

    # Auto-detect a local build produced by scripts/build_llvm{20,19}_local.ps1.
    # Prefer LLVM 20 if both are present.
    for major in (20, 19):
        local = _find_local_llvm(major)
        if local:
            os.environ["LLVM_DIR"] = local
            print(f":: Using local LLVM {major} install at {local}")
            return

    u = platform.uname()

    # Tier 1: LLVM 20 URLs
    url, out_name = _pick_url(u, 20)
    if url:
        out = get_cache_home() / out_name
        download_dep(url, out, strip=1 if u.system != "Windows" else 0)
        os.environ["LLVM_DIR"] = str(out)
        return

    # Tier 2: LLVM 19 URLs
    url, out_name = _pick_url(u, 19)
    if url:
        out = get_cache_home() / out_name
        download_dep(url, out, strip=1 if u.system != "Windows" else 0)
        os.environ["LLVM_DIR"] = str(out)
        return

    # Tier 3: legacy LLVM 15 fallback on Linux/macOS; error out on Windows.
    if u.system == "Linux":
        if cmake_args.get_effective("TI_WITH_AMDGPU"):
            legacy_url = "https://github.com/GaleSeLee/assets/releases/download/v0.0.5/taichi-llvm-15.0.0-linux.zip"
            out = get_cache_home() / "llvm15-amdgpu-005"
        elif is_manylinux2014():
            legacy_url = "https://github.com/ailzhang/torchhub_example/releases/download/0.3/taichi-llvm-15-linux.zip"
            out = get_cache_home() / "llvm15-manylinux2014"
        else:
            legacy_url = "https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-linux.zip"
            out = get_cache_home() / "llvm15"
        download_dep(legacy_url, out, strip=1)
    elif (u.system, u.machine) == ("Darwin", "arm64"):
        legacy_url = "https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-m1-nozstd.zip"
        out = get_cache_home() / "llvm15-m1-nozstd"
        download_dep(legacy_url, out, strip=1)
    elif (u.system, u.machine) == ("Darwin", "x86_64"):
        legacy_url = "https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/llvm-15-mac10.15.zip"
        out = get_cache_home() / "llvm15-mac"
        download_dep(legacy_url, out, strip=1)
    elif (u.system, u.machine) == ("Windows", "AMD64"):
        raise RuntimeError(
            "LLVM 20/19 for Windows is not available for download and no "
            "local install was found.\n\n"
            "Please build it once locally:\n"
            "    pwsh -File scripts/build_llvm20_local.ps1\n"
            "  or\n"
            "    pwsh -File scripts/build_llvm19_local.ps1\n\n"
            "Then re-run `python build.py`. The script installs to "
            "dist/taichi-llvm-20 (or dist/taichi-llvm-19) which setup_llvm "
            "auto-detects.\n\n"
            "Alternatively, set LLVM20_WIN_URL (or LLVM19_WIN_URL) to a zip "
            "URL you host yourself, or set LLVM_DIR to an existing install "
            "prefix."
        )
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    os.environ["LLVM_DIR"] = str(out)
