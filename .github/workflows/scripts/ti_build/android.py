# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform
from pathlib import Path

# -- third party --
# -- own --
from .cmake import cmake_args
from .misc import banner, path_prepend
from .python import path_prepend
from .tinysh import Command, sh


# -- code --
@banner("Setup Android NDK")
def setup_android_ndk() -> None:
    # TODO: Auto install
    s = platform.system()
    if s != "Linux":
        raise RuntimeError(f"Android NDK is only supported on Linux, but the current platform is {s}.")

    ndkroot = Path(os.environ.get("ANDROID_NDK_ROOT", "/android-sdk/ndk-bundle"))
    toolchain = ndkroot / "build/cmake/android.toolchain.cmake"
    if not toolchain.exists():
        raise RuntimeError(f"ANDROID_NDK_ROOT is set to {ndkroot}, but the path does not exist.")

    p = ndkroot.resolve()
    os.environ["ANDROID_NDK_ROOT"] = str(p)
    cmake_args["CMAKE_TOOLCHAIN_FILE"] = str(toolchain)
    cmake_args["ANDROID_NATIVE_API_LEVEL"] = "29"
    cmake_args["ANDROID_ABI"] = "arm64-v8a"
    path_prepend("PATH", p / "toolchains/llvm/prebuilt/linux-x86_64/bin")


@banner("Build Taichi Android C-API Shared Library")
def build_android(python: Command, pip: Command) -> None:
    """
    Build the Taichi Android C-API shared library
    """
    cmake_args["TI_WITH_BACKTRACE"] = False
    cmake_args["TI_WITH_LLVM"] = False
    cmake_args["TI_WITH_C_API"] = True
    cmake_args["TI_BUILD_TESTS"] = False
    cmake_args.writeback()
    # scikit-build-core reads `SKBUILD_WHEEL_TAG_PLATFORM` to override the
    # wheel platform tag. The legacy scikit-build path used the env var
    # `TAICHI_FORCE_PLAT_NAME`; keep it set so user-level scripts still
    # produce the right tag.
    os.environ["TAICHI_FORCE_PLAT_NAME"] = "android-arm64"
    os.environ["SKBUILD_WHEEL_TAG_PLATFORM"] = "android_arm64"
    pip.install("-r", "requirements_dev.txt")
    python("setup.py", "clean")
    # Trigger a CMake-only build via `pip install -e .` (runs the full
    # scikit-build-core configure+build pipeline without packaging a wheel).
    python("-m", "pip", "install", "-e", ".", "--no-build-isolation", "-v")
    for p in Path(os.getcwd()).glob("**/libtaichi_c_api.so"):
        sh("aarch64-linux-android-strip", p)
