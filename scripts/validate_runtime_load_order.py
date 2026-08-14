#!/usr/bin/env python3
"""Qualify split-runtime symbol isolation in both ELF load orders."""

import argparse
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


_PROVIDER_SOURCE = r"""
int LLVMContextCreate(void) {
  return 14;
}
"""

_DRIVER_SOURCE = r"""
int LLVMContextCreate(void);

int forge_collision_driver_value(void) {
  return LLVMContextCreate();
}
"""

_PRIVATE_ABI_COLLISION_SOURCE = r"""
void taichi_runtime_anchor(void) {
}
"""


def _compile_probe(directory: Path) -> tuple[Path, Path]:
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("a C compiler is required for the ELF load-order probe")
    provider_source = directory / "provider.c"
    driver_source = directory / "driver.c"
    private_source = directory / "private_abi_collision.c"
    provider = directory / "libforge_collision_provider.so"
    driver = directory / "libforge_collision_driver.so"
    private_provider = directory / "libforge_private_abi_collision.so"
    provider_source.write_text(_PROVIDER_SOURCE, encoding="utf-8")
    driver_source.write_text(_DRIVER_SOURCE, encoding="utf-8")
    private_source.write_text(_PRIVATE_ABI_COLLISION_SOURCE, encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            str(provider_source),
            "-Wl,-soname,libforge_collision_provider.so",
            "-o",
            str(provider),
        ],
        check=True,
    )
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            str(driver_source),
            f"-L{directory}",
            "-lforge_collision_provider",
            "-Wl,-rpath,$ORIGIN",
            "-o",
            str(driver),
        ],
        check=True,
    )
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            str(private_source),
            "-o",
            str(private_provider),
        ],
        check=True,
    )
    return driver, private_provider


def _load_driver(path: Path, *, global_scope: bool) -> ctypes.CDLL:
    scope = os.RTLD_GLOBAL if global_scope else os.RTLD_LOCAL
    driver = ctypes.CDLL(str(path), mode=scope | os.RTLD_NOW)
    driver.forge_collision_driver_value.argtypes = []
    driver.forge_collision_driver_value.restype = ctypes.c_int
    return driver


def _make_fill_kernel(ti):
    @ti.kernel
    def fill(output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in output:
            output[i] = i + 1

    return fill


def _child(
    mode: str,
    driver_path: Path | None,
    private_provider_path: Path | None,
) -> None:
    if mode == "private-abi-collision":
        assert private_provider_path is not None
        private_provider = ctypes.CDLL(
            str(private_provider_path), mode=os.RTLD_GLOBAL | os.RTLD_NOW
        )
        try:
            import taichi_forge  # noqa: F401
        except RuntimeError as exc:
            if "process-global Taichi private ABI" not in str(exc):
                raise
        else:
            raise RuntimeError(
                "process-global private ABI collision was not rejected"
            )
        if private_provider._handle == 0:
            raise RuntimeError("private ABI collision provider was unloaded")
        print("[runtime-load-order] private ABI collision: rejected", flush=True)
        return

    assert driver_path is not None
    if mode == "driver-first":
        driver = _load_driver(driver_path, global_scope=True)
        if driver.forge_collision_driver_value() != 14:
            raise RuntimeError("driver provider was preempted before Forge import")

    import taichi_forge as ti

    if mode == "runtime-first":
        driver = _load_driver(driver_path, global_scope=False)
    if driver.forge_collision_driver_value() != 14:
        raise RuntimeError(
            f"{mode} driver reference resolved to Forge's private LLVM symbol"
        )

    # Force creation and teardown of the LLVM CPU backend. In driver-first
    # order this also verifies that the driver's global collision cannot
    # preempt the runtime's localized LLVM implementation.
    ti.init(arch=ti.cpu, offline_cache=False)
    values = ti.ndarray(ti.i32, shape=4)
    fill = _make_fill_kernel(ti)
    fill(values)
    if values.to_numpy().tolist() != [1, 2, 3, 4]:
        raise RuntimeError(f"{mode} CPU execution produced incorrect values")
    ti.reset()
    if driver.forge_collision_driver_value() != 14:
        raise RuntimeError(f"{mode} driver provider changed after Forge teardown")
    print(f"[runtime-load-order] {mode}: passed", flush=True)


def _parent() -> None:
    if sys.platform != "linux":
        print("[runtime-load-order] skipped: ELF qualification is Linux-only")
        return
    with tempfile.TemporaryDirectory(prefix="taichi-forge-load-order-") as td:
        directory = Path(td)
        driver, private_provider = _compile_probe(directory)
        environment = dict(os.environ)
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            str(directory)
            if not existing
            else f"{directory}{os.pathsep}{existing}"
        )
        for mode in ("runtime-first", "driver-first"):
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    "--child",
                    mode,
                    "--driver",
                    str(driver),
                ],
                check=True,
                env=environment,
            )
        subprocess.run(
            [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--child",
                "private-abi-collision",
                "--private-provider",
                str(private_provider),
            ],
            check=True,
            env=environment,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--child",
        choices=["runtime-first", "driver-first", "private-abi-collision"],
    )
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--private-provider", type=Path)
    args = parser.parse_args()
    if args.child is None:
        _parent()
        return
    if args.child == "private-abi-collision":
        if args.private_provider is None or not args.private_provider.is_file():
            raise SystemExit(
                "--private-provider is required in private collision mode"
            )
        _child(args.child, None, args.private_provider.resolve())
        return
    if args.driver is None or not args.driver.is_file():
        raise SystemExit("--driver is required in load-order child mode")
    _child(args.child, args.driver.resolve(), None)


if __name__ == "__main__":
    main()
