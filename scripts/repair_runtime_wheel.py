#!/usr/bin/env python3
"""Repair taichi-forge-runtime wheel contents from CMake build artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PACKAGE = "taichi_forge_runtime"
WRONG_PACKAGE = "taichi_forge"
CUDA_RUNTIME_MAJOR_MANIFEST = "cuda_runtime_major.txt"


def _cmake_cache_values(roots: list[Path] | None = None) -> dict[str, str]:
    values: dict[str, str] = {}
    search_roots = roots if roots is not None else [Path("_skbuild")]
    for cache in sorted(
        path
        for root in search_roots
        for path in root.rglob("CMakeCache.txt")
    ):
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            key_type, sep, value = line.partition("=")
            if not sep:
                continue
            key = key_type.split(":", 1)[0]
            values.setdefault(key, value)
    return values


def _enabled(value: str | None) -> bool:
    return value is not None and value.upper() in {"1", "ON", "TRUE", "YES"}


def _hash_record(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode()
    return f"sha256={digest.rstrip('=')}", str(len(data))


def _choose_artifact(candidates: list[Path], name: str) -> Path:
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise SystemExit(f"Could not find required runtime artifact: {name}")
    existing.sort(key=lambda path: (path.stat().st_size, str(path)), reverse=True)
    chosen = existing[0]
    print(f"Using {name}: {chosen} ({chosen.stat().st_size} bytes)")
    return chosen


def _artifact_roots(
    build_dir: Path | None = None, platform: str | None = None
) -> list[Path]:
    if build_dir is not None:
        if not build_dir.is_dir():
            raise SystemExit(f"Runtime build directory does not exist: {build_dir}")
        if platform == "linux":
            # Linux shared libraries keep their default build-tree location.
            return [build_dir]
        cache = _cmake_cache_values([build_dir])
        source_dir = cache.get("CMAKE_HOME_DIRECTORY")
        if not source_dir:
            raise SystemExit(
                f"CMAKE_HOME_DIRECTORY is missing from {build_dir / 'CMakeCache.txt'}"
            )
        runtime_dir = Path(source_dir) / "runtimes"
        if not runtime_dir.is_dir():
            raise SystemExit(
                f"Runtime output directory from selected build does not exist: "
                f"{runtime_dir}"
            )
        return [runtime_dir]
    roots = [Path("_skbuild"), Path("runtimes"), Path("dist-runtime")]
    return [root for root in roots if root.exists()]


def _candidate_dirs_from_env(names: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for name in names:
        value = os.environ.get(name)
        if value:
            dirs.append(Path(value))
    return dirs


def _existing_dirs(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    existing: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        existing.append(resolved)
    return existing


def _dynamic_cuda_runtime_required(cache: dict[str, str]) -> bool:
    return _enabled(cache.get("TI_WITH_CUDA_TOOLKIT")) and _enabled(
        cache.get("TI_CUDA_CUB_SORT_DYNAMIC_CUDART")
    )


def _windows_cudart_name(runtime_dll: Path) -> str | None:
    match = re.search(rb"cudart64_\d+\.dll", runtime_dll.read_bytes())
    if match is None:
        return None
    return match.group(0).decode("ascii")


def _linux_cudart_name(runtime_so: Path) -> str | None:
    matches = sorted(
        {
            match.group(0).decode("ascii")
            for match in re.finditer(
                rb"libcudart\.so\.[0-9]+(?:\.[0-9]+)*",
                runtime_so.read_bytes(),
            )
        },
        key=len,
        reverse=True,
    )
    return matches[0] if matches else None


def _cuda_runtime_major_from_name(platform: str, name: str) -> int:
    if platform == "windows":
        match = re.fullmatch(r"cudart64_(\d+)\.dll", name)
    elif platform == "linux":
        match = re.fullmatch(r"libcudart\.so\.(\d+)(?:\.\d+)*", name)
    else:
        raise ValueError(f"Unsupported CUDA runtime platform: {platform}")
    if match is None:
        raise ValueError(f"Unrecognized {platform} CUDA runtime name: {name}")
    return int(match.group(1))


def _is_cuda_runtime_name(platform: str, name: str) -> bool:
    try:
        _cuda_runtime_major_from_name(platform, name)
    except ValueError:
        return False
    return True


def _cuda_runtime_artifacts(
    platform: str, artifacts: dict[str, Path], roots: list[Path]
) -> dict[str, Path]:
    cache = _cmake_cache_values(roots)
    if not _dynamic_cuda_runtime_required(cache):
        return {}

    if platform == "windows":
        runtime_name = _windows_cudart_name(artifacts["taichi_runtime.dll"])
        if runtime_name is None:
            raise SystemExit(
                "Runtime build uses dynamic CUDA runtime, but taichi_runtime.dll "
                "does not reference cudart64_*.dll"
            )
        roots = _candidate_dirs_from_env(["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"])
        if cache.get("CUDAToolkit_BIN_DIR"):
            roots.insert(0, Path(cache["CUDAToolkit_BIN_DIR"]))
        search_dirs = _existing_dirs(
            roots
            + [root / "bin" for root in roots]
            + [root / "bin" / "x64" for root in roots]
            + [root / "x64" for root in roots]
        )
        candidates = [path / runtime_name for path in search_dirs]
        return {runtime_name: _choose_artifact(candidates, runtime_name)}

    if platform == "linux":
        runtime_name = _linux_cudart_name(artifacts["libtaichi_runtime.so"])
        if runtime_name is None:
            raise SystemExit(
                "Runtime build uses dynamic CUDA runtime, but libtaichi_runtime.so "
                "does not reference a versioned libcudart.so"
            )
        roots = _candidate_dirs_from_env(["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"])
        implicit_dirs = cache.get("_cmake_CUDAToolkit_implicit_link_directories", "")
        for item in implicit_dirs.split(";"):
            if item:
                roots.append(Path(item))
        search_dirs = _existing_dirs(
            roots
            + [root / "lib64" for root in roots]
            + [root / "lib" for root in roots]
            + [Path("/usr/local/cuda/lib64"), Path("/usr/local/cuda/lib")]
        )
        candidates = [path / runtime_name for path in search_dirs]
        return {runtime_name: _choose_artifact(candidates, runtime_name)}

    return {}


def _runtime_artifacts(
    platform: str, build_dir: Path | None = None
) -> dict[str, Path]:
    roots = _artifact_roots(build_dir, platform)
    cache_roots = [build_dir] if build_dir is not None else roots
    if not roots:
        raise SystemExit("No runtime artifact search roots exist")

    artifacts: dict[str, Path] = {}
    if platform == "linux":
        candidates = [path for root in roots for path in root.rglob("libtaichi_runtime.so")]
        artifacts["libtaichi_runtime.so"] = _choose_artifact(
            candidates, "libtaichi_runtime.so"
        )
    elif platform == "windows":
        dlls = [path for root in roots for path in root.rglob("taichi_runtime.dll")]
        libs = [path for root in roots for path in root.rglob("taichi_runtime.lib")]
        artifacts["taichi_runtime.dll"] = _choose_artifact(
            dlls, "taichi_runtime.dll"
        )
        artifacts["taichi_runtime.lib"] = _choose_artifact(
            libs, "taichi_runtime.lib"
        )
    else:
        raise SystemExit(f"Unsupported platform: {platform}")
    artifacts.update(_cuda_runtime_artifacts(platform, artifacts, cache_roots))
    return artifacts


def _move_wrong_package_files(root: Path) -> None:
    wrong_root = root / WRONG_PACKAGE / "_lib"
    if not wrong_root.exists():
        return

    right_root = root / PACKAGE / "_lib"
    right_root.mkdir(parents=True, exist_ok=True)
    for src in sorted(wrong_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(wrong_root)
        dst = right_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        print(f"Moved wheel entry {WRONG_PACKAGE}/_lib/{rel} -> {PACKAGE}/_lib/{rel}")

    shutil.rmtree(root / WRONG_PACKAGE, ignore_errors=True)


def _rewrite_record(root: Path) -> None:
    dist_infos = sorted(root.glob("*.dist-info"))
    if len(dist_infos) != 1:
        raise SystemExit(f"Expected one .dist-info directory, found {dist_infos}")

    record = dist_infos[0] / "RECORD"
    rows: list[list[str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path == record:
            rows.append([rel, "", ""])
        else:
            digest, size = _hash_record(path)
            rows.append([rel, digest, size])

    with record.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def _rewrite_wheel(wheel: Path, root: Path) -> None:
    tmp = wheel.with_suffix(".whl.tmp")
    if tmp.exists():
        tmp.unlink()
    with ZipFile(tmp, "w", ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())
    tmp.replace(wheel)


def repair_wheel(
    wheel: Path, platform: str, build_dir: Path | None = None
) -> None:
    artifacts = _runtime_artifacts(platform, build_dir)
    with tempfile.TemporaryDirectory(prefix="taichi-runtime-wheel-") as td:
        root = Path(td)
        with ZipFile(wheel) as zf:
            zf.extractall(root)

        _move_wrong_package_files(root)

        native_dir = root / PACKAGE / "_lib" / "runtime_native"
        native_dir.mkdir(parents=True, exist_ok=True)
        manifest = native_dir / CUDA_RUNTIME_MAJOR_MANIFEST
        manifest.unlink(missing_ok=True)
        for existing in native_dir.iterdir():
            if existing.is_file() and _is_cuda_runtime_name(
                platform, existing.name
            ):
                existing.unlink()
                print(
                    "Removed stale wheel CUDA runtime: "
                    f"{existing.relative_to(root)}"
                )
        for filename, artifact in artifacts.items():
            dst = native_dir / filename
            shutil.copy2(artifact, dst)
            print(f"Installed wheel native artifact: {dst.relative_to(root)}")

        cuda_runtime_names = [
            name
            for name in artifacts
            if (platform == "windows" and name.startswith("cudart64_"))
            or (platform == "linux" and name.startswith("libcudart.so."))
        ]
        if cuda_runtime_names:
            if len(cuda_runtime_names) != 1:
                raise SystemExit(
                    "Expected one bundled CUDA runtime, found "
                    f"{cuda_runtime_names}"
                )
            cuda_major = _cuda_runtime_major_from_name(
                platform, cuda_runtime_names[0]
            )
            with manifest.open("w", encoding="ascii", newline="\n") as f:
                f.write(f"{cuda_major}\n")
            print(
                "Installed wheel CUDA runtime manifest: "
                f"{manifest.relative_to(root)} -> {cuda_major}"
            )

        wrong_entries = sorted((root / WRONG_PACKAGE).rglob("*")) if (root / WRONG_PACKAGE).exists() else []
        if wrong_entries:
            raise SystemExit(f"Unexpected {WRONG_PACKAGE} entries remain: {wrong_entries}")

        _rewrite_record(root)
        _rewrite_wheel(wheel, root)
    print(f"Repaired runtime wheel: {wheel}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", default="dist-runtime")
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Use artifacts and CMake metadata only from this wheel build directory",
    )
    parser.add_argument("--platform", choices=["linux", "windows"], required=True)
    args = parser.parse_args()

    wheels = sorted(Path(args.wheel_dir).glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"Expected one runtime wheel, found {[str(w) for w in wheels]}")
    repair_wheel(wheels[0], args.platform, args.build_dir)


if __name__ == "__main__":
    main()
