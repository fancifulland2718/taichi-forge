#!/usr/bin/env python3
"""Repair taichi-forge-runtime wheel contents from CMake build artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PACKAGE = "taichi_forge_runtime"
WRONG_PACKAGE = "taichi_forge"


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


def _artifact_roots() -> list[Path]:
    roots = [Path("_skbuild"), Path("runtimes"), Path("dist-runtime")]
    return [root for root in roots if root.exists()]


def _runtime_artifacts(platform: str) -> dict[str, Path]:
    roots = _artifact_roots()
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


def repair_wheel(wheel: Path, platform: str) -> None:
    artifacts = _runtime_artifacts(platform)
    with tempfile.TemporaryDirectory(prefix="taichi-runtime-wheel-") as td:
        root = Path(td)
        with ZipFile(wheel) as zf:
            zf.extractall(root)

        _move_wrong_package_files(root)

        native_dir = root / PACKAGE / "_lib" / "runtime_native"
        native_dir.mkdir(parents=True, exist_ok=True)
        for filename, artifact in artifacts.items():
            dst = native_dir / filename
            shutil.copy2(artifact, dst)
            print(f"Installed wheel native artifact: {dst.relative_to(root)}")

        wrong_entries = sorted((root / WRONG_PACKAGE).rglob("*")) if (root / WRONG_PACKAGE).exists() else []
        if wrong_entries:
            raise SystemExit(f"Unexpected {WRONG_PACKAGE} entries remain: {wrong_entries}")

        _rewrite_record(root)
        _rewrite_wheel(wheel, root)
    print(f"Repaired runtime wheel: {wheel}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", default="dist-runtime")
    parser.add_argument("--platform", choices=["linux", "windows"], required=True)
    args = parser.parse_args()

    wheels = sorted(Path(args.wheel_dir).glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"Expected one runtime wheel, found {[str(w) for w in wheels]}")
    repair_wheel(wheels[0], args.platform)


if __name__ == "__main__":
    main()
