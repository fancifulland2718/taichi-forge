#!/usr/bin/env python3
"""Validate taichi-forge-runtime wheel identity and bundled native artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from zipfile import ZipFile

from packaging.utils import canonicalize_name, parse_wheel_filename


PROJECT = "taichi-forge-runtime"
PACKAGE = "taichi_forge_runtime"
MANIFEST = f"{PACKAGE}/_lib/runtime_native/cuda_runtime_major.txt"
EXPORT_MANIFEST = (
    f"{PACKAGE}/_lib/runtime_native/taichi_runtime.exports.json"
)
CUDA_VARIANT = re.compile(r"(?:^|[+_.-])(?:cu|cuda)\d+", re.IGNORECASE)
FORBIDDEN_VENDOR_RUNTIME = re.compile(
    r"(?:"
    r"(?:cublas(?:lt)?64_|cusparse64_|cusolver64_|cufft(?:w)?64_|"
    r"curand64_|cupti64_|nvrtc(?:-builtins)?64_|nvjitlink_|nvoptix|nvcuda)"
    r"[^/]*\.dll"
    r"|lib(?:cublas(?:lt)?|cusparse|cusolver|cufft(?:w)?|curand|cupti|"
    r"nvrtc(?:-builtins)?|nvjitlink|nvoptix|cuda)(?:-[^.]+)?\.so(?:\..*)?"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RuntimeWheelInfo:
    path: Path
    platform: str
    version: str
    dependency_class: str
    cuda_major: int | None


def _wheel_platform(wheel: Path) -> str:
    name = wheel.name.lower()
    if "win_amd64" in name:
        return "windows"
    if "manylinux" in name:
        return "manylinux"
    if "linux_x86_64" in name:
        return "linux"
    if "macosx" in name:
        return "macos"
    raise RuntimeError(f"Unsupported runtime wheel platform tag: {wheel.name}")


def _cudart_major(platform: str, name: str) -> int | None:
    if platform == "windows":
        match = re.fullmatch(r"cudart64_(\d+)\.dll", name, re.IGNORECASE)
    else:
        match = re.fullmatch(
            r"libcudart(?:-[^.]+)?\.so\.(\d+)(?:\.\d+)*",
            name,
            re.IGNORECASE,
        )
    return int(match.group(1)) if match else None


def _export_digest(symbols: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(symbols).encode("utf-8"))
    return digest.hexdigest()


def _validate_export_manifest(
    zf: ZipFile, names: list[str], platform: str
) -> dict:
    manifests = [name for name in names if name == EXPORT_MANIFEST]
    if len(manifests) != 1:
        raise RuntimeError(
            f"Expected one {EXPORT_MANIFEST}, found {manifests}"
        )
    try:
        payload = json.loads(zf.read(manifests[0]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Invalid runtime export manifest") from exc
    schema = payload.get("schema_version") if isinstance(payload, dict) else None
    if schema not in {1, 2}:
        raise RuntimeError("Unsupported runtime export manifest schema")
    if platform != "windows" and schema != 2:
        raise RuntimeError("POSIX runtime requires export manifest schema 2")
    if schema == 2:
        expected_manifest_platform = {
            "windows": "windows-msvc",
            "linux": "linux-elf",
            "manylinux": "linux-elf",
            "macos": "macos-macho",
        }[platform]
        if payload.get("platform") != expected_manifest_platform:
            raise RuntimeError(
                "Runtime export manifest platform mismatch: "
                f"expected={expected_manifest_platform}, "
                f"actual={payload.get('platform')}"
            )
        if payload.get("abi_revision") != 1:
            raise RuntimeError("Unsupported runtime private ABI revision")
        if payload.get("binary_audited") is not True:
            raise RuntimeError("Runtime export manifest was not binary-audited")
        if payload.get("forbidden_export_families") != []:
            raise RuntimeError("Runtime exports bundled third-party APIs")
        collision_probes = payload.get("private_abi_collision_probe_symbols")
        if (
            not isinstance(collision_probes, list)
            or collision_probes != sorted(set(collision_probes))
            or not all(
                isinstance(symbol, str) and symbol
                for symbol in collision_probes
            )
            or "taichi_runtime_anchor" not in collision_probes
        ):
            raise RuntimeError(
                "Runtime private ABI collision probes are not canonical"
            )
    if platform == "windows" and payload.get("dll_audited") is not True:
        raise RuntimeError("Windows runtime export manifest was not DLL-audited")
    if platform in {"linux", "manylinux"}:
        if payload.get("elf_audited") is not True:
            raise RuntimeError("ELF runtime export manifest was not audited")
        if payload.get("unexpected_export_count") != 0:
            raise RuntimeError("ELF runtime export closure contains unexpected APIs")
        probes = payload.get("global_scope_probe_symbols")
        if (
            not isinstance(probes, list)
            or probes != sorted(set(probes))
            or not all(isinstance(symbol, str) and symbol for symbol in probes)
        ):
            raise RuntimeError("ELF runtime global-scope probes are not canonical")
    elif platform == "macos":
        if payload.get("macho_audited") is not True:
            raise RuntimeError("Mach-O runtime export manifest was not audited")
        if payload.get("unexpected_export_count") != 0:
            raise RuntimeError(
                "Mach-O runtime export closure contains unexpected APIs"
            )
        probes = payload.get("global_scope_probe_symbols")
        if (
            not isinstance(probes, list)
            or probes != sorted(set(probes))
            or not all(isinstance(symbol, str) and symbol for symbol in probes)
        ):
            raise RuntimeError(
                "Mach-O runtime global-scope probes are not canonical"
            )

    requested = payload.get("exports")
    actual = payload.get("actual_exports")
    if (
        not isinstance(requested, list)
        or not isinstance(actual, list)
        or not all(isinstance(symbol, str) and symbol for symbol in requested)
        or not all(isinstance(symbol, str) and symbol for symbol in actual)
        or requested != sorted(set(requested))
        or actual != sorted(set(actual))
    ):
        raise RuntimeError("Runtime export sets are not canonical")

    requested_set = set(requested)
    actual_set = set(actual)
    required_count = payload.get("shim_required_runtime_symbol_count")
    direct_count = payload.get("shim_direct_runtime_symbol_count")
    odr_count = payload.get("shim_shared_odr_symbol_count")
    raw_count = payload.get("raw_defined_symbol_count")
    limit = payload.get("configured_export_limit")
    counts = (
        required_count,
        raw_count,
        limit,
        payload.get("exported_symbol_count"),
        payload.get("actual_exported_symbol_count"),
        payload.get("implicit_exported_symbol_count"),
        payload.get("dropped_raw_symbol_count"),
    )
    if not all(isinstance(value, int) and value >= 0 for value in counts):
        raise RuntimeError("Runtime export counts are invalid")
    if schema == 2:
        if not all(
            isinstance(value, int) and value >= 0
            for value in (direct_count, odr_count)
        ):
            raise RuntimeError("Runtime private ABI closure counts are invalid")
        if direct_count + odr_count != required_count:
            raise RuntimeError(
                "Runtime direct and ODR closure counts are inconsistent"
            )
    if required_count <= 0 or raw_count < required_count:
        raise RuntimeError("Runtime export closure is empty or inconsistent")
    if payload["exported_symbol_count"] != len(requested):
        raise RuntimeError("Runtime requested export count is inconsistent")
    if payload["actual_exported_symbol_count"] != len(actual):
        raise RuntimeError("Runtime actual export count is inconsistent")
    if payload["implicit_exported_symbol_count"] != len(
        actual_set - requested_set
    ):
        raise RuntimeError("Runtime implicit export count is inconsistent")
    if payload["dropped_raw_symbol_count"] != raw_count - required_count:
        raise RuntimeError("Runtime dropped export count is inconsistent")
    if not requested_set.issubset(actual_set):
        raise RuntimeError("Runtime binary is missing requested exports")
    expected_anchor = (
        "_taichi_runtime_anchor" if platform == "macos" else "taichi_runtime_anchor"
    )
    if expected_anchor not in requested_set:
        raise RuntimeError("Runtime export manifest is missing its ABI anchor")
    if limit <= 0 or limit > 65_535 or len(actual) > limit:
        raise RuntimeError("Runtime export set exceeds its safety limit")
    if payload.get("export_set_sha256") != _export_digest(requested):
        raise RuntimeError("Runtime requested export digest is inconsistent")
    if payload.get("actual_export_set_sha256") != _export_digest(actual):
        raise RuntimeError("Runtime actual export digest is inconsistent")
    return payload


def _strict_binary_exports(
    zf: ZipFile,
    member: str,
    platform: str,
    expected: list[str],
) -> None:
    with tempfile.TemporaryDirectory(prefix="taichi-runtime-export-audit-") as td:
        binary = Path(td) / Path(member).name
        binary.write_bytes(zf.read(member))
        if platform == "windows":
            tool = shutil.which("dumpbin")
            if tool is None:
                raise RuntimeError("strict Windows runtime audit requires dumpbin")
            command = [tool, "/nologo", "/exports", str(binary)]
            pattern = re.compile(
                r"^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(?P<symbol>\S+)"
            )
        elif platform in {"linux", "manylinux"}:
            tool = shutil.which("nm") or shutil.which("llvm-nm")
            if tool is None:
                raise RuntimeError("strict ELF runtime audit requires nm")
            command = [tool, "-D", "-P", "-g", "--defined-only", str(binary)]
            pattern = None
        else:
            tool = shutil.which("nm") or shutil.which("llvm-nm")
            if tool is None:
                raise RuntimeError("strict Mach-O runtime audit requires nm")
            command = [tool, "-P", "-g", "-U", str(binary)]
            pattern = None
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "strict runtime binary export audit failed: "
                f"{completed.stdout.strip()}"
            )
        if pattern is not None:
            actual = {
                match.group("symbol")
                for line in completed.stdout.splitlines()
                if (match := pattern.match(line)) is not None
            }
        else:
            actual = set()
            for line in completed.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and len(parts[1]) == 1:
                    actual.add(parts[0].split("@", 1)[0])
        expected_set = set(expected)
        if actual != expected_set:
            missing = sorted(expected_set - actual)[:8]
            unexpected = sorted(actual - expected_set)[:8]
            raise RuntimeError(
                "final wheel runtime exports differ from the audited manifest: "
                f"missing={missing}, unexpected={unexpected}"
            )


def inspect_runtime_wheel(
    wheel: Path,
    expected_cuda_major: int | None = None,
    expected_dependency_class: str = "either",
    strict_binary: bool = False,
    required_export_manifest_schema: int | None = None,
) -> RuntimeWheelInfo:
    if expected_dependency_class not in {
        "driver-only",
        "toolkit-reference",
        "either",
    }:
        raise ValueError(
            "expected_dependency_class must be driver-only, "
            "toolkit-reference, or either"
        )
    platform = _wheel_platform(wheel)
    if not wheel.name.startswith("taichi_forge_runtime-"):
        raise RuntimeError(f"Unexpected runtime distribution name: {wheel.name}")
    distribution, filename_version, _, tags = parse_wheel_filename(wheel.name)
    if canonicalize_name(distribution) != PROJECT:
        raise RuntimeError(f"Unexpected runtime wheel distribution: {distribution}")
    if not tags or any(
        tag.interpreter != "py3" or tag.abi != "none" for tag in tags
    ):
        raise RuntimeError(
            "Runtime wheel must be Python-independent and tagged py3-none: "
            f"{wheel.name}"
        )

    with ZipFile(wheel) as zf:
        corrupt = zf.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Corrupt wheel member in {wheel.name}: {corrupt}")
        names = zf.namelist()
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        record_names = [
            name for name in names if name.endswith(".dist-info/RECORD")
        ]
        if len(metadata_names) != 1 or len(record_names) != 1:
            raise RuntimeError(
                f"Expected one METADATA and RECORD in {wheel.name}, found "
                f"metadata={metadata_names}, record={record_names}"
            )
        metadata = Parser().parsestr(
            zf.read(metadata_names[0]).decode("utf-8", errors="replace")
        )
        project = metadata.get("Name")
        version = metadata.get("Version") or ""
        if project != PROJECT:
            raise RuntimeError(f"Unexpected wheel project {project!r}: {wheel.name}")
        if str(filename_version) != version:
            raise RuntimeError(
                "Runtime wheel filename and METADATA versions differ: "
                f"filename={filename_version}, metadata={version}"
            )
        if CUDA_VARIANT.search(version):
            raise RuntimeError(
                f"CUDA-versioned runtime wheel versions are forbidden: {version}"
            )
        requirements = metadata.get_all("Requires-Dist", [])
        if requirements:
            raise RuntimeError(
                "Runtime wheels must not declare mandatory Python or provider "
                f"dependencies: {requirements}"
            )

        bundled_vendor_runtimes = sorted(
            name
            for name in names
            if FORBIDDEN_VENDOR_RUNTIME.fullmatch(Path(name).name)
        )
        if bundled_vendor_runtimes:
            raise RuntimeError(
                "Runtime wheel bundles optional CUDA or hardware-provider "
                f"libraries: {bundled_vendor_runtimes}"
            )

        manifests = [name for name in names if name == MANIFEST]
        if len(manifests) > 1:
            raise RuntimeError(
                f"Expected at most one CUDA runtime manifest in {wheel.name}, "
                f"found {manifests}"
            )

        cudarts = []
        for name in names:
            major = _cudart_major(platform, Path(name).name)
            if major is not None:
                cudarts.append((name, major))
        runtime_native_prefix = f"{PACKAGE}/_lib/runtime_native/"
        auditwheel_prefix = f"{PACKAGE}.libs/"
        cuda_major = None
        if manifests:
            major_text = zf.read(manifests[0]).decode("ascii").strip()
            if not major_text.isdigit() or int(major_text) <= 0:
                raise RuntimeError(
                    f"Invalid CUDA runtime manifest in {wheel.name}: "
                    f"{major_text!r}"
                )
            cuda_major = int(major_text)
            if len(cudarts) != 1 or cudarts[0][1] != cuda_major:
                raise RuntimeError(
                    f"Expected one CUDART matching manifest major {cuda_major} "
                    f"in {wheel.name}, found {cudarts}"
                )
            cudart_path = cudarts[0][0]
            if not (
                cudart_path.startswith(runtime_native_prefix)
                or (
                    platform == "manylinux"
                    and cudart_path.startswith(auditwheel_prefix)
                )
            ):
                raise RuntimeError(
                    f"CUDART is outside the runtime package in {wheel.name}: "
                    f"{cudart_path}"
                )
            dependency_class = "toolkit-reference"
        else:
            if cudarts:
                raise RuntimeError(
                    f"Driver-only runtime wheel {wheel.name} contains CUDART "
                    f"without a manifest: {cudarts}"
                )
            dependency_class = "driver-only"

        if (
            expected_dependency_class != "either"
            and dependency_class != expected_dependency_class
        ):
            raise RuntimeError(
                f"Runtime dependency class mismatch in {wheel.name}: "
                f"expected={expected_dependency_class}, actual={dependency_class}"
            )
        if expected_cuda_major is not None and cuda_major != expected_cuda_major:
            raise RuntimeError(
                f"CUDA runtime major mismatch in {wheel.name}: "
                f"expected={expected_cuda_major}, manifest={cuda_major}"
            )

        if platform == "windows":
            native_runtimes = [
                name
                for name in names
                if name == f"{PACKAGE}/_lib/runtime_native/taichi_runtime.dll"
            ]
        elif platform == "macos":
            native_runtimes = [
                name
                for name in names
                if name
                == f"{PACKAGE}/_lib/runtime_native/libtaichi_runtime.dylib"
            ]
        else:
            # The shim has a direct DT_NEEDED entry for the stable SONAME
            # libtaichi_runtime.so.  auditwheel may hash grafted dependencies,
            # but it must leave this wheel-owned primary ELF at its canonical
            # package path.  Accepting a hashed primary runtime here would let
            # the runtime wheel pass while the independently built shim could
            # no longer resolve its dependency after installation.
            native_runtimes = [
                name
                for name in names
                if name == f"{PACKAGE}/_lib/runtime_native/libtaichi_runtime.so"
            ]
            hashed_primary_runtimes = [
                name
                for name in names
                if (
                    name.startswith(runtime_native_prefix)
                    or name.startswith(auditwheel_prefix)
                )
                and re.fullmatch(
                    r"libtaichi_runtime-[^.]+\.so", Path(name).name
                )
            ]
            if hashed_primary_runtimes:
                raise RuntimeError(
                    "The primary Linux runtime must retain its canonical "
                    "libtaichi_runtime.so name and package path; found "
                    f"auditwheel-style hashed copies: {hashed_primary_runtimes}"
                )
        if len(native_runtimes) != 1:
            raise RuntimeError(
                f"Expected one platform native runtime in {wheel.name}, "
                f"found {native_runtimes}"
            )
        export_manifest = _validate_export_manifest(zf, names, platform)
        if (
            required_export_manifest_schema is not None
            and export_manifest["schema_version"]
            != required_export_manifest_schema
        ):
            raise RuntimeError(
                "Runtime export manifest schema mismatch in "
                f"{wheel.name}: expected={required_export_manifest_schema}, "
                f"actual={export_manifest['schema_version']}"
            )
        if strict_binary:
            _strict_binary_exports(
                zf,
                native_runtimes[0],
                platform,
                export_manifest["actual_exports"],
            )
        if platform == "windows":
            import_library = f"{PACKAGE}/_lib/runtime_native/taichi_runtime.lib"
            if names.count(import_library) != 1:
                raise RuntimeError(
                    f"Expected one {import_library} in {wheel.name}"
                )

        wrong_entries = [
            name for name in names if name.startswith("taichi_forge/_lib/")
        ]
        if wrong_entries:
            raise RuntimeError(
                f"Runtime artifacts were installed into the shim package: "
                f"{wrong_entries}"
            )

    return RuntimeWheelInfo(
        wheel, platform, version, dependency_class, cuda_major
    )


def validate_runtime_wheels(
    wheel_dir: Path,
    expected_platform: str,
    expected_cuda_major: int | None = None,
    expected_dependency_class: str = "either",
    strict_binary: bool = False,
    required_export_manifest_schema: int | None = None,
) -> list[RuntimeWheelInfo]:
    wheels = sorted(wheel_dir.glob("*.whl"))
    expected_count = 2 if expected_platform == "pair" else 1
    if len(wheels) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} runtime wheel(s) in {wheel_dir}, "
            f"found {[wheel.name for wheel in wheels]}"
        )
    infos = [
        inspect_runtime_wheel(
            wheel,
            expected_cuda_major,
            expected_dependency_class,
            strict_binary,
            required_export_manifest_schema,
        )
        for wheel in wheels
    ]
    if expected_platform == "pair":
        platforms = sorted(info.platform for info in infos)
        if platforms != ["manylinux", "windows"]:
            raise RuntimeError(
                "Expected one Windows and one manylinux runtime wheel, "
                f"found {platforms}"
            )
        versions = {info.version for info in infos}
        if len(versions) != 1:
            raise RuntimeError(f"Runtime wheel versions differ: {sorted(versions)}")
        dependency_classes = {info.dependency_class for info in infos}
        if len(dependency_classes) != 1:
            raise RuntimeError(
                "Runtime wheel dependency classes differ: "
                f"{sorted(dependency_classes)}"
            )
        cuda_majors = {info.cuda_major for info in infos}
        if len(cuda_majors) != 1:
            raise RuntimeError(
                "Runtime wheel CUDART majors differ: "
                f"{sorted(str(major) for major in cuda_majors)}"
            )
    elif infos[0].platform != expected_platform:
        raise RuntimeError(
            f"Expected a {expected_platform} runtime wheel, "
            f"found {infos[0].platform}: {infos[0].path.name}"
        )
    return infos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument(
        "--platform",
        choices=["windows", "linux", "manylinux", "macos", "pair"],
        required=True,
    )
    parser.add_argument("--cuda-major", type=int)
    parser.add_argument(
        "--dependency-class",
        choices=["driver-only", "toolkit-reference", "either"],
        default="either",
        help=(
            "Required CUDA runtime dependency class. Standard release "
            "workflows must use driver-only; either preserves validation of "
            "already-published legacy wheels."
        ),
    )
    parser.add_argument(
        "--strict-binary",
        action="store_true",
        help="Re-audit the native binary inside the final wheel",
    )
    parser.add_argument(
        "--export-manifest-schema",
        type=int,
        choices=[1, 2],
        help=(
            "Require an exact runtime export manifest schema. Local current-"
            "source consumers should require schema 2; omission preserves "
            "validation of already-published legacy Windows wheels."
        ),
    )
    args = parser.parse_args()

    try:
        infos = validate_runtime_wheels(
            args.wheel_dir,
            args.platform,
            args.cuda_major,
            args.dependency_class,
            args.strict_binary,
            args.export_manifest_schema,
        )
    except (OSError, RuntimeError, UnicodeError) as exc:
        raise SystemExit(str(exc)) from exc
    for info in infos:
        suffix = (
            f", bundled CUDART major={info.cuda_major}"
            if info.cuda_major is not None
            else ", bundled CUDART=none"
        )
        print(
            f"Validated {info.path.name}: platform={info.platform}, "
            f"version={info.version}, dependency={info.dependency_class}"
            f"{suffix}"
        )


if __name__ == "__main__":
    main()
