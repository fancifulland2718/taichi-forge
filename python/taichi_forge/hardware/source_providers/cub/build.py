"""Build the optional CUB provider without rebuilding or relinking Forge.

This module is deliberately dormant during normal import.  It only invokes
NVCC when a user runs it explicitly, and writes the provider binary plus its
strict binding manifest into a caller-selected directory.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

_PROVIDER_ABI = "taichi-forge-cub-source-provider-c-abi1"
# Keep the standalone builder runnable from a source checkout that has no
# Forge extension module yet. The manifest loader's unit test pins this value.
_MANIFEST_SCHEMA_VERSION = 2


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_output(command):
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def _nvcc_version(nvcc):
    output = _run_output([str(nvcc), "--version"])
    match = re.search(r"\bV(\d+\.\d+\.\d+)\b", output)
    if not match:
        raise RuntimeError("could not determine the NVCC version")
    return match.group(1)


def _toolkit_versions(toolkit_root):
    version_path = toolkit_root / "version.json"
    if not version_path.is_file():
        raise RuntimeError(f"CUDA Toolkit version manifest is missing: {version_path}")
    document = json.loads(version_path.read_text(encoding="utf-8"))

    def version(component):
        try:
            value = document[component]["version"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"CUDA Toolkit version manifest lacks {component}") from exc
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"CUDA Toolkit {component} version is invalid")
        return value

    return version("cuda"), version("cuda_cudart")


def _cub_identity(toolkit_root):
    candidates = (
        toolkit_root / "include" / "cccl" / "cub" / "version.cuh",
        toolkit_root / "include" / "cub" / "version.cuh",
    )
    for path in candidates:
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^#define\s+CUB_VERSION\s+(\d+)\b", source, re.MULTILINE)
        if match:
            encoded = int(match.group(1))
            version = f"{encoded // 100000}.{encoded // 100 % 1000}.{encoded % 100}"
            return version, path
    raise RuntimeError("could not determine the CUB/CCCL version")


def _find_msvc_cl():
    direct = shutil.which("cl.exe")
    if direct:
        return Path(direct).resolve()
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
        "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    if not vswhere.is_file():
        return None
    output = _run_output(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ]
    ).strip()
    if not output:
        return None
    tools = Path(output) / "VC" / "Tools" / "MSVC"
    versions = sorted((item for item in tools.iterdir() if item.is_dir()), reverse=True)
    for version in versions:
        candidate = version / "bin" / "Hostx64" / "x64" / "cl.exe"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _host_compiler(explicit):
    if explicit:
        compiler = Path(explicit).resolve()
    elif os.name == "nt":
        compiler = _find_msvc_cl()
    else:
        candidate = shutil.which("c++") or shutil.which("g++")
        compiler = None if candidate is None else Path(candidate).resolve()
    if compiler is None or not compiler.is_file():
        raise RuntimeError(
            "could not find the NVCC host compiler; pass --host-compiler explicitly"
        )
    identity = f"{compiler.name}:sha256:{_sha256(compiler)}"
    cxx_abi = "msvc-x64" if os.name == "nt" else f"{platform.system().lower()}-{platform.machine()}"
    return compiler, identity, cxx_abi


def _target_code(value):
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise ValueError("--target-code must contain unique sm_NN/compute_NN entries")
    for item in result:
        if re.fullmatch(r"(?:sm|compute)_\d+", item) is None:
            raise ValueError("--target-code entries must use sm_NN or compute_NN")
    return result


def _gencode_flags(target_code):
    result = []
    for code in target_code:
        capability = code.split("_", 1)[1]
        result.extend(("-gencode", f"arch=compute_{capability},code={code}"))
    return result


def build_cub_source_provider(
    output_directory,
    *,
    target_code,
    nvcc=None,
    host_compiler=None,
):
    """Explicitly builds one user-owned provider and returns its manifest path."""

    nvcc_path = Path(nvcc or shutil.which("nvcc") or "").resolve()
    if not nvcc_path.is_file():
        raise RuntimeError("NVCC is unavailable; pass --nvcc or configure PATH")
    toolkit_root = nvcc_path.parent.parent.resolve()
    cuda_version, cudart_version = _toolkit_versions(toolkit_root)
    cub_version, cub_version_header = _cub_identity(toolkit_root)
    nvcc_version = _nvcc_version(nvcc_path)
    compiler, compiler_identity, cxx_abi = _host_compiler(host_compiler)
    targets = _target_code(target_code) if isinstance(target_code, str) else tuple(target_code)

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = ".dll" if os.name == "nt" else ".so"
    binary = output / f"taichi_forge_cub_source_provider_abi1{suffix}"
    manifest = output / "cub_source_provider.json"
    source = Path(__file__).with_name("provider.cu").resolve()

    flags = [
        "--shared",
        "-O3",
        "--std=c++17",
        "--cudart=static",
        "--compiler-bindir",
        str(compiler.parent),
        *_gencode_flags(targets),
    ]
    if os.name == "nt":
        # Current CCCL requires the conforming MSVC preprocessor. Keep the
        # CRT choice explicit in the binding manifest as well. Static CUDART
        # uses the static CRT on Windows, so matching /MT avoids a mixed-CRT
        # provider binary while the C ABI keeps Forge's CRT boundary isolated.
        flags.extend(("-Xcompiler=/Zc:preprocessor", "-Xcompiler=/MT"))
    else:
        flags.append("-Xcompiler=-fPIC")
    command = [str(nvcc_path), str(source), *flags, "-o", str(binary)]
    subprocess.run(command, check=True)
    if not binary.is_file():
        raise RuntimeError("NVCC completed without producing the provider binary")

    document = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "provider_id": "cub_reference",
        "provider_abi": _PROVIDER_ABI,
        "provider_abi_version": 1,
        "binary": {"path": binary.name, "sha256": _sha256(binary)},
        "toolchain": {
            "cuda_toolkit": cuda_version,
            "nvcc": nvcc_version,
            "host_compiler": compiler_identity,
            "cxx_abi": cxx_abi,
            "build_flags": flags,
            "target_code": list(targets),
            "source_dependencies": [
                {
                    "name": "cccl/cub",
                    "version": cub_version,
                    "sha256": _sha256(cub_version_header),
                }
            ],
        },
        "runtime_dependencies": [
            {
                "name": "cudart",
                "linkage": "static",
                "version": cudart_version,
                "sha256": None,
            }
        ],
        "source_identity": {"kind": "sha256", "value": _sha256(source)},
        "specializations": [
            {
                "operation": "radix_sort_pairs",
                "key_dtype": "u32",
                "value_dtype": "u32",
                "temporary_storage": "caller_owned",
            },
            {
                "operation": "radix_sort_pairs",
                "key_dtype": "u64",
                "value_dtype": "u32",
                "temporary_storage": "caller_owned",
            },
            {
                "operation": "exclusive_scan",
                "value_dtype": "u32",
                "temporary_storage": "caller_owned",
            },
            {
                "operation": "select_flagged",
                "value_dtype": "u32",
                "flag_dtype": "u32",
                "count_dtype": "u32",
                "temporary_storage": "caller_owned",
            },
        ],
    }
    manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="provider output directory")
    parser.add_argument(
        "--target-code",
        required=True,
        help="comma-separated cubin/PTX targets, for example sm_89,compute_89",
    )
    parser.add_argument("--nvcc", help="path to NVCC; defaults to PATH")
    parser.add_argument("--host-compiler", help="path to cl.exe/c++")
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    path = build_cub_source_provider(
        arguments.output,
        target_code=arguments.target_code,
        nvcc=arguments.nvcc,
        host_compiler=arguments.host_compiler,
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
