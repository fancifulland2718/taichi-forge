"""Compile a caller-owned shader with an explicitly selected local compiler.

This optional development tool is not a runtime dependency. It does not install
tools, select a CUDA version, download SDKs, or alter Forge's built-in shaders.
The JSON sidecar is evidence of the invoked compiler, not a compatibility proof.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def compile_shader(*, kind, compiler, source, output, target, stage=None, includes=(), options=()):
    compiler = Path(compiler).resolve(strict=True)
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    sidecar = output.with_suffix(output.suffix + ".json")
    if source in (output, sidecar) or compiler in (output, sidecar):
        raise ValueError("output must not overwrite the source or compiler")
    if kind == "spirv":
        if stage not in ("vertex", "fragment", "compute", "task", "mesh", "raygen", "miss", "closesthit", "anyhit"):
            raise ValueError("SPIR-V compilation requires an explicit glslc shader stage")
        args = [f"--target-env={target}", f"-fshader-stage={stage}"]
    elif kind == "optix-ptx":
        if stage is not None:
            raise ValueError("PTX modules contain named entries, not a glslc stage")
        args = ["--ptx", "--std=c++17", f"--gpu-architecture={target}"]
    else:
        raise ValueError("kind must be spirv or optix-ptx")
    include_paths = tuple(str(Path(path).resolve(strict=True)) for path in includes)
    options = tuple(options)
    # Version failure does not invalidate successfully generated code.
    try:
        version = (
            subprocess.run(
                [str(compiler), "--version"], capture_output=True, text=True, check=True, timeout=20
            ).stdout.strip()
            or None
        )
    except (subprocess.SubprocessError, OSError):
        version = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="forge-shader-") as temporary:
        binary = Path(temporary) / output.name
        command = [
            str(compiler),
            *args,
            *(f"-I{path}" for path in include_paths),
            *options,
            str(source),
            "-o",
            str(binary),
        ]
        subprocess.run(command, check=True)
        code = binary.read_bytes()
    facts = {
        "schema": "taichi_forge.external_shader_build.v1",
        "format": kind,
        "stage": stage,
        "code_sha256": hashlib.sha256(code).hexdigest(),
        "build": {
            "target": target,
            "compiler": compiler.name,
            "compiler_version": version,
            "options": [*args, *options],
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        # Paths are diagnostic, not an artifact/execution identity.
        "inputs": {"compiler_path": str(compiler), "source_path": str(source), "include_paths": include_paths},
        "limitations": [
            "Include dependencies are not content-hashed; this is not a build cache.",
            "Successful compilation does not prove runtime or shader semantic compatibility.",
        ],
    }
    output.write_bytes(code)
    sidecar.write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("spirv", "optix-ptx"))
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--stage")
    parser.add_argument("--include", action="append", default=[], dest="includes", type=Path)
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        dest="options",
        help="Additional compiler argument; use --option=-O for leading dashes",
    )
    args = parser.parse_args()
    compile_shader(**vars(args))


if __name__ == "__main__":
    main()
