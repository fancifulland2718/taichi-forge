"""Explicit standalone FidelityFX SPD source compilation at plan preparation."""

from hashlib import sha256
from pathlib import Path
import struct
import subprocess
import tempfile


def compile_shader(compiler_path, source_path, source_format, output_format, levels, reduction):
    compiler = Path(compiler_path).expanduser().resolve(strict=True)
    sources = Path(source_path).expanduser().resolve(strict=True)
    headers = tuple((sources / name).read_bytes() for name in ("ffx_a.h", "ffx_spd.h"))
    template = Path(__file__).with_name("_spd_sources").joinpath("downsample.comp").read_text(encoding="utf-8")
    declarations = "\n".join(
        f"layout(set=0, binding={level + 2}, {output_format}) coherent uniform image2D mip{level};"
        for level in range(levels)
    )
    stores = "\n".join(
        f"case {level}u: if (all(lessThan(p, imageSize(mip{level})))) imageStore(mip{level}, p, value); break;"
        for level in range(levels)
    )
    load = "return imageLoad(mip5, clamp(p, ivec2(0), imageSize(mip5) - 1));" if levels > 6 else "return vec4(0.0);"
    expression = {
        "mean": "(v0 + v1 + v2 + v3) * 0.25",
        "min": "min(min(v0, v1), min(v2, v3))",
        "max": "max(max(v0, v1), max(v2, v3))",
    }[reduction]
    source = (
        template.replace("@SOURCE_FORMAT@", source_format)
        .replace("@OUTPUT_DECLARATIONS@", declarations)
        .replace("@OUTPUT_STORES@", stores)
        .replace("@MIP5_LOAD@", load)
        .replace("@REDUCTION@", expression)
    )
    options = {"capture_output": True, "text": True, "timeout": 120}
    version = subprocess.run([str(compiler), "--version"], check=True, **options)
    with tempfile.TemporaryDirectory(prefix="forge-spd-") as temporary:
        shader = Path(temporary) / "downsample.comp"
        binary = Path(temporary) / "downsample.spv"
        shader.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [
                str(compiler),
                "-V",
                "--target-env",
                "vulkan1.1",
                f"-I{sources}",
                str(shader),
                "-o",
                str(binary),
            ],
            **options,
        )
        if result.returncode:
            raise RuntimeError(f"SPD shader compilation failed: {result.stdout}\n{result.stderr}")
        data = binary.read_bytes()
    if len(data) % 4 or not data.startswith(b"\x03\x02\x23\x07"):
        raise RuntimeError("SPD compiler produced invalid SPIR-V")
    facts = {
        "provider": "fidelityfx_spd",
        "source_path": str(sources),
        "source_sha256": sha256(b"".join(headers)).hexdigest(),
        "wrapper_sha256": sha256(source.encode()).hexdigest(),
        "compiler_path": str(compiler),
        "compiler_sha256": sha256(compiler.read_bytes()).hexdigest(),
        "compiler_version": version.stdout.strip(),
        "shader_sha256": sha256(data).hexdigest(),
        "target": "vulkan1.1/GLSL450",
        "precision": "fp32",
        "reduction_strategy": "official_spd_lds",
    }
    return struct.unpack(f"<{len(data) // 4}I", data), facts
