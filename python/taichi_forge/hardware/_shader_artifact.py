"""Cold, immutable artifacts for externally compiled hardware programs.

Metadata records caller/compiler facts, not a safety proof. SPIR-V and PTX stay
separate formats. No compiler, driver, or provider is loaded by this module.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import struct


def _digest(facts):
    return hashlib.sha256(
        json.dumps(facts, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ShaderBuildInfo:
    """Optional declared provenance; unknown facts remain None.

    Paths/timestamps are deliberately not identities. ``options`` is ordered;
    the compiler may give repeated flags an order-dependent meaning.
    """

    target: str | None = None
    compiler: str | None = None
    compiler_version: str | None = None
    options: tuple[str, ...] = ()
    source_sha256: str | None = None

    def __post_init__(self):
        for name in ("target", "compiler", "compiler_version", "source_sha256"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string or None")
        if isinstance(self.options, str):
            raise TypeError("options must be a sequence of compiler arguments")
        options = tuple(self.options)
        if any(not isinstance(value, str) for value in options):
            raise TypeError("options must contain strings")
        if self.source_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None:
            raise ValueError("source_sha256 must be a lowercase SHA256 digest")
        object.__setattr__(self, "options", options)


def _binary(value, label):
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be bytes-like")
    return bytes(value)


# OpEntryPoint execution models. Resource reflection remains native-owned.
_SPIRV_MODELS = {
    "vertex": 0,
    "fragment": 4,
    "compute": 5,
    "task": 5364,
    "mesh": 5365,
    "raygen": 5313,
    "any_hit": 5315,
    "closest_hit": 5316,
    "miss": 5317,
}


def _spirv_entries(code):
    if len(code) < 20 or len(code) % 4:
        raise ValueError("SPIR-V requires a complete, word-aligned module")
    words = struct.unpack(f"<{len(code) // 4}I", code)
    if words[0] != 0x07230203:
        raise ValueError("SPIR-V magic is invalid (expected little-endian words)")
    cursor, entries = 5, set()
    while cursor < len(words):
        size, opcode = words[cursor] >> 16, words[cursor] & 0xFFFF
        if size == 0 or cursor + size > len(words):
            raise ValueError("SPIR-V instruction is truncated")
        if opcode == 15:  # OpEntryPoint
            if size < 4:
                raise ValueError("SPIR-V entry-point instruction is truncated")
            name_bytes = code[(cursor + 3) * 4 : (cursor + size) * 4]
            if b"\x00" not in name_bytes:
                raise ValueError("SPIR-V entry point is not terminated")
            entries.add((words[cursor + 1], name_bytes.split(b"\x00", 1)[0].decode("utf-8")))
        cursor += size
    return entries


@dataclass(frozen=True)
class SpirvShader:
    """One declared entry in an immutable SPIR-V module.

    Graphics currently consumes ``main``; RT entry support is backend-specific.
    Reflection does not prove external shader effects or bounds.
    """

    code: bytes = field(repr=False)
    stage: str
    entry_point: str = "main"
    build: ShaderBuildInfo = field(default_factory=ShaderBuildInfo)
    code_sha256: str = field(init=False)
    artifact_id: str = field(init=False)

    def __post_init__(self):
        code = _binary(self.code, "SPIR-V")
        if self.stage not in _SPIRV_MODELS:
            raise ValueError("unsupported SPIR-V shader stage")
        if not isinstance(self.build, ShaderBuildInfo):
            raise TypeError("build must be ShaderBuildInfo")
        if (_SPIRV_MODELS[self.stage], self.entry_point) not in _spirv_entries(code):
            raise ValueError(f"SPIR-V has no {self.stage} entry {self.entry_point!r}")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "code_sha256", hashlib.sha256(code).hexdigest())
        object.__setattr__(self, "artifact_id", "spirv:" + _digest(self.to_dict()))

    @classmethod
    def from_file(cls, path, *, stage, entry_point="main", build=None):
        return cls(Path(path).read_bytes(), stage, entry_point, build or ShaderBuildInfo())

    def to_dict(self):
        return {
            "format": "spirv",
            "code_sha256": self.code_sha256,
            "stage": self.stage,
            "entry_point": self.entry_point,
            "build": asdict(self.build),
        }


@dataclass(frozen=True)
class PtxModule:
    """Externally compiled OptiX PTX, not CUDA source or a generic CUDA kernel.

    PTX version/target are read from the module; OptiX validates program entries
    when creating a program. A newer user module may need a newer driver than
    the wheel's built-in PTX. The original bytes are never rewritten.
    """

    code: bytes = field(repr=False)
    build: ShaderBuildInfo = field(default_factory=ShaderBuildInfo)
    code_sha256: str = field(init=False)
    ptx_version: str = field(init=False)
    ptx_target: str = field(init=False)
    artifact_id: str = field(init=False)

    def __post_init__(self):
        code = _binary(self.code, "PTX")
        if not isinstance(self.build, ShaderBuildInfo):
            raise TypeError("build must be ShaderBuildInfo")
        if b"\x00" in code:
            raise ValueError("PTX must not contain embedded NUL bytes")
        text = code.decode("utf-8")
        # Ignore comments so a source comment cannot masquerade as a header.
        text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
        version = re.search(r"(?m)^\s*\.version\s+(\d+\.\d+)\s*$", text)
        target = re.search(r"(?m)^\s*\.target\s+(sm_\d+[a-z]?)(?:\s*,[^\n]*)?\s*$", text)
        address = re.search(r"(?m)^\s*\.address_size\s+64\s*$", text)
        if not version or not target or not address:
            raise ValueError("OptiX PTX requires version, target and 64-bit address headers")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "code_sha256", hashlib.sha256(code).hexdigest())
        object.__setattr__(self, "ptx_version", version[1])
        object.__setattr__(self, "ptx_target", target[1])
        object.__setattr__(self, "artifact_id", "optix-ptx:" + _digest(self.to_dict()))

    @classmethod
    def from_file(cls, path, *, build=None):
        return cls(Path(path).read_bytes(), build or ShaderBuildInfo())

    def to_dict(self):
        return {
            "format": "optix-ptx",
            "code_sha256": self.code_sha256,
            "ptx_version": self.ptx_version,
            "ptx_target": self.ptx_target,
            "build": asdict(self.build),
        }
