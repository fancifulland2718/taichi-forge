"""Strict manifests for user-built hardware source providers.

Source providers are compiled by the user against an SDK that is intentionally
absent from Forge wheels.  A manifest is therefore the trust and compatibility
boundary between an installed Forge runtime and a separately built binary.  It
does not build, load, probe, or select a provider by itself.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION = 2

_TOP_LEVEL_FIELDS = frozenset(
    (
        "schema_version",
        "provider_id",
        "provider_abi",
        "provider_abi_version",
        "binary",
        "toolchain",
        "runtime_dependencies",
        "source_identity",
        "specializations",
    )
)
_BINARY_FIELDS = frozenset(("path", "sha256"))
_TOOLCHAIN_FIELDS = frozenset(
    (
        "cuda_toolkit",
        "nvcc",
        "host_compiler",
        "cxx_abi",
        "build_flags",
        "target_code",
        "source_dependencies",
    )
)
_SOURCE_IDENTITY_FIELDS = frozenset(("kind", "value"))
_DEPENDENCY_FIELDS = frozenset(("name", "linkage", "version", "sha256"))
_SOURCE_DEPENDENCY_FIELDS = frozenset(("name", "version", "sha256"))


class SourceProviderManifestError(ValueError):
    """Raised when a user-built provider manifest is not exact and auditable."""


def _mapping(value, name):
    if not isinstance(value, dict):
        raise SourceProviderManifestError(f"{name} must be a JSON object")
    return value


def _exact_fields(value, expected, name):
    fields = frozenset(value)
    if fields == expected:
        return
    missing = sorted(expected.difference(fields))
    unexpected = sorted(fields.difference(expected))
    details = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise SourceProviderManifestError(f"{name} fields are not exact: " + "; ".join(details))


def _string(value, name):
    if not isinstance(value, str) or not value:
        raise SourceProviderManifestError(f"{name} must be a nonempty string")
    return value


def _string_tuple(value, name, *, allow_empty=False, allow_duplicates=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        suffix = "a JSON string array" if allow_empty else "a nonempty JSON string array"
        raise SourceProviderManifestError(f"{name} must be {suffix}")
    result = tuple(_string(item, f"{name} entry") for item in value)
    if not allow_duplicates and len(set(result)) != len(result):
        raise SourceProviderManifestError(f"{name} must not contain duplicates")
    return result


def _sha256(value, name):
    value = _string(value, name).lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise SourceProviderManifestError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _confined_binary_path(manifest_path, relative_path):
    relative = Path(_string(relative_path, "binary.path"))
    if relative.is_absolute():
        raise SourceProviderManifestError("binary.path must be relative to the manifest directory")
    root = manifest_path.parent.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SourceProviderManifestError("binary.path escapes the manifest directory") from exc
    if not candidate.is_file():
        raise SourceProviderManifestError(f"source-provider binary does not exist: {candidate}")
    return candidate


def _target_code(value):
    result = _string_tuple(value, "toolchain.target_code")
    for item in result:
        prefix, separator, capability = item.partition("_")
        if separator != "_" or prefix not in ("sm", "compute") or not capability.isdigit():
            raise SourceProviderManifestError(
                "toolchain.target_code entries must use sm_NN or compute_NN"
            )
    return result


@dataclass(frozen=True)
class SourceProviderRuntimeDependency:
    name: str
    linkage: str
    version: str
    sha256: str | None


@dataclass(frozen=True)
class SourceProviderSourceDependency:
    name: str
    version: str
    sha256: str


def _source_dependencies(value):
    if not isinstance(value, list) or not value:
        raise SourceProviderManifestError(
            "toolchain.source_dependencies must be a nonempty JSON object array"
        )
    dependencies = []
    names = set()
    for index, item in enumerate(value):
        name = f"toolchain.source_dependencies[{index}]"
        item = _mapping(item, name)
        _exact_fields(item, _SOURCE_DEPENDENCY_FIELDS, name)
        dependency_name = _string(item["name"], f"{name}.name")
        if dependency_name in names:
            raise SourceProviderManifestError(
                "toolchain source dependency names must be unique"
            )
        names.add(dependency_name)
        dependencies.append(
            SourceProviderSourceDependency(
                name=dependency_name,
                version=_string(item["version"], f"{name}.version"),
                sha256=_sha256(item["sha256"], f"{name}.sha256"),
            )
        )
    return tuple(dependencies)


@dataclass(frozen=True)
class SourceProviderManifest:
    manifest_path: Path
    provider_id: str
    provider_abi: str
    provider_abi_version: int
    binary_path: Path
    binary_sha256: str
    toolchain: object
    runtime_dependencies: tuple
    source_identity: object
    specializations: tuple
    manifest_sha256: str

    def supports_cuda_compute_capability(self, capability):
        """Checks exact cubin or forward-compatible embedded PTX coverage."""

        if isinstance(capability, bool) or not isinstance(capability, int) or capability <= 0:
            raise ValueError("CUDA compute capability must be a positive integer")
        target_code = self.toolchain["target_code"]
        if f"sm_{capability}" in target_code:
            return True
        return any(
            int(item.removeprefix("compute_")) <= capability
            for item in target_code
            if item.startswith("compute_")
        )

    @property
    def identity(self):
        return MappingProxyType(
            {
                "provider_id": self.provider_id,
                "provider_abi": self.provider_abi,
                "provider_abi_version": self.provider_abi_version,
                "binary_sha256": self.binary_sha256,
                "manifest_sha256": self.manifest_sha256,
                "source_identity": dict(self.source_identity),
                "target_code": self.toolchain["target_code"],
            }
        )


def load_source_provider_manifest(
    manifest_path,
    *,
    expected_provider_id=None,
    expected_provider_abi=None,
):
    """Loads and verifies one source-provider manifest without loading its DLL."""

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise SourceProviderManifestError(f"source-provider manifest does not exist: {path}")
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceProviderManifestError("source-provider manifest must be UTF-8 JSON") from exc
    document = _mapping(document, "manifest")
    _exact_fields(document, _TOP_LEVEL_FIELDS, "manifest")
    if document["schema_version"] != SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION:
        raise SourceProviderManifestError(
            "unsupported source-provider manifest schema "
            f"{document['schema_version']!r}; expected {SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION}"
        )

    provider_id = _string(document["provider_id"], "provider_id")
    provider_abi = _string(document["provider_abi"], "provider_abi")
    provider_abi_version = document["provider_abi_version"]
    if (
        isinstance(provider_abi_version, bool)
        or not isinstance(provider_abi_version, int)
        or provider_abi_version <= 0
    ):
        raise SourceProviderManifestError("provider_abi_version must be a positive integer")
    if expected_provider_id is not None and provider_id != expected_provider_id:
        raise SourceProviderManifestError(
            f"source-provider id {provider_id!r} does not match {expected_provider_id!r}"
        )
    if expected_provider_abi is not None and provider_abi != expected_provider_abi:
        raise SourceProviderManifestError(
            f"source-provider ABI {provider_abi!r} does not match {expected_provider_abi!r}"
        )

    binary = _mapping(document["binary"], "binary")
    _exact_fields(binary, _BINARY_FIELDS, "binary")
    binary_path = _confined_binary_path(path, binary["path"])
    binary_sha256 = _sha256(binary["sha256"], "binary.sha256")
    observed_binary_sha256 = _file_sha256(binary_path)
    if observed_binary_sha256 != binary_sha256:
        raise SourceProviderManifestError(
            "source-provider binary SHA-256 mismatch: "
            f"manifest={binary_sha256}, observed={observed_binary_sha256}"
        )

    toolchain_value = _mapping(document["toolchain"], "toolchain")
    _exact_fields(toolchain_value, _TOOLCHAIN_FIELDS, "toolchain")
    toolchain = MappingProxyType(
        {
            "cuda_toolkit": _string(toolchain_value["cuda_toolkit"], "toolchain.cuda_toolkit"),
            "nvcc": _string(toolchain_value["nvcc"], "toolchain.nvcc"),
            "host_compiler": _string(toolchain_value["host_compiler"], "toolchain.host_compiler"),
            "cxx_abi": _string(toolchain_value["cxx_abi"], "toolchain.cxx_abi"),
            "build_flags": _string_tuple(
                toolchain_value["build_flags"],
                "toolchain.build_flags",
                allow_empty=True,
                allow_duplicates=True,
            ),
            "target_code": _target_code(toolchain_value["target_code"]),
            "source_dependencies": _source_dependencies(
                toolchain_value["source_dependencies"]
            ),
        }
    )

    dependencies_value = document["runtime_dependencies"]
    if not isinstance(dependencies_value, list):
        raise SourceProviderManifestError("runtime_dependencies must be a JSON array")
    dependencies = []
    dependency_names = set()
    for index, item in enumerate(dependencies_value):
        item = _mapping(item, f"runtime_dependencies[{index}]")
        _exact_fields(item, _DEPENDENCY_FIELDS, f"runtime_dependencies[{index}]")
        name = _string(item["name"], f"runtime_dependencies[{index}].name")
        if name in dependency_names:
            raise SourceProviderManifestError("runtime dependency names must be unique")
        dependency_names.add(name)
        linkage = _string(item["linkage"], f"runtime_dependencies[{index}].linkage")
        if linkage not in ("dynamic", "static"):
            raise SourceProviderManifestError("runtime dependency linkage must be dynamic or static")
        dependency_hash = item["sha256"]
        if dependency_hash is not None:
            dependency_hash = _sha256(
                dependency_hash, f"runtime_dependencies[{index}].sha256"
            )
        dependencies.append(
            SourceProviderRuntimeDependency(
                name=name,
                linkage=linkage,
                version=_string(item["version"], f"runtime_dependencies[{index}].version"),
                sha256=dependency_hash,
            )
        )

    source_identity_value = _mapping(document["source_identity"], "source_identity")
    _exact_fields(source_identity_value, _SOURCE_IDENTITY_FIELDS, "source_identity")
    source_kind = _string(source_identity_value["kind"], "source_identity.kind")
    if source_kind not in ("git", "sha256"):
        raise SourceProviderManifestError("source_identity.kind must be git or sha256")
    source_value = _string(source_identity_value["value"], "source_identity.value")
    if source_kind == "sha256":
        source_value = _sha256(source_value, "source_identity.value")
    source_identity = MappingProxyType({"kind": source_kind, "value": source_value})

    specializations_value = document["specializations"]
    if not isinstance(specializations_value, list) or not specializations_value:
        raise SourceProviderManifestError("specializations must be a nonempty JSON object array")
    specializations = []
    for index, item in enumerate(specializations_value):
        item = _mapping(item, f"specializations[{index}]")
        if not item:
            raise SourceProviderManifestError(f"specializations[{index}] must not be empty")
        for key, value in item.items():
            _string(key, f"specializations[{index}] key")
            if not isinstance(value, (str, int, bool)) or isinstance(value, float):
                raise SourceProviderManifestError(
                    f"specializations[{index}].{key} must be a string, integer, or boolean"
                )
        specializations.append(MappingProxyType(dict(item)))

    return SourceProviderManifest(
        manifest_path=path,
        provider_id=provider_id,
        provider_abi=provider_abi,
        provider_abi_version=provider_abi_version,
        binary_path=binary_path,
        binary_sha256=binary_sha256,
        toolchain=toolchain,
        runtime_dependencies=tuple(dependencies),
        source_identity=source_identity,
        specializations=tuple(specializations),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = (
    "SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION",
    "SourceProviderManifest",
    "SourceProviderManifestError",
    "SourceProviderRuntimeDependency",
    "SourceProviderSourceDependency",
    "load_source_provider_manifest",
)
