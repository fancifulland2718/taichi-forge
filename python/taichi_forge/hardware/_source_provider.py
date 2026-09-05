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


SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION = 3

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
_BUILD_PROFILE_FIELDS = frozenset(
    ("schema_version", "kind", "abi_boundary", "driver_contract")
)
_DRIVER_CONTRACT_FIELDS = frozenset(("minimum_api_version", "ptx_api_version", "basis"))


def _build_profile(value):
    value = _mapping(value, "build_profile")
    _exact_fields(value, _BUILD_PROFILE_FIELDS, "build_profile")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise SourceProviderManifestError("unsupported build-profile schema")
    if (
        value["kind"] != "cuda-toolkit-addon"
        or value["abi_boundary"] != "provider-c-abi"
    ):
        raise SourceProviderManifestError(
            "source providers require a CUDA addon with a provider C ABI"
        )
    driver = _mapping(value["driver_contract"], "build_profile.driver_contract")
    _exact_fields(driver, _DRIVER_CONTRACT_FIELDS, "build_profile.driver_contract")
    for name in ("minimum_api_version", "ptx_api_version"):
        version = driver[name]
        if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
            raise SourceProviderManifestError(
                f"driver_contract.{name} must be a positive integer"
            )
    if driver["ptx_api_version"] < driver["minimum_api_version"]:
        raise SourceProviderManifestError(
            "PTX driver API requirement cannot precede the runtime floor"
        )
    _string(driver["basis"], "driver_contract.basis")
    return MappingProxyType(
        {**value, "driver_contract": MappingProxyType(dict(driver))}
    )


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
    raise SourceProviderManifestError(
        f"{name} fields are not exact: " + "; ".join(details)
    )


def _string(value, name):
    if not isinstance(value, str) or not value:
        raise SourceProviderManifestError(f"{name} must be a nonempty string")
    return value


def _string_tuple(value, name, *, allow_empty=False, allow_duplicates=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        suffix = (
            "a JSON string array" if allow_empty else "a nonempty JSON string array"
        )
        raise SourceProviderManifestError(f"{name} must be {suffix}")
    result = tuple(_string(item, f"{name} entry") for item in value)
    if not allow_duplicates and len(set(result)) != len(result):
        raise SourceProviderManifestError(f"{name} must not contain duplicates")
    return result


def _sha256(value, name):
    value = _string(value, name).lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise SourceProviderManifestError(
            f"{name} must be a lowercase SHA-256 hex digest"
        )
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
        raise SourceProviderManifestError(
            "binary.path must be relative to the manifest directory"
        )
    root = manifest_path.parent.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SourceProviderManifestError(
            "binary.path escapes the manifest directory"
        ) from exc
    if not candidate.is_file():
        raise SourceProviderManifestError(
            f"source-provider binary does not exist: {candidate}"
        )
    return candidate


def _target_code(value):
    result = _string_tuple(value, "toolchain.target_code")
    for item in result:
        prefix, separator, capability = item.partition("_")
        if (
            separator != "_"
            or prefix not in ("sm", "compute")
            or not capability.isdigit()
        ):
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


def _source_dependencies(value, field="source_dependencies"):
    if not isinstance(value, list) or not value:
        raise SourceProviderManifestError(
            f"toolchain.{field} must be a nonempty JSON object array"
        )
    dependencies = []
    names = set()
    for index, item in enumerate(value):
        name = f"toolchain.{field}[{index}]"
        item = _mapping(item, name)
        _exact_fields(item, _SOURCE_DEPENDENCY_FIELDS, name)
        dependency_name = _string(item["name"], f"{name}.name")
        if dependency_name in names:
            raise SourceProviderManifestError(f"toolchain {field} names must be unique")
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
    build_profile: object = None

    def cuda_compatibility(self, capability, driver_api_version):
        """Evaluate declared code/driver requirements at explicit load/search.

        Eligibility is not execution qualification. Library initialization,
        optional APIs and numerical qualification still belong to the provider.
        Legacy manifests do not acquire an invented driver contract.
        """
        if (
            isinstance(capability, bool)
            or not isinstance(capability, int)
            or capability <= 0
        ):
            raise ValueError("CUDA compute capability must be a positive integer")
        if driver_api_version is not None and (
            isinstance(driver_api_version, bool)
            or not isinstance(driver_api_version, int)
            or driver_api_version <= 0
        ):
            raise ValueError(
                "CUDA driver API version must be a positive integer or None"
            )
        targets = self.toolchain["target_code"]
        sass = [
            code
            for code in targets
            if code.startswith("sm_")
            and int(code[3:]) // 10 == capability // 10
            and int(code[3:]) <= capability
        ]
        ptx = [
            code
            for code in targets
            if code.startswith("compute_") and int(code[8:]) <= capability
        ]
        path = "sass" if sass else "ptx_jit" if ptx else None
        required = None
        reason = None if path else "target_code_unavailable"
        if self.build_profile is not None and path:
            contract = self.build_profile["driver_contract"]
            required = contract["minimum_api_version" if sass else "ptx_api_version"]
            if driver_api_version is None:
                reason = "driver_version_unavailable"
            elif driver_api_version < required:
                reason = "driver_api_too_old" if sass else "ptx_jit_driver_too_old"
        return {
            "eligible": reason is None,
            "code_path": path,
            "required_driver_api_version": required,
            "observed_driver_api_version": driver_api_version,
            "unavailable_reason": reason,
            "driver_contract_declared": self.build_profile is not None,
            "execution_qualified": False,
        }

    def build_report(self):
        """JSON-safe build facts, without probing or loading any external runtime."""
        compilers = self.toolchain.get("compiler_components", ())
        components = [
            *(
                {
                    "name": item.name,
                    "version": item.version,
                    "sha256": item.sha256,
                    "role": "compiler",
                }
                for item in compilers
            ),
            *(
                {
                    "name": item.name,
                    "version": item.version,
                    "sha256": item.sha256,
                    "role": "source",
                }
                for item in self.toolchain["source_dependencies"]
            ),
            *(
                {
                    "name": item.name,
                    "version": item.version,
                    "sha256": item.sha256,
                    "role": item.linkage,
                }
                for item in self.runtime_dependencies
            ),
        ]
        if not compilers:
            components.insert(
                0,
                {"name": "nvcc", "version": self.toolchain["nvcc"], "role": "compiler"},
            )
        profile = (
            None
            if self.build_profile is None
            else {
                **self.build_profile,
                "driver_contract": dict(self.build_profile["driver_contract"]),
            }
        )
        facts = {
            "profile": profile,
            "toolkit_release": self.toolchain["cuda_toolkit"],
            "components": components,
            "target_code": list(self.toolchain["target_code"]),
            "provider_abi": self.provider_abi,
            "provider_abi_version": self.provider_abi_version,
            "binary_sha256": self.binary_sha256,
            "host_abi": self.toolchain["cxx_abi"],
        }
        # Binary identity binds compile options and source; local compiler paths,
        # manifest whitespace, and repository HEAD are not compatibility axes.
        facts["build_identity"] = (
            "source-build:"
            + hashlib.sha256(
                json.dumps(facts, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        return facts

    def supports_cuda_compute_capability(self, capability):
        """Check standard cubin/PTX architecture coverage, not driver eligibility."""
        return self.cuda_compatibility(capability, None)["code_path"] is not None

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
        raise SourceProviderManifestError(
            f"source-provider manifest does not exist: {path}"
        )
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceProviderManifestError(
            "source-provider manifest must be UTF-8 JSON"
        ) from exc
    document = _mapping(document, "manifest")
    schema = document.get("schema_version")
    if type(schema) is not int or schema not in (
        2,
        SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION,
    ):
        raise SourceProviderManifestError(
            "unsupported source-provider manifest schema "
            f"{schema!r}; expected 2 or {SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION}"
        )
    _exact_fields(
        document,
        _TOP_LEVEL_FIELDS | ({"build_profile"} if schema == 3 else set()),
        "manifest",
    )
    build_profile = _build_profile(document["build_profile"]) if schema == 3 else None

    provider_id = _string(document["provider_id"], "provider_id")
    provider_abi = _string(document["provider_abi"], "provider_abi")
    provider_abi_version = document["provider_abi_version"]
    if (
        isinstance(provider_abi_version, bool)
        or not isinstance(provider_abi_version, int)
        or provider_abi_version <= 0
    ):
        raise SourceProviderManifestError(
            "provider_abi_version must be a positive integer"
        )
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
    _exact_fields(
        toolchain_value,
        _TOOLCHAIN_FIELDS | ({"compiler_components"} if schema == 3 else set()),
        "toolchain",
    )
    toolchain = MappingProxyType(
        {
            "cuda_toolkit": _string(
                toolchain_value["cuda_toolkit"], "toolchain.cuda_toolkit"
            ),
            "nvcc": _string(toolchain_value["nvcc"], "toolchain.nvcc"),
            "host_compiler": _string(
                toolchain_value["host_compiler"], "toolchain.host_compiler"
            ),
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
            **(
                {
                    "compiler_components": _source_dependencies(
                        toolchain_value["compiler_components"], "compiler_components"
                    )
                }
                if schema == 3
                else {}
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
            raise SourceProviderManifestError(
                "runtime dependency linkage must be dynamic or static"
            )
        dependency_hash = item["sha256"]
        if dependency_hash is not None:
            dependency_hash = _sha256(
                dependency_hash, f"runtime_dependencies[{index}].sha256"
            )
        dependencies.append(
            SourceProviderRuntimeDependency(
                name=name,
                linkage=linkage,
                version=_string(
                    item["version"], f"runtime_dependencies[{index}].version"
                ),
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
        raise SourceProviderManifestError(
            "specializations must be a nonempty JSON object array"
        )
    specializations = []
    for index, item in enumerate(specializations_value):
        item = _mapping(item, f"specializations[{index}]")
        if not item:
            raise SourceProviderManifestError(
                f"specializations[{index}] must not be empty"
            )
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
        build_profile=build_profile,
    )


__all__ = (
    "SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION",
    "SourceProviderManifest",
    "SourceProviderManifestError",
    "SourceProviderRuntimeDependency",
    "SourceProviderSourceDependency",
    "load_source_provider_manifest",
)
