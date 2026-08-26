import hashlib
import json

import pytest

from taichi_forge.hardware._source_provider import (
    SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION,
    SourceProviderManifestError,
    load_source_provider_manifest,
)


def _manifest(binary_name, binary_hash):
    return {
        "schema_version": SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION,
        "provider_id": "cub_reference",
        "provider_abi": "taichi-forge-cub-source-provider-c-abi1",
        "provider_abi_version": 1,
        "binary": {"path": binary_name, "sha256": binary_hash},
        "toolchain": {
            "cuda_toolkit": "13.2",
            "cccl": "3.2.0",
            "nvcc": "13.2.51",
            "host_compiler": "MSVC 19.44",
            "cxx_abi": "msvc-19",
            "build_flags": ["--shared", "-O3"],
            "target_code": ["sm_89", "compute_89"],
        },
        "runtime_dependencies": [
            {
                "name": "cudart",
                "linkage": "static",
                "version": "13.2",
                "sha256": None,
            }
        ],
        "source_identity": {"kind": "sha256", "value": "a" * 64},
        "specializations": [
            {
                "operation": "radix_sort_pairs",
                "key_dtype": "u32",
                "value_dtype": "u32",
                "temporary_storage": "caller_owned",
            }
        ],
    }


def _write_manifest(tmp_path, document):
    path = tmp_path / "provider.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def test_source_provider_manifest_binds_binary_toolchain_and_target(tmp_path):
    binary = tmp_path / "cub_provider.dll"
    binary.write_bytes(b"provider-binary")
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    path = _write_manifest(tmp_path, _manifest(binary.name, binary_hash))

    manifest = load_source_provider_manifest(
        path,
        expected_provider_id="cub_reference",
        expected_provider_abi="taichi-forge-cub-source-provider-c-abi1",
    )

    assert manifest.binary_path == binary.resolve()
    assert manifest.binary_sha256 == binary_hash
    assert manifest.identity["provider_id"] == "cub_reference"
    assert manifest.supports_cuda_compute_capability(89)
    assert manifest.supports_cuda_compute_capability(90)
    assert not manifest.supports_cuda_compute_capability(80)
    assert manifest.runtime_dependencies[0].linkage == "static"
    assert manifest.specializations[0]["temporary_storage"] == "caller_owned"


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda item: item.update(extra=True), "fields are not exact"),
        (lambda item: item["binary"].update(path="../escape.dll"), "escapes"),
        (lambda item: item["binary"].update(sha256="0" * 64), "SHA-256 mismatch"),
        (lambda item: item["toolchain"].update(target_code=["all"]), "sm_NN"),
        (lambda item: item.update(provider_abi_version=0), "positive integer"),
    ],
)
def test_source_provider_manifest_fails_closed(tmp_path, mutation, match):
    binary = tmp_path / "cub_provider.dll"
    binary.write_bytes(b"provider-binary")
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    document = _manifest(binary.name, binary_hash)
    mutation(document)
    path = _write_manifest(tmp_path, document)

    with pytest.raises(SourceProviderManifestError, match=match):
        load_source_provider_manifest(path)


def test_source_provider_manifest_rejects_provider_identity_mismatch(tmp_path):
    binary = tmp_path / "cub_provider.dll"
    binary.write_bytes(b"provider-binary")
    path = _write_manifest(
        tmp_path,
        _manifest(binary.name, hashlib.sha256(binary.read_bytes()).hexdigest()),
    )

    with pytest.raises(SourceProviderManifestError, match="does not match"):
        load_source_provider_manifest(path, expected_provider_id="mathdx_reference")
