import hashlib
import json
import os

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from taichi_forge.hardware._cub_source_provider import load_cub_source_provider
from taichi_forge.hardware._retained import retained_execution_contract
from taichi_forge.hardware._source_provider import (
    SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION,
    SourceProviderManifestError,
    load_source_provider_manifest,
)
from taichi_forge.lang import impl


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


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cub_source_provider_executes_explicit_primitives_and_graph():
    manifest_path = os.environ.get("TI_FORGE_TEST_CUB_SOURCE_PROVIDER_MANIFEST")
    if not manifest_path:
        pytest.skip("a user-built CUB source-provider manifest was not supplied")

    provider = load_cub_source_provider(manifest_path)
    assert load_cub_source_provider(manifest_path)._library is provider._library
    n = 257
    rng = np.random.default_rng(20260827)
    keys_u32_values = rng.integers(0, 19, size=n, dtype=np.uint32)
    values = np.arange(n, dtype=np.uint32)
    keys_u32 = ti.ndarray(ti.u32, shape=n)
    values_in = ti.ndarray(ti.u32, shape=n)
    keys_u32_out = ti.ndarray(ti.u32, shape=n)
    values_out = ti.ndarray(ti.u32, shape=n)
    keys_u32.from_numpy(keys_u32_values)
    values_in.from_numpy(values)
    sort_u32 = provider.plan("radix_sort_pairs_u32", n)
    program = impl.get_runtime().prog
    native_before = program._runtime_statistics_snapshot()["submission"][
        "native_submissions"
    ]
    sort_u32.run(
        keys_in=keys_u32,
        values_in=values_in,
        keys_out=keys_u32_out,
        values_out=values_out,
    )
    native_after = program._runtime_statistics_snapshot()["submission"][
        "native_submissions"
    ]
    assert native_after == native_before + 1
    ti.sync()
    stable_order = np.argsort(keys_u32_values, kind="stable")
    np.testing.assert_array_equal(keys_u32_out.to_numpy(), keys_u32_values[stable_order])
    np.testing.assert_array_equal(values_out.to_numpy(), values[stable_order])

    keys_u64_values = (
        rng.integers(0, 31, size=n, dtype=np.uint64) << np.uint64(33)
    ) | rng.integers(0, 37, size=n, dtype=np.uint64)
    keys_u64 = ti.ndarray(ti.u64, shape=n)
    keys_u64_out = ti.ndarray(ti.u64, shape=n)
    keys_u64.from_numpy(keys_u64_values)
    sort_u64 = provider.plan("radix_sort_pairs_u64", n)
    sort_u64.run(
        keys_in=keys_u64,
        values_in=values_in,
        keys_out=keys_u64_out,
        values_out=values_out,
    )
    ti.sync()
    stable_order = np.argsort(keys_u64_values, kind="stable")
    np.testing.assert_array_equal(keys_u64_out.to_numpy(), keys_u64_values[stable_order])
    np.testing.assert_array_equal(values_out.to_numpy(), values[stable_order])

    scan_values = rng.integers(0, 7, size=n, dtype=np.uint32)
    scan_in = ti.ndarray(ti.u32, shape=n)
    scan_out = ti.ndarray(ti.u32, shape=n)
    scan_in.from_numpy(scan_values)
    scan = provider.plan("exclusive_scan_u32", n)
    scan.run(input=scan_in, output=scan_out)
    ti.sync()
    expected_scan = np.zeros(n, dtype=np.uint32)
    expected_scan[1:] = np.cumsum(scan_values[:-1], dtype=np.uint32)
    np.testing.assert_array_equal(scan_out.to_numpy(), expected_scan)

    flags_values = rng.integers(0, 2, size=n, dtype=np.uint32)
    flags = ti.ndarray(ti.u32, shape=n)
    selected = ti.ndarray(ti.u32, shape=n)
    selected_count = ti.ndarray(ti.u32, shape=1)
    flags.from_numpy(flags_values)
    select = provider.plan("select_flagged_u32", n)
    select.run(input=values_in, flags=flags, output=selected, count=selected_count)
    ti.sync()
    expected_selected = values[flags_values != 0]
    assert int(selected_count.to_numpy()[0]) == len(expected_selected)
    np.testing.assert_array_equal(
        selected.to_numpy()[: len(expected_selected)], expected_selected
    )

    scan_out.fill(0)
    builder = ti.graph.GraphBuilder()
    builder.append_native(scan)
    graph = builder.compile()
    for _ in range(3):
        graph.run({"input": scan_in, "output": scan_out})
    ti.sync()
    np.testing.assert_array_equal(scan_out.to_numpy(), expected_scan)
    assert graph._debug_info["native_count"] == 1
    assert graph._spec.lifetime_leases

    retained = retained_execution_contract(scan)
    assert retained.identity.provider_id == "cub_reference"
    assert retained.identity.to_dict()["problem_scope"] == {
        "num_items": n,
        "operation": "exclusive_scan_u32",
    }
    assert tuple(item.name for item in retained.cost_model.fixed_costs) == (
        "manifest_and_binary_validation",
        "provider_library_load",
        "workspace_query_and_allocation",
        "ctypes_dispatch",
        "submission_registration",
    )
    assert retained.cost_model.scale_costs[0].dimensions == ("num_items",)
    assert scan.workspace_bytes > 0

    with pytest.raises(RuntimeError, match="must not alias"):
        scan.run(input=scan_in, output=scan_in)
    wrong = ti.ndarray(ti.u64, shape=n)
    with pytest.raises(RuntimeError, match="compact scalar"):
        scan.run(input=wrong, output=scan_out)
