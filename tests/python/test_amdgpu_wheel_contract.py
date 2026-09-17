"""Distribution contracts, independent of a local HIP installation or GPU."""

from zipfile import ZipFile

import pytest

from scripts import validate_runtime_wheel as validator


def test_amdgpu_compiler_payload_is_complete_and_optional(tmp_path):
    prefix = f"{validator.PACKAGE}/_lib/runtime/"
    leaves = (
        "runtime_amdgpu.bc", "ocml.bc", "ockl.bc", "opencl.bc",
        "oclc_abi_version_500.bc", "oclc_wavefrontsize64_on.bc",
        "oclc_wavefrontsize64_off.bc", "oclc_isa_version_1100.bc",
        "AMDGPU-DEVICE-LIBS-LICENSE.txt",
    )
    for missing in (None, *leaves):
        with ZipFile(tmp_path / "payload.zip", "w") as zf:
            for leaf in leaves:
                if leaf != missing:
                    zf.writestr(prefix + leaf, b"fixture")
        with ZipFile(tmp_path / "payload.zip") as zf:
            if missing is None:
                validator._validate_amdgpu_payload(zf, zf.namelist(), True)
            else:
                with pytest.raises(RuntimeError, match="AMDGPU"):
                    validator._validate_amdgpu_payload(zf, zf.namelist(), True)
    with ZipFile(tmp_path / "old.zip", "w") as zf:
        validator._validate_amdgpu_payload(zf, [], False)
        with pytest.raises(RuntimeError, match="AMDGPU"):
            validator._validate_amdgpu_payload(zf, [], True)


def test_hip_runtime_is_neither_bundled_nor_imported(monkeypatch, tmp_path):
    for platform, dependency in (
        ("windows", "amdhip64_7.dll"), ("windows", "amdhip64.dll"),
        ("windows", "hiprtc.dll"), ("linux", "libamdhip64.so.7"),
        ("linux", "libhiprtc.so.7"), ("linux", "libhsa-runtime64.so.1"),
    ):
        assert validator.FORBIDDEN_VENDOR_RUNTIME.fullmatch(dependency)
        monkeypatch.setattr(validator, "_binary_imports", lambda *_, dep=dependency: {dep})
        with pytest.raises(RuntimeError):
            validator._validate_binary_dependencies(tmp_path / "runtime", platform)
