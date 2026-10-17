"""OptiX program factory contracts; launch/export qualification is separate."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_optix_program_factory_is_cuda_only():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.native_resource_capabilities("optix_program")
        assert not runtime.probe_native_resource("optix_program")["available"]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_program_resources_and_failed_compile_retirement(tmp_path):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit common OptiX adapter required")
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_optix_program_resource_probe.py")),
                mode,
                adapter,
                str(tmp_path / f"{mode}.json"),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
