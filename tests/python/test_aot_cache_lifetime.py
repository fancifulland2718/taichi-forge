"""Cache-hit producer clones retain their native compilation context."""

import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from taichi_forge._lib import utils
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_cached_aot_producer_outlives_program(tmp_path):
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join(sys.path),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        TI_ENABLE_VERSION_CHECK="0",
        PYTHONFAULTHANDLER="1",
    )
    for phase in ("cold", "warm"):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("aot_cache_lifetime_probe.py")),
                arch,
                str(tmp_path),
                phase,
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "CACHE_PRODUCER_RETIRED" in output
        assert not any(
            item in output for item in ("VUID-", "Validation Error", "SYNC-HAZARD")
        ), output
