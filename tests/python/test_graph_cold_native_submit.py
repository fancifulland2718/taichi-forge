"""Fresh-process guard against lifecycle read-to-write deadlocks."""

import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_cold_native_submit_materializes_before_transaction():
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("graph_cold_native_submit_probe.py")), arch],
        env=os.environ.copy(), capture_output=True, text=True, timeout=40,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert not any(marker in output for marker in ("Validation Error", "SYNC-HAZARD", "Timeout (")), output
