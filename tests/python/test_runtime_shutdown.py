import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.lang import impl
from tests import test_utils


def run_probe(arch, mode, tmp_path):
    marker = tmp_path / "retired.txt"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("runtime_shutdown_probe.py")), arch, mode, str(marker)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    log = result.stdout + result.stderr
    assert result.returncode == 0, log
    assert "Exception ignored in atexit" not in log, log
    assert "Validation Error" not in log, log
    assert marker.is_file(), log
    assert marker.read_text(encoding="utf-8") == "retired", log


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("mode", ["live", "reset"])
def test_automatic_shutdown_retires_live_or_reset_owner(mode, tmp_path):
    if not hasattr(core, "_borrow_aot_runtime"):
        pytest.skip("Requires a split runtime containing the C API")
    arch = impl.current_cfg().arch
    run_probe("cpu" if arch == ti.cpu else str(arch).split(".")[-1], mode, tmp_path)


def test_automatic_shutdown_without_initialization(tmp_path):
    run_probe("cpu", "uninitialized", tmp_path)
