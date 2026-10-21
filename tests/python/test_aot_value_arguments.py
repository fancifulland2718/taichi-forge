"""Public aggregate values share ordinary/prepared storage and resource owners."""

import json
import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from taichi_forge._lib import utils
from tests import test_utils
from tests.cpp.aot.python_scripts.value_argument_aot import export


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_value_arguments_forward_adjoint_and_prepared_retirement(tmp_path):
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    seconds = export(arch, tmp_path)
    print(json.dumps(dict(arch=arch, export_seconds=seconds,
                          artifact_bytes=(tmp_path / "values.tcm").stat().st_size)))
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join([str(root / "packaging/runtime/src"), *sys.path]),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        TI_ENABLE_VERSION_CHECK="0",
    )
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(__file__).with_name("aot_value_argument_probe.py")),
             mode, arch, str(tmp_path / "values.tcm")],
            env=environment, capture_output=True, text=True, timeout=120,
        )
        output = result.stdout + result.stderr
        print(output)
        assert result.returncode == 0, output
        assert "VALUE_ARGUMENTS_PASSED" in output
        assert not any(item in output for item in ("VUID-", "Validation Error", "SYNC-HAZARD")), output
