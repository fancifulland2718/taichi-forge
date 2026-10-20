"""Public source-free AoS bindings reuse the native layout and owner contracts."""

import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from taichi_forge._lib import utils
from tests import test_utils
from tests.cpp.aot.python_scripts.struct_ndarray_aot import export


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_structure_consumer_bindings_gradients_and_retirement(tmp_path):
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    export(arch, tmp_path, initialize=False)
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join([str(root / "packaging/runtime/src"), *sys.path]),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        TI_ENABLE_VERSION_CHECK="0",
    )
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("aot_struct_consumer_probe.py")),
                mode,
                arch,
                str(tmp_path),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "Validation Error" not in output and "SYNC-HAZARD" not in output, output
