"""CiG AOT numerical/ownership coverage; stream isolation has its own probe.

The ordinary execution-context suite retains the stronger independent-progress
gate. Do not interpret this narrower suite as qualification of pending external
waits: the local CiG driver serializes independent kernels behind those waits.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti

from tests import test_utils
from tests.python.test_aot_execution_context import export_execution_context_module


@pytest.mark.skipif(not os.environ.get("TI_TEST_CIG"), reason="Requires configured native CiG fixture")
@pytest.mark.parametrize("artifact_format,explicit_target", (("bc", False), ("ptx", False), ("ptx", True), ("cubin", True)))
@pytest.mark.parametrize("tier", ("fast", "balanced", "full"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cig_aot_numerics_and_external_epochs(artifact_format, explicit_target, tier, tmp_path):
    from taichi_forge_runtime.aot import cuda_context_info

    assert cuda_context_info()["context_mode"] == "vulkan_cig"
    executable = Path(os.environ["TI_TEST_CPP_EXECUTABLE"])
    directory = export_execution_context_module(tmp_path, artifact_format, explicit_target, tier)
    result = subprocess.run(
        [str(executable), "--gtest_filter=CudaExecutionContextAot.*:CudaPublicExecutionAot.*"
         "-CudaPublicExecutionAot.ExactDependenciesBusyAndReleasedInstanceLifetime:"
         "CudaExecutionContextAot.IsolatesTemporariesInsideAnInterleavedKernel"],
        env=dict(os.environ, TAICHI_F3_CONTEXT_AOT_FOLDER=str(directory)),
        capture_output=True, text=True, timeout=90,
    )
    (tmp_path / "native-numerics.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0 and "[  PASSED  ] 8 tests." in result.stdout, result.stdout + result.stderr
    for name, arguments in (
        ("aot_execution_domain_probe.py", [str(directory)]),
        ("cuda_cig_lifetime_probe.py", [str(directory), str(tmp_path / "external-epochs.json"), "100"]),
    ):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(__file__).with_name(name)), *arguments],
            capture_output=True, text=True, timeout=90,
        )
        (tmp_path / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        assert result.returncode == 0, result.stdout + result.stderr
    from tests.python.aot_execution_domain_probe import check
    with ti.aot.borrow_runtime() as borrower:
        check(borrower, directory)
