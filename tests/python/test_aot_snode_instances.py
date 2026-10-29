"""Fixed SNode state binding through the public runtime-only preparation API."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import taichi_forge as ti
import taichi_forge_runtime
from taichi_forge_runtime.aot import ExecutionPlan
from tests import test_utils


@test_utils.test(arch=ti.cuda, require=ti.extension.sparse, offline_cache=False)
def test_same_loaded_sparse_code_runs_independent_runtime_instances(tmp_path):
    left, right = ti.field(ti.i32), ti.field(ti.i32)
    first, second = ti.FieldsBuilder(), ti.FieldsBuilder()
    pointer = first.pointer(ti.i, 4)
    pointer.dense(ti.i, 4).place(left)
    second.bitmasked(ti.i, 4).dense(ti.i, 4).place(right)
    first.finalize()
    second.finalize()

    @ti.kernel
    def seed(base: ti.i32):
        for i in range(5):
            left[i] = base + i
        for i in range(9):
            right[i] = 2 * base + i

    @ti.kernel
    def step(drop: ti.i32):
        if drop:
            ti.deactivate(pointer, 0)
        for i in left:
            left[i] += 1
        for i in right:
            right[i] += 2

    @ti.kernel
    def read(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(16):
            out[i] = left[i]
            out[16 + i] = right[i]

    seed(999)
    artifact = ti.aot.Module()
    for kernel in (seed, step, read):
        artifact.add_kernel(kernel)
    artifact.add_field("left", left)
    artifact.add_field("right", right)
    artifact.add_execution_plan("step", ExecutionPlan(
        [dict(kind="kernel", entry="step", arguments=["drop"]),
         dict(kind="kernel", entry="read", arguments=["out"])],
        initialization=[dict(kind="kernel", entry="read", arguments=["empty"]),
                        dict(kind="kernel", entry="seed", arguments=["base"])],
    ))
    folder = tmp_path / "artifact"
    artifact.save(folder)
    frozen = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.rglob("*") if p.is_file()}

    # Physically exclude the DSL and this producer's numerical source from the
    # consumer package, rather than merely avoiding imports in the happy path.
    package = tmp_path / "consumer_package"
    shutil.copytree(Path(taichi_forge_runtime.__file__).parent, package / "taichi_forge_runtime",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    consumer = tmp_path / "consume.py"
    shutil.copy2(Path(__file__).with_name("aot_snode_instance_probe.py"), consumer)
    environment = dict(os.environ, FORGE_AOT_TEST_PACKAGE=str(package))
    # Use only the runtime package's matching embedded bitcode, not a developer
    # tree inherited from the producer test process.
    environment["TI_LIB_DIR"] = str(package / "taichi_forge_runtime/_lib/runtime")
    result = subprocess.run([sys.executable, "-I", "-S", str(consumer), str(folder)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PUBLIC_SNODE_INSTANCES_OK" in result.stdout
    assert frozen == {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in folder.rglob("*") if p.is_file()}
