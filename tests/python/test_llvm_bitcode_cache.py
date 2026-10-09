import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_bitcode_cache_reloads_without_compilation_in_a_fresh_process(tmp_path):
    arch = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"
    script = tmp_path / "cached_kernel.py"
    script.write_text(
        """import json
import os
from pathlib import Path
import taichi_forge as ti

ti.init(arch=getattr(ti, os.environ['CACHE_TEST_ARCH']), enable_fallback=False,
        offline_cache=True, offline_cache_file_path=os.environ['CACHE_TEST_DIR'],
        fast_math=False)
runtime = ti.lang.impl.get_runtime()
runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
runtime.debug_kernel_executable_lifecycle_stats(True)

@ti.kernel(opt_level='full')
def evaluate(a: ti.types.ndarray(dtype=ti.i32, ndim=1), base: ti.i32):
    for i in a:
        a[i] = base + i * i

values = ti.ndarray(dtype=ti.i32, shape=17)
evaluate(values, 9)
ti.sync()
stats = runtime.debug_kernel_executable_lifecycle_stats()
assert values.to_numpy().tolist() == [9 + i * i for i in range(17)]
Path(os.environ['CACHE_TEST_RESULT']).write_text(json.dumps(stats))
ti.reset()
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    # Use this test's actual Python/native pair, including staged test builds.
    env["PYTHONPATH"] = os.pathsep.join([str(Path(ti.__file__).parents[1]), env.get("PYTHONPATH", "")])
    env.update(CACHE_TEST_ARCH=arch, CACHE_TEST_DIR=str(tmp_path / "cache"), TI_SKIP_VERSION_CHECK="ON")
    stats = []
    for attempt in range(2):
        result = tmp_path / f"result-{attempt}.json"
        env["CACHE_TEST_RESULT"] = str(result)
        completed = subprocess.run(
            [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=90
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        stats.append(json.loads(result.read_text()))
    assert stats[0]["compiler_invocations"] == 1
    assert stats[1]["compiler_invocations"] == 0
    assert stats[1]["disk_loads"] == 1
    files = tuple((tmp_path / "cache").glob("*.tic"))
    assert len(files) == 1
    payload = files[0].read_bytes()
    assert payload[:4] == b"TIC\0"
    # Container fields use native byte order; the LLVM payload is self-identifying.
    metadata_size = struct.unpack_from("=Q", payload, 8)[0]
    assert payload[24 + metadata_size : 28 + metadata_size] == b"BC\xc0\xde"
