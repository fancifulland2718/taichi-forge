"""Bounded subprocess regressions for Python/native Graph retirement ordering."""

import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.lang import impl
from tests import test_utils


def _retire_during_submission(mode):
    ti.init(arch=ti.vulkan, offline_cache=False)

    @ti.kernel
    def increment(data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in data:
            data[i] += 1

    builder = ti.graph.GraphBuilder()
    arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "data", ti.i32, ndim=1)
    builder.dispatch(increment, arg)
    builder.dispatch(increment, arg)
    graph = builder.compile()
    source = graph._spec.nodes[0].compiled_graph
    program = impl.get_runtime().prog
    data = ti.ndarray(ti.i32, 256)
    data.fill(0)
    args = {"data": data.arr}
    if mode.startswith("cache"):
        holder = [core.CompiledGraphJITCache()]
        source.jit_run_cached(impl.current_cfg(), args, holder[0])
    elif mode == "graph_close":
        holder = [builder.compile()]
        holder[0].run({"data": data})
    else:
        holder = [core._prepare_vulkan_graph_recording(program, [source], args)]
        holder[0].run()
    survivor = core._prepare_vulkan_graph_recording(program, [source], args)
    ti.sync()
    started = threading.Event()
    errors = []

    def retire():
        try:
            started.set()
            if mode == "cache_clear":
                holder[0].clear_runtime_state()
            elif mode == "cache_retire":
                holder[0].retire_snode_tree_runtime_state()
            elif mode in ("frame_close", "graph_close"):
                holder[0].close()
            else:
                holder.clear()  # Last Python owner; exercise the native destructor.
        except BaseException as error:
            errors.append(error)

    # Keep the real native batch open across Python calls, as Graph.submit does.
    # Retirement must wait without holding the GIL or an inverse registry lock.
    transaction = program._begin_runtime_submission_transaction()
    worker = threading.Thread(target=retire)
    worker.start()
    assert started.wait(5)
    worker.join(0.1)  # Allow the competing retirement to enter its native wait.
    assert worker.is_alive(), "retirement should be ordered after the open batch"
    survivor.run()
    completion = transaction._finish()
    worker.join(5)
    assert not worker.is_alive()
    assert not errors
    completion.wait()
    np.testing.assert_array_equal(data.to_numpy(), np.full(256, 4, dtype=np.int32))
    survivor.close()
    graph.close()
    ti.reset()
    holder.clear()  # Late cache destruction after runtime retirement remains safe.


@pytest.mark.parametrize(
    "mode", ["cache_clear", "cache_retire", "cache_destroy", "frame_close", "frame_destroy", "graph_close"]
)
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_graph_retirement_during_native_batch(mode):
    # Load exactly the parent's native extension, including source/native local
    # development checks. A deadlock kills only this bounded child process.
    code = (
        "import sys, types\n"
        f"sys.path[:] = {json.dumps(sys.path)}\n"
        "package = types.ModuleType('taichi_forge._lib.core')\n"
        f"package.__path__ = [{json.dumps(str(Path(core.__file__).parent))}]\n"
        "sys.modules[package.__name__] = package\n"
        "from tests.python.test_graph_retirement_concurrency import _retire_during_submission\n"
        f"_retire_during_submission({mode!r})\n"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
