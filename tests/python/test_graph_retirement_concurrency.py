"""Bounded subprocess regressions for Python/native resource ordering."""

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
    elif mode in (
        "texture_create", "ndarray_create", "transaction_submit", "transaction_abort", "tree_destroy",
        "kernel_registration",
    ):
        holder = []
        graph.run({"data": data})
    else:
        holder = [core._prepare_vulkan_graph_recording(program, [source], args)]
        holder[0].run()
    survivor = core._prepare_vulkan_graph_recording(program, [source], args)
    if mode == "tree_destroy":
        fields = ti.FieldsBuilder()
        unrelated = ti.field(ti.i32)
        fields.dense(ti.i, 8).place(unrelated)
        holder.append(fields.finalize())
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
            elif mode == "texture_create":
                holder.append(ti.Texture(ti.Format.rgba8, (3, 2), mip_levels=2))
            elif mode in ("ndarray_create", "transaction_abort"):
                holder.append(ti.ndarray(ti.i32, 128))
            elif mode == "transaction_submit":
                nested = program._begin_runtime_submission_transaction()
                survivor.run()
                holder.append(nested._finish())
            elif mode == "tree_destroy":
                holder[0].destroy()
            elif mode == "kernel_registration":
                holder.append(program._debug_kernel_executable_lifecycle_stats())
            elif mode in ("frame_close", "graph_close"):
                holder[0].close()
            else:
                holder.clear()  # Last Python owner; exercise the native destructor.
        except BaseException as error:
            errors.append(error)

    # Keep the real native batch open across Python calls, as Graph.submit does.
    # Resource creation/retirement must wait without holding the GIL or an
    # inverse registry lock. Merely prewarming allocations cannot fix this.
    transaction = program._begin_runtime_submission_transaction()
    worker = threading.Thread(target=retire)
    worker.start()
    assert started.wait(5)
    worker.join(0.1)  # Allow the competing retirement to enter its native wait.
    assert not errors, errors
    assert worker.is_alive(), "resource operation should be ordered after the open batch"
    survivor.run()
    if mode == "transaction_abort":
        transaction._abort()
        completion = None
    else:
        completion = transaction._finish()
    worker.join(5)
    assert not worker.is_alive()
    assert not errors
    if completion is not None:
        completion.wait()
    else:
        ti.sync()
    expected = 6 if mode == "transaction_submit" else 4
    np.testing.assert_array_equal(data.to_numpy(), np.full(256, expected, dtype=np.int32))
    if mode == "texture_create":

        @ti.kernel
        def write_image(image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8)):
            for x, y in image:
                image.store(ti.Vector([x, y]), ti.Vector([1.0, 0.0, 1.0, 1.0]))

        @ti.kernel
        def read_image(image: ti.types.texture(num_dimensions=2), result: ti.types.ndarray()):
            for x, y in ti.ndrange(3, 2):
                value = image.fetch(ti.Vector([x, y]), 0)
                for c in ti.static(range(4)):
                    result[x, y, c] = value[c]

        result = ti.ndarray(ti.f32, (3, 2, 4))
        write_image(holder[0])
        read_image(holder[0], result)
        np.testing.assert_array_equal(result.to_numpy(), np.broadcast_to([1, 0, 1, 1], (3, 2, 4)))
        holder.clear()
    survivor.close()
    graph.close()
    ti.reset()
    holder.clear()  # Late cache destruction after runtime retirement remains safe.


@pytest.mark.parametrize(
    "mode",
    [
        "cache_clear", "cache_retire", "cache_destroy", "frame_close", "frame_destroy", "graph_close",
        "texture_create", "ndarray_create", "transaction_submit", "transaction_abort", "tree_destroy",
        "kernel_registration",
    ],
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
