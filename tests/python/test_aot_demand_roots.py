"""Late directory corruption cannot publish a root or poison another owner."""

from contextlib import ExitStack, closing
import json
import struct

import pytest
import taichi_forge as ti
from taichi_forge_runtime import aot
from tests import test_utils


@pytest.mark.parametrize("mode", ("standalone", "borrowed"))
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_demand_root_failure_retires_only_its_module(tmp_path, mode):
    state = ti.field(ti.i32, shape=4)
    state.fill(5)

    @ti.kernel
    def advance(out: ti.types.ndarray(ti.i32, ndim=1)):
        for i in state:
            out[i] = state[i]
            state[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1))
    graph = builder.compile()
    producer = ti.aot.Module()
    producer.add_execution_plan("step", graph, snapshot_state=True)
    producer.save(tmp_path)
    graph.close()
    metadata = json.loads((tmp_path / "aot_metadata.json").read_text())
    snapshot = metadata["initial_root_data"][0]
    payload = tmp_path / "data" / (snapshot["sha256"] + ".bin")
    original = payload.read_bytes()
    arch = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"

    def read(array):
        return struct.unpack("<4i", array.read())

    owner = ti.aot.borrow_runtime() if mode == "borrowed" else aot.Runtime(arch)
    with owner as runtime, ExitStack() as owners:
        failed = owners.enter_context(runtime.load_module(tmp_path))
        survivor = owners.enter_context(runtime.load_module(tmp_path))
        output = owners.enter_context(closing(runtime.ndarray("i32", 4)))
        sentinel = owners.enter_context(closing(runtime.ndarray("i32", 4)))
        sentinel.write(struct.pack("<4i", *([-99] * 4)))
        good = owners.enter_context(closing(survivor.prepare_execution("step", {"out": output})))
        good()
        assert read(output) == (5,) * 4
        try:
            payload.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with pytest.raises(RuntimeError, match="integrity validation"):
                failed.prepare_execution("step", {"out": sentinel})
            assert read(sentinel) == (-99,) * 4
        finally:
            payload.write_bytes(original)
        with pytest.raises(RuntimeError, match="invalid|corrupt"):
            failed.prepare_execution("step", {"out": sentinel})
        failed.close()
        good()
        assert read(output) == (6,) * 4
        with runtime.load_module(tmp_path) as replacement:
            with replacement.prepare_execution("step", {"out": sentinel}) as call:
                call()
                assert read(sentinel) == (5,) * 4
        good()
        assert read(output) == (7,) * 4
