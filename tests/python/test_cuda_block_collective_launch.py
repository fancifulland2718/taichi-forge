"""Host range pruning retains shared-memory lanes and mutable replay bounds."""
from contextlib import ExitStack

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False, saturating_grid_dim=2)
def test_shared_blocks_cover_grid_stride_and_loaded_instances(tmp_path):
    @ti.kernel
    def reduce_blocks(out: ti.types.ndarray(ti.i32, 1), blocks: ti.i32):
        ti.loop_config(block_dim=32)
        for work in range(blocks * 32):
            shared = ti.simt.block.SharedArray((32,), ti.i32)
            lane, group = work % 32, work // 32
            shared[lane] = group * 32 + lane
            ti.simt.block.sync()
            # Both shared memory and a warp intrinsic retain the same lanes.
            first = ti.simt.warp.shfl_sync_i32(ti.u32(0xffffffff), work, 0)
            mask = ti.u32(ti.simt.warp.ballot(ti.cast(lane >= 0, ti.i32)))
            valid = ti.simt.warp.all_nonzero(mask, ti.cast(work >= 0, ti.i32))
            if lane == 0:
                total = 0
                for k in range(32):
                    total += shared[k]
                out[group] = total + first + valid + ti.cast(mask == ti.u32(0xffffffff), ti.i32)
            ti.simt.block.sync()

    builder = ti.graph.GraphBuilder()
    builder.dispatch(reduce_blocks,
                     ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1),
                     ti.graph.Arg(ti.graph.ArgKind.SCALAR, "blocks", ti.i32))
    graph = builder.compile()
    output = ti.ndarray(ti.i32, 7)

    def expected(count):
        result = np.full(7, -91, np.int32)
        result[:count] = np.arange(count, dtype=np.int32) * 1056 + 498
        return result

    for count in (0, 1, 2, 7, 1):
        for invoke in (lambda: reduce_blocks(output, count),
                       lambda: graph.run(dict(out=output, blocks=count))):
            output.fill(-91)
            invoke()
            np.testing.assert_array_equal(output.to_numpy(), expected(count))
    tasks = reduce_blocks.task_manifest(output, 1)
    assert len(tasks) == 1
    assert tasks[0].range_bounds.empty_group_elision == "cuda_block_collective"
    assert tasks[0].actual_geometry_kind == "host_argument_bound"
    assert tasks[0].device_range_dispatch_version == 0

    # Mutable values must retain saturation at prepare, including when the
    # initial extent is zero. New values can cross the original grid width.
    with ti.graph.execution_domain(capacity=1) as domain:
        with graph.prepare(dict(out=output, blocks=0), execution_domain=domain,
                           mutable_parameters=("blocks",)) as call:
            call.initialize()
            for count in (0, 1, 7, 2):
                output.fill(-91)
                ticket = call.submit(values=dict(blocks=count))
                ticket.wait()
                ticket.close()
                np.testing.assert_array_equal(output.to_numpy(), expected(count))

    module = ti.aot.Module()
    module.add_execution_plan("blocks", graph)
    artifact = tmp_path / "blocks.tcm"
    module.archive(str(artifact))
    from taichi_forge_runtime.aot import Scalar
    with ExitStack() as scope:
        runtime = scope.enter_context(ti.aot.borrow_runtime())
        loaded = scope.enter_context(runtime.load_module(artifact))
        calls, outputs = [], []
        for count in (1, 7):
            out = runtime.ndarray("i32", (7,))
            scope.callback(out.close)
            out.write(np.full(7, -91, np.int32))
            outputs.append(out)
            calls.append(scope.enter_context(loaded.prepare_execution(
                "blocks", dict(out=out, blocks=Scalar("i32", count)))))
        for call, out, count in zip(calls, outputs, (1, 7)):
            call()
            np.testing.assert_array_equal(np.frombuffer(out.read(), np.int32), expected(count))
        calls[0].close()
        calls[1]()
        np.testing.assert_array_equal(np.frombuffer(outputs[1].read(), np.int32), expected(7))
    graph.close()
