"""Exercise cold materialization outside the Graph submission read boundary."""

import faulthandler
import sys

import numpy as np
import taichi_forge as ti


def main(arch):
    faulthandler.dump_traceback_later(20, repeat=True)
    ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
    runtime = ti.lang.impl.get_runtime()
    record = ti.types.struct(value=ti.f32, tag=ti.i32)
    source = ti.ndarray(record, 17)
    output = ti.ndarray(ti.f32, 17)
    data = np.zeros(17, dtype=source.numpy_dtype)
    data["value"] = np.arange(17)
    data["tag"] = 91
    source.from_numpy(data)
    method = dict(cpu="cpu_native", cuda="cuda_device", vulkan="vulkan_native")[arch]
    sequence = ti.algorithms.primitive_sequence().transform(
        source.field("value"), output, scale=3, bias=7, method=method)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    pacer = ti.graph.SubmissionPacer(2)

    def submit():
        graph.submit({}, pacer=pacer).wait()
        np.testing.assert_array_equal(output.to_numpy(), data["value"] * 3 + 7)
        assert runtime._active_graph_submissions == 0

    # No helper-kernel compilation or prewarm occurs before this submission.
    submit()
    pending = ti.field(ti.i32, shape=3)
    submit()
    assert pending.to_numpy().shape == (3,)
    trees = tuple(runtime.prog.get_active_snode_tree_ids())
    for iteration in (1, 2):
        data["value"] += iteration
        source.from_numpy(data)
        submit()
        assert tuple(runtime.prog.get_active_snode_tree_ids()) == trees
    invalid = ti.field(ti.i32)
    sequence_before = runtime.prog._debug_runtime_completion_stats()["next_sequence"]
    try:
        graph.submit({}, pacer=pacer)
    except RuntimeError as error:
        assert "not placed" in str(error), str(error)
    else:
        raise AssertionError("Unplaced fields must fail before starting a submission")
    assert runtime._active_graph_submissions == 0
    assert runtime.prog._debug_runtime_completion_stats()["next_sequence"] == sequence_before
    assert pacer.statistics()["in_flight"] == 0
    ti.root.dense(ti.i, 1).place(invalid)
    submit()
    graph.close()
    sequence.clear()
    faulthandler.cancel_dump_traceback_later()
    print("Cold native submission, pending roots, failure cleanup and warm reuse passed", arch)


if __name__ == "__main__":
    main(*sys.argv[1:])
