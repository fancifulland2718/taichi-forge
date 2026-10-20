"""Host boundaries drain a Graph prefix without completing its later work."""

import faulthandler
import sys

import numpy as np
import taichi_forge as ti
from taichi_forge.algorithms import _algorithms as algorithms
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode


class Parent(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, graph, frame):
        self.graph, self.frame = graph, frame

    def compile(self):
        return self

    def run(self):
        self.graph.run(self.frame)


def main(arch):
    faulthandler.dump_traceback_later(25, exit=True)
    ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
    runtime = ti.lang.impl.get_runtime()
    n = 17
    # Strided keys also exercise the automatic host fallback.
    keys, sibling = ti.field(ti.i32), ti.field(ti.i32)
    ti.root.dense(ti.i, n).place(keys, sibling)
    values = ti.Vector.field(3, ti.i32, n)
    seed = ti.field(ti.i32, ())
    output = ti.ndarray(ti.i32, n)

    @ti.kernel
    def prefix():
        for i in range(n):
            keys[i] = (16 - i + seed[None]) % 5
            sibling[i] = 123
            values[i] = ti.Vector([i, i + 100, i + 200])

    @ti.kernel
    def suffix(dst: ti.types.ndarray(ti.i32, 1)):
        for i in range(n):
            dst[i] = keys[i] * 1000 + values[i][0]

    host_sort = algorithms._host_stable_sort
    fail = [False]
    calls = []

    def observed_host(*args, **kwargs):
        host_sort(*args, **kwargs)
        calls.append(True)
        if fail[0]:
            raise RuntimeError("host stage failed after synchronization")

    algorithms._host_stable_sort = observed_host
    pacer = ti.graph.SubmissionPacer(2)
    last_ticket = None
    try:
        for method in ("host_stable", "auto"):
            sequence = ti.algorithms.primitive_sequence().sort(keys, values, method=method)
            builder = ti.graph.GraphBuilder()
            builder.dispatch(prefix)
            builder.append_native(sequence)
            builder.dispatch(suffix, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.i32, ndim=1))
            graph = builder.compile()
            frame = dict(dst=output)
            parent_builder = ti.graph.GraphBuilder()
            parent_builder.append_native(Parent(graph, frame))
            parent = parent_builder.compile()
            for offset, selected, bindings, telemetry in (
                (0, graph, frame, False), (3, parent, {}, False),
                (2, graph, frame, "timestamps"),
            ):
                seed[None] = offset
                last_ticket = selected.submit(bindings, pacer=pacer, telemetry=telemetry)
                last_ticket.wait()
                data = (16 - np.arange(n) + offset) % 5
                order = np.argsort(data, kind="stable")
                np.testing.assert_array_equal(output.to_numpy(), data[order] * 1000 + order)
                np.testing.assert_array_equal(sibling.to_numpy(), np.full(n, 123))
                assert runtime._active_graph_submissions == 0
                assert pacer.statistics()["in_flight"] == 0
                if telemetry:
                    assert last_ticket.telemetry() is not None

            output.fill(-99)
            fail[0] = True
            try:
                graph.submit(frame, pacer=pacer)
            except RuntimeError as error:
                assert "host stage failed after synchronization" in str(error)
            else:
                raise AssertionError("Expected the host failure")
            finally:
                fail[0] = False
            assert runtime._active_graph_submissions == 0
            assert pacer.statistics()["in_flight"] == 0
            np.testing.assert_array_equal(output.to_numpy(), np.full(n, -99))
            # The failed transaction restores the gate; another submit and its
            # suffix remain usable, including closure before the final wait.
            last_ticket = graph.submit(frame, pacer=pacer, wait_mode="blocking")
            parent.close()
            graph.close()
            sequence.clear()
            last_ticket.wait()
            data = (16 - np.arange(n) + 2) % 5
            order = np.argsort(data, kind="stable")
            np.testing.assert_array_equal(output.to_numpy(), data[order] * 1000 + order)
        assert len(calls) == 10
    finally:
        algorithms._host_stable_sort = host_sort
        ti.reset()
    assert last_ticket.done()
    last_ticket.wait()
    faulthandler.cancel_dump_traceback_later()
    print("Host prefix sync, suffix completion, parent, telemetry, failure and retirement passed", arch)


if __name__ == "__main__":
    main(*sys.argv[1:])
