"""Interleave module loading, JIT tree creation, retirement and ID reuse."""

import sys
import json
from pathlib import Path
import time

import taichi_forge as ti
from tests.python.aot_optix_probe import read


def main(arch, artifact):
    ti.init(arch=getattr(ti, arch), offline_cache=False)
    before = ti.field(ti.i32, shape=4)
    before[0] = 11
    with ti.aot.borrow_runtime() as runtime:
        pending = ti.field(ti.i32)
        pending_builder = ti.FieldsBuilder()
        pending_builder.dense(ti.i, 4).place(pending)
        pending_tree = pending_builder._finalize_for_aot()
        load_ms = []

        def load():
            started = time.perf_counter()
            module = runtime.load_module(artifact)
            load_ms.append((time.perf_counter() - started) * 1000)
            return module

        first = load()
        second = load()
        out = runtime.ndarray("i32", 1)
        first_step = first.kernel("step").bind(out)
        second_step = second.kernel("step").bind(out)

        def check(call, expected):
            call()
            assert read(out, "i") == [expected]

        check(first_step, 1)
        pending[0] = 21  # LLVM storage is only materialized after both modules.
        late = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, 4).place(late)
        tree = builder.finalize()
        late[0] = 31
        check(first_step, 2)
        check(second_step, 1)
        assert before[0] == 11 and pending[0] == 21 and late[0] == 31
        first.close()
        replacement = load()
        check(replacement.kernel("step").bind(out), 1)
        tree.destroy()
        third = load()  # Claims a retired JIT slot.
        after = ti.field(ti.i32, shape=4)
        after[0] = 41
        check(second_step, 2)
        check(replacement.kernel("step").bind(out), 2)
        check(third.kernel("step").bind(out), 1)
        assert before[0] == 11 and pending[0] == 21 and after[0] == 41

        x = ti.field(ti.f32, shape=2, needs_grad=True, needs_dual=True)
        y = ti.field(ti.f32, shape=2, needs_grad=True, needs_dual=True)

        @ti.kernel
        def square():
            for i in x:
                y[i] = x[i] * x[i]

        def check_gradients():
            for i in range(2):
                x[i], x.grad[i], y.grad[i] = i + 2, 0, 1
            square()
            square.grad()
            assert [y[i] for i in range(2)] == [4, 9]
            assert [x.grad[i] for i in range(2)] == [4, 6]
            with ti.ad.FwdMode(loss=y, param=x, seed=[1, 1]):
                square()
            assert [y.dual[i] for i in range(2)] == [4, 6]

        check_gradients()
        replacement.close()
        second.close()
        third.close()
        pending_tree.destroy()
        check_gradients()
        assert before[0] == 11 and after[0] == 41
    ti.reset()
    Path(artifact).with_name("consumer.json").write_text(json.dumps({"load_ms": load_ms}))


if __name__ == "__main__":
    main(*sys.argv[1:])
