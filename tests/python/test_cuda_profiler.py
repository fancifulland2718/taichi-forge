import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils
from tests.python.test_external_cuda_context import _ambient_context


@test_utils.test(arch=ti.cuda, offline_cache=False, kernel_profiler=True)
def test_cuda_profiler_repeated_query_is_idempotent():
    value = ti.field(ti.i32, shape=())

    @ti.kernel
    def increment():
        value[None] += 1

    increment()
    ti.sync()
    ti.profiler.clear_kernel_profiler_info()

    for _ in range(3):
        increment()

    first = ti.profiler.query_kernel_profiler_info(increment.__name__)
    second = ti.profiler.query_kernel_profiler_info(increment.__name__)

    assert first.counter == 3
    assert second.counter == first.counter
    assert second.min == first.min
    assert second.max == first.max
    assert second.avg == first.avg


@pytest.mark.parametrize("mode", ["empty", "foreign"])
@test_utils.test(arch=ti.cuda, offline_cache=False, kernel_profiler=True)
def test_cuda_profiler_queries_restore_callers_context(mode):
    value = ti.field(ti.i32, shape=())

    @ti.kernel
    def increment():
        value[None] += 1

    increment()
    ti.sync()
    program = impl.get_runtime().prog
    with _ambient_context(mode) as (current, ambient, _):
        assert ti.profiler.set_kernel_profiler_toolkit("default")
        assert current() == ambient
        ti.profiler.clear_kernel_profiler_info()
        assert current() == ambient
        assert program.get_kernel_profiler_records() == []
        for _ in range(2):
            increment()
        assert current() == ambient

        first = ti.profiler.query_kernel_profiler_info(increment.__name__)
        assert first.counter == 2 and first.min >= 0
        assert current() == ambient
        second = ti.profiler.query_kernel_profiler_info(increment.__name__)
        assert (first.counter, first.min, first.max, first.avg) == (
            second.counter, second.min, second.max, second.avg
        )
        assert current() == ambient
        assert ti.profiler.get_kernel_profiler_total_time() >= 0
        assert current() == ambient
        # Event profiling does not support metric reconfiguration.
        assert not program.reinit_kernel_profiler_with_metrics([])
        assert current() == ambient
        ti.profiler.clear_kernel_profiler_info()
        assert current() == ambient
        assert program.get_kernel_profiler_records() == []
    assert value[None] == 3

