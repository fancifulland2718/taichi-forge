import taichi_forge as ti
from tests import test_utils


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

