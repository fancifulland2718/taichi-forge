import re

import pytest
from taichi_forge.lang.misc import get_host_arch_list

import taichi_forge as ti
from tests import test_utils


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_minimal():
    @ti.kernel
    def func():
        assert 0

    @ti.kernel
    def func2():
        assert False

    with pytest.raises(AssertionError):
        func()
    with pytest.raises(AssertionError):
        func2()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_basic():
    @ti.kernel
    def func():
        x = 20
        assert 10 <= x < 20

    with pytest.raises(AssertionError):
        func()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message():
    @ti.kernel
    def func():
        x = 20
        assert 10 <= x < 20, "Foo bar"

    with pytest.raises(AssertionError, match="Foo bar"):
        func()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message_formatted():
    x = ti.field(dtype=int, shape=16)
    x[10] = 42

    @ti.kernel
    def assert_formatted():
        for i in x:
            assert x[i] == 0, "x[%d] expect=%d got=%d" % (i, 0, x[i])

    @ti.kernel
    def assert_float():
        y = 0.5
        assert y < 0, "y = %f" % y

    with pytest.raises(AssertionError, match=r"x\[10\] expect=0 got=42"):
        assert_formatted()
    with pytest.raises(AssertionError, match=r"y = 0.5"):
        assert_float()

    # success case
    x[10] = 0
    assert_formatted()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message_formatted_fstring():
    x = ti.field(dtype=int, shape=16)
    x[10] = 42

    @ti.kernel
    def assert_formatted():
        for i in x:
            assert x[i] == 0, f"x[{i}] expect={0} got={x[i]}"

    @ti.kernel
    def assert_float():
        y = 0.5
        assert y < 0, f"y = {y}"

    with pytest.raises(AssertionError, match=r"x\[10\] expect=0 got=42"):
        assert_formatted()
    with pytest.raises(AssertionError, match=r"y = 0.5"):
        assert_float()

    # success case
    x[10] = 0
    assert_formatted()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_ok():
    @ti.kernel
    def func():
        x = 20
        assert 10 <= x <= 20

    func()


@test_utils.test(
    arch=ti.cpu,
    debug=True,
    gdb_trigger=False,
    cpu_max_num_threads=1,
)
def test_cpu_assert_cancels_current_range_and_struct_work_items():
    n = 128
    range_before = ti.field(ti.i32, shape=n)
    range_after = ti.field(ti.i32, shape=n)
    struct_before = ti.field(ti.i32, shape=n)
    struct_after = ti.field(ti.i32, shape=n)
    recovered = ti.field(ti.i32, shape=())

    @ti.kernel
    def fail_range():
        ti.loop_config(block_dim=64)
        for i in range(n):
            range_before[i] = 1
            assert i != 7, "range failure %d" % i
            range_after[i] = 1

    @ti.kernel
    def fail_struct():
        for i in struct_before:
            struct_before[i] = 1
            assert i != 7, "struct failure %d" % i
            struct_after[i] = 1

    @ti.kernel
    def recover():
        recovered[None] = 123

    with pytest.raises(ti.TaichiAssertionError, match="range failure 7"):
        fail_range()
    assert range_before[7] == 1
    assert range_after[7] == 0
    assert sum(range_before.to_numpy()) == 8
    assert sum(range_after.to_numpy()) == 7

    with pytest.raises(ti.TaichiAssertionError, match="struct failure 7"):
        fail_struct()
    assert struct_before[7] == 1
    assert struct_after[7] == 0
    assert sum(struct_before.to_numpy()) == 8
    assert sum(struct_after.to_numpy()) == 7

    recover()
    assert recovered[None] == 123


@test_utils.test(
    arch=ti.cpu,
    debug=True,
    gdb_trigger=False,
    cpu_max_num_threads=8,
)
def test_cpu_assert_keeps_first_multithreaded_fault_coherent_and_recovers():
    recovered = ti.field(ti.i32, shape=())

    @ti.kernel
    def fail_parallel():
        ti.loop_config(block_dim=1)
        for i in range(4096):
            assert i == 0, "failure i=%d pair=%d" % (i, i * 2)

    @ti.kernel
    def recover(value: ti.i32):
        recovered[None] = value

    for attempt in range(4):
        with pytest.raises(ti.TaichiAssertionError) as error:
            fail_parallel()
        match = re.search(r"failure i=(\d+) pair=(\d+)", str(error.value))
        assert match is not None
        index, pair = map(int, match.groups())
        assert pair == index * 2

        recover(attempt + 1)
        assert recovered[None] == attempt + 1


@test_utils.test(
    require=ti.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_assert_with_check_oob():
    @ti.kernel
    def func():
        n = 15
        assert n >= 0

    func()


@test_utils.test(arch=get_host_arch_list())
def test_static_assert_message():
    x = 3

    @ti.kernel
    def func():
        ti.static_assert(x == 4, "Oh, no!")

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def test_static_assert_vector_n_ok():
    x = ti.Vector.field(4, ti.f32, ())

    @ti.kernel
    def func():
        ti.static_assert(x.n == 4)

    func()


@test_utils.test(arch=get_host_arch_list())
def test_static_assert_data_type_ok():
    x = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        ti.static_assert(x.dtype == ti.f32)

    func()


@test_utils.test()
def test_static_assert_nonstatic_condition():
    @ti.kernel
    def foo():
        value = False
        ti.static_assert(value, "Oh, no!")

    with pytest.raises(ti.TaichiTypeError, match="Static assert with non-static condition"):
        foo()
