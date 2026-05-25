import taichi_forge as ti
from tests import test_utils

_ARCHES = [ti.cpu, ti.cuda, ti.vulkan]
_EXCLUDE = [(ti.vulkan, "Darwin")]


@test_utils.test(arch=_ARCHES, exclude=_EXCLUDE)
def test_parallel_compile_precompiles_without_launching():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def set_x(v: ti.i32):
        x[None] = v

    assert ti.parallel_compile([(set_x, (7,))]) == 1
    assert x[None] == 0

    set_x(7)
    assert x[None] == 7


@test_utils.test(arch=_ARCHES, exclude=_EXCLUDE)
def test_parallel_compile_accepts_kwargs_task():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def set_x(a: ti.i32, b: ti.i32):
        x[None] = a * 10 + b

    assert ti.compile_kernels([(set_x, (), {"b": 2, "a": 3})]) == 1
    assert x[None] == 0

    set_x(3, 2)
    assert x[None] == 32


@test_utils.test(arch=_ARCHES, exclude=_EXCLUDE)
def test_compile_profile_captures_python_frontend_events():
    x = ti.field(ti.i32, shape=())

    @ti.func
    def add_one(v):
        return v + 1

    @ti.kernel
    def set_x(v: ti.i32):
        x[None] = add_one(v)

    with ti.compile_profile() as prof:
        ti.parallel_compile([(set_x, (5,))])

    rows = prof.python_events()
    paths = [row["path"] for row in rows]
    assert any(path.startswith("python.frontend.set_x.") for path in paths)
    assert any(path == "python.kernel.ast_transform:set_x" for path in paths)
    assert any(path == "python.func.inline_transform:add_one" for path in paths)
    assert any(path == "python.parallel_compile.submit" for path in paths)
    assert prof.top_n(5)
