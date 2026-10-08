import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def builds(profile, name):
    return sum(
        int(row["calls"]) for row in profile.python_events() if row["path"] == "python.func.inline_ir_build:" + name
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, inline_ir_cache=True)
def test_inline_ir_reuses_specializations_and_rebinds_locals():
    out = ti.field(ti.f32, shape=4)

    @ti.func
    def helper(x, variant: ti.template()):
        for j in ti.static(range(4)):
            x = x * 0.9 + (j + 1) * 0.01
        if x > 0.5:
            x = x - 0.1
        if ti.static(variant):
            x = x + 0.25
        return x

    @ti.kernel
    def run():
        for i in out:
            x = ti.cast(i, ti.f32) * 0.1
            for j in ti.static(range(8)):
                x = helper(x, j % 2)
            out[i] = x

    with ti.compile_profile() as profile:
        run()
    assert builds(profile, "helper") == 2
    expected = np.arange(4, dtype=np.float32) * 0.1
    for j in range(8):
        for k in range(4):
            expected = expected * 0.9 + (k + 1) * 0.01
        expected = np.where(expected > 0.5, expected - 0.1, expected)
        expected += (j % 2) * 0.25
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=2e-6)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, inline_ir_cache=True)
def test_inline_ir_refreshes_closure_within_and_between_kernels():
    out = ti.field(ti.i32, shape=4)
    gain = 2

    def advance():
        nonlocal gain
        gain += 1
        return gain

    @ti.func
    def helper(x):
        return x * gain

    @ti.kernel
    def run(specialization: ti.template()):
        out[0] = helper(10)
        out[1] = helper(20)
        ti.static(advance())
        out[2] = helper(10)
        out[3] = helper(20)

    for specialization, start in enumerate((2, 3)):
        with ti.compile_profile() as profile:
            run(specialization)
        assert builds(profile, "helper") == 2
        np.testing.assert_array_equal(out.to_numpy(), [10 * start, 20 * start, 10 * (start + 1), 20 * (start + 1)])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, inline_ir_cache=True)
def test_inline_ir_keeps_python_constant_returns_and_callbacks():
    out = ti.field(ti.i32, shape=())
    seen = []

    def observe():
        seen.append(1)
        return len(seen)

    @ti.func
    def constant():
        return 3

    @ti.func
    def callback(x):
        offset = ti.static(observe())
        return x + offset

    @ti.kernel
    def run():
        total = 0
        for j in ti.static(range(constant())):
            total += callback(j)
        out[None] = total

    run()
    assert out[None] == 9
    assert seen == [1, 1, 1]


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, inline_ir_cache=True)
def test_inline_ir_separates_argument_types_and_preserves_casts():
    out = ti.field(ti.f32, shape=3)

    @ti.func
    def helper(x):
        return x + 0.25

    @ti.func
    def cast_value(x: ti.i32) -> ti.f32:
        return x + 1

    @ti.kernel
    def run():
        out[0] = helper(1)
        out[1] = helper(1.5)
        out[2] = cast_value(2.75)

    with ti.compile_profile() as profile:
        run()
    assert builds(profile, "helper") == 2
    np.testing.assert_allclose(out.to_numpy(), [1.25, 1.75, 3.0])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, inline_ir_cache=True)
def test_inline_ir_graph_survives_template_scope_and_recompilation():
    out = ti.field(ti.i32, shape=4)

    @ti.func
    def helper(x):
        return x * 2 + 1

    @ti.kernel
    def run(offset: ti.i32):
        for i in out:
            out[i] = helper(helper(i + offset))

    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(run, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "offset", ti.i32))
    graph = graph_builder.compile()
    for offset in (2, 4):
        graph.run({"offset": offset})
        np.testing.assert_array_equal(out.to_numpy(), 4 * (np.arange(4) + offset) + 3)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, inline_ir_cache=True)
def test_inline_ir_keeps_resource_and_matrix_functions_on_ordinary_path():
    source = ti.field(ti.i32, shape=4)
    out = ti.field(ti.i32, shape=4)
    source.from_numpy(np.arange(4, dtype=np.int32))

    @ti.func
    def resource(i):
        source[i] += 1
        value = source[i]
        return value

    @ti.func
    def vector(v):
        return v.dot(v)

    @ti.kernel
    def run():
        for i in out:
            out[i] = resource(i) + resource(i) + vector(ti.Vector([i, 1]))

    with ti.compile_profile() as profile:
        run()
    assert builds(profile, "resource") == 0
    assert builds(profile, "vector") == 0
    indices = np.arange(4)
    np.testing.assert_array_equal(source.to_numpy(), indices + 2)
    np.testing.assert_array_equal(out.to_numpy(), indices * indices + 2 * indices + 4)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, inline_ir_cache=True)
def test_inline_ir_compilation_error_does_not_retain_kernel_scope():
    out = ti.field(ti.f32, shape=())

    @ti.func
    def helper(x, fail: ti.template()):
        if ti.static(fail):
            x = ti.static(1 / 0)
        return x + 1

    @ti.kernel
    def run(fail: ti.template()):
        out[None] = helper(2.0, fail) + helper(3.0, fail)

    with pytest.raises(ti.TaichiCompilationError, match="division by zero"):
        run(True)
    run(False)
    assert out[None] == 7


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, inline_ir_cache=True)
def test_inline_ir_templates_retire_with_field_specializations():
    @ti.func
    def helper(x):
        return x * 2 + 1

    @ti.kernel
    def run(out: ti.template()):
        for i in out:
            out[i] = helper(helper(i))

    for size in (4, 6):
        out = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, size).place(out)
        tree = builder.finalize()
        run(out)
        np.testing.assert_array_equal(out.to_numpy(), 4 * np.arange(size) + 3)
        tree.destroy()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, inline_ir_cache=False)
def test_inline_ir_can_be_disabled():
    out = ti.field(ti.i32, shape=())

    @ti.func
    def helper(x):
        return x + 1

    @ti.kernel
    def run():
        out[None] = helper(helper(2))

    with ti.compile_profile() as profile:
        run()
    assert builds(profile, "helper") == 0
    assert out[None] == 4


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, inline_ir_cache=True)
def test_inline_ir_preserves_returned_private_local_lvalues():
    out = ti.field(ti.i32, shape=2)

    @ti.func
    def identity(x):
        return x

    @ti.kernel
    def run():
        out[0] = ti.atomic_add(identity(7), 2)
        out[1] = ti.atomic_add(identity(11), 3)

    run()
    np.testing.assert_array_equal(out.to_numpy(), [7, 11])


@pytest.mark.parametrize("mode", ["forward", "reverse"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, inline_ir_cache=True)
def test_inline_ir_autodiff_sees_expanded_body(mode):
    x = ti.field(ti.f32, shape=4, needs_grad=True, needs_dual=True)
    y = ti.field(ti.f32, shape=4, needs_grad=True, needs_dual=True)
    x.fill(0.125)

    @ti.func
    def helper(v):
        value = v * v + v
        return value

    @ti.kernel
    def run():
        for i in range(4):
            y[i] = helper(helper(helper(x[i])))

    expected, derivative = 0.125, 1.0
    for _ in range(3):
        derivative *= 2 * expected + 1
        expected = expected * expected + expected
    with ti.compile_profile() as profile:
        if mode == "forward":
            with ti.ad.FwdMode(loss=y, param=x, seed=[1.0] * 4):
                run()
        else:
            run()
            y.grad.fill(1)
            run.grad()
    assert builds(profile, "helper") == (1 if mode == "forward" else 2)
    np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-6)
    actual = y.dual.to_numpy() if mode == "forward" else x.grad.to_numpy()
    np.testing.assert_allclose(actual, derivative, rtol=1e-6)
