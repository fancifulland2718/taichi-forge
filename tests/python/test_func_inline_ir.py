import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def builds(profile, name):
    return sum(
        int(row["calls"]) for row in profile.python_events() if row["path"] == "python.func.inline_ir_build:" + name
    )


def library_builds(profile, name):
    return sum(
        int(row["calls"]) for row in profile.python_events() if row["path"] == "python.func.real_transform:" + name
    )


@pytest.mark.parametrize("tier", ["fast", "full"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, jit_function_library=True)
def test_jit_function_library_nested_control_flow_and_tuple(tier):
    output = ti.Vector.field(2, ti.i32, shape=8)

    @ti.func
    def leaf(value: ti.i32) -> ti.i32:
        result = 0
        j = 0
        while j < value:
            j += 1
            if j == 3:
                continue
            if j > 5:
                break
            result += j
        return result

    @ti.func
    def shared(value: ti.i32) -> (ti.i32, ti.i32):
        a = leaf(value)
        b = leaf(value + 1)
        return a, b

    @ti.kernel(opt_level=tier)
    def run():
        for i in output:
            a, b = shared(i)
            output[i] = ti.Vector([a, b])

    with ti.compile_profile() as profile:
        run()
    assert library_builds(profile, "shared") == 1
    assert library_builds(profile, "leaf") == 1
    sums = [sum(j for j in range(1, min(i, 5) + 1) if j != 3) for i in range(9)]
    np.testing.assert_array_equal(output.to_numpy(), list(zip(sums[:-1], sums[1:])))


@test_utils.test(arch=ti.vulkan, offline_cache=False, jit_function_library=True, compile_tier="fast")
def test_jit_function_library_spirv_retains_one_typed_body(tmp_path):
    output = ti.field(ti.i32, shape=8)

    @ti.func
    def shared(value: ti.i32) -> ti.i32:
        return value * value + 1

    @ti.kernel
    def run():
        for i in output:
            value = i
            for j in ti.static(range(8)):
                value = shared(value) % 97
            output[i] = value

    run()
    module = ti.aot.Module(ti.vulkan)
    module.add_kernel(run)
    module.save(str(tmp_path))
    modules = list(tmp_path.glob("*.spv"))
    assert len(modules) == 1
    words = np.frombuffer(modules[0].read_bytes(), dtype="<u4")
    opcodes = []
    offset = 5
    while offset < len(words):
        word = int(words[offset])
        opcodes.append(word & 0xFFFF)
        offset += word >> 16
    assert offset == len(words)
    assert opcodes.count(54) == 2  # OpFunction: main plus one helper
    assert opcodes.count(57) == 8  # OpFunctionCall: eight references


@test_utils.test(arch=ti.vulkan, offline_cache=False, jit_function_library=True)
def test_jit_function_library_spirv_keeps_array_abi_inline():
    output = ti.ndarray(ti.i32, shape=8)

    @ti.func
    def shared(data: ti.types.ndarray(), i: ti.i32) -> ti.i32:
        data[i] = i + 1
        return data[i]

    @ti.kernel
    def run(data: ti.types.ndarray()):
        for i in range(8):
            data[i] = shared(data, i) + 1

    with ti.compile_profile() as profile:
        run(output)
    assert library_builds(profile, "shared") == 0
    np.testing.assert_array_equal(output.to_numpy(), np.arange(8) + 2)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, jit_function_library=True)
def test_jit_function_library_keeps_field_state_and_bindings():
    left = ti.field(ti.i32, shape=8)
    right = ti.field(ti.i32, shape=8)
    output = ti.field(ti.i32, shape=8)
    left.fill(1)
    right.fill(10)

    @ti.func
    def advance(data: ti.template(), i: ti.i32) -> ti.i32:
        data[i] += 1
        return data[i]

    @ti.kernel
    def run():
        for i in output:
            a = advance(left, i)
            b = advance(right, i)
            c = advance(left, i)
            output[i] = a * 100 + b * 10 + c

    with ti.compile_profile() as profile:
        run()
    assert library_builds(profile, "advance") == 2
    np.testing.assert_array_equal(output.to_numpy(), [313] * 8)
    np.testing.assert_array_equal(left.to_numpy(), [3] * 8)
    np.testing.assert_array_equal(right.to_numpy(), [11] * 8)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, jit_function_library=True)
def test_jit_function_library_reuses_value_body_across_calls():
    output = ti.field(ti.i32, shape=8)

    @ti.func
    def shared(value):
        for j in ti.static(range(4)):
            value += j
        return value

    @ti.kernel
    def first():
        for i in output:
            value = i
            for j in ti.static(range(8)):
                value = shared(value)
            output[i] = value

    @ti.kernel
    def second():
        for i in output:
            output[i] = shared(i) * 2

    with ti.compile_profile() as profile:
        first()
        second()
    assert library_builds(profile, "shared") == 2  # once per kernel, not per call
    native = profile.records(include_python=False)
    if ti.lang.impl.current_cfg().arch == ti.vulkan:
        # One SPIR-V body per module, despite eight calls in the first kernel.
        assert (
            sum(int(row["calls"]) for row in native if row["path"].endswith("/cpp.compile.spirv.function_body_emit"))
            == 2
        ), native
    else:
        assert (
            sum(int(row["calls"]) for row in native if row["path"].endswith("/cpp.compile.llvm.function_body_emit"))
            == 1
        ), native
        assert (
            sum(int(row["calls"]) for row in native if row["path"].endswith("/cpp.compile.llvm.function_unit_reuse"))
            == 1
        ), native
    np.testing.assert_array_equal(output.to_numpy(), (np.arange(8) + 6) * 2)
    first()
    np.testing.assert_array_equal(output.to_numpy(), np.arange(8) + 48)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, jit_function_library=True)
def test_jit_function_library_tracks_closure_and_keeps_callbacks():
    output = ti.field(ti.i32, shape=4)
    gain = 2
    seen = []

    def advance():
        nonlocal gain
        gain += 1

    def observe():
        seen.append(1)
        return len(seen)

    @ti.func
    def shared(value):
        value *= gain
        return value

    @ti.func
    def callback(value):
        value += ti.static(observe())
        return value

    @ti.kernel
    def run():
        for i in range(1):
            output[0] = shared(i + 10)
            ti.static(advance())
            output[1] = shared(i + 10)
            output[2] = callback(i)
            output[3] = callback(i)

    with ti.compile_profile() as profile:
        run()
    assert library_builds(profile, "shared") == 2
    assert library_builds(profile, "callback") == 0
    np.testing.assert_array_equal(output.to_numpy(), [20, 30, 1, 2])
    assert seen == [1, 1]


@pytest.mark.parametrize("tier", ["fast", "full"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, jit_function_library=True)
def test_jit_function_library_preserves_tensor_copies(tier):
    vec = ti.types.vector(3, ti.f32)
    mat = ti.types.matrix(3, 3, ti.f32)
    output = vec.field(shape=4)

    @ti.func
    def shared(a: mat, v: vec) -> vec:
        v[0] += 2
        a[1, 1] += 3
        return a @ v

    @ti.kernel(opt_level=tier)
    def run():
        for i in output:
            a = ti.Matrix.identity(ti.f32, 3)
            v = ti.Vector([i, i + 1, i + 2], dt=ti.f32)
            output[i] = shared(a, v) + shared(a, v) + v

    with ti.compile_profile() as profile:
        run()
    assert library_builds(profile, "shared") == 1
    np.testing.assert_array_equal(output.to_numpy(), [[3 * i + 4, 9 * (i + 1), 3 * (i + 2)] for i in range(4)])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, jit_function_library=True)
def test_jit_function_library_tracks_nested_owner_and_static_dependencies():
    @ti.data_oriented
    class Owner:
        def __init__(self):
            self.extra = False
            self.data = ti.Vector.field(2, ti.f32, shape=4)

    owner = Owner()
    owner.data.fill(2)
    output = ti.field(ti.f32, shape=4)
    gain = 2

    def advance():
        nonlocal gain
        gain = 3

    @ti.func
    def leaf(query, i, value):
        value += query.data[i].dot(ti.Vector([gain, 1]))
        if ti.static(query.extra):
            value += query.absent[i]
        return ti.max(value, 0)

    @ti.func
    def shared(query: ti.template(), i, value):
        value += leaf(query, i, value)
        return value

    @ti.kernel
    def run():
        for i in range(4):
            a = shared(owner, i, 1.0)
            b = shared(owner, i, 2.0)
            ti.static(advance())
            output[i] = a + b + shared(owner, i, 3.0)

    with ti.compile_profile() as profile:
        run()
    assert library_builds(profile, "shared") == 2
    np.testing.assert_array_equal(output.to_numpy(), [32.0] * 4)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, jit_function_library=True)
def test_jit_function_library_retains_kernel_root_parallel_loops():
    output = ti.field(ti.i32, shape=8)

    @ti.func
    def parallel(value: ti.i32) -> ti.i32:
        for i in output:
            output[i] = i + value
        return value

    @ti.kernel
    def run():
        parallel(3)

    with ti.compile_profile() as profile:
        run()
    assert library_builds(profile, "parallel") == 0
    np.testing.assert_array_equal(output.to_numpy(), np.arange(8) + 3)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, jit_function_library=True)
def test_jit_function_library_does_not_skip_python_attribute_effects():
    seen = []

    class Owner:
        gain = 2

        def __getattribute__(self, name):
            if name == "gain":
                seen.append(1)
            return object.__getattribute__(self, name)

    owner = Owner()
    output = ti.field(ti.i32, shape=4)

    @ti.func
    def shared(query: ti.template(), value):
        value += query.gain
        return value

    @ti.kernel
    def run():
        for i in output:
            output[i] = shared(owner, i) + shared(owner, i)

    with ti.compile_profile() as profile:
        run()
    assert library_builds(profile, "shared") == 0
    assert seen == [1, 1]
    np.testing.assert_array_equal(output.to_numpy(), 2 * np.arange(4) + 4)


@pytest.mark.parametrize("mode", ["forward", "reverse"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, jit_function_library=True)
def test_jit_function_library_retains_autodiff_expansion(mode):
    x = ti.field(ti.f32, shape=4, needs_grad=True, needs_dual=True)
    y = ti.field(ti.f32, shape=4, needs_grad=True, needs_dual=True)
    x.fill(0.25)

    @ti.func
    def shared(value):
        value *= value
        return value

    @ti.kernel
    def run():
        for i in x:
            y[i] = shared(shared(x[i]))

    if mode == "forward":
        with ti.ad.FwdMode(loss=y, param=x, seed=[1.0] * 4):
            run()
        derivative = y.dual.to_numpy()
    else:
        run()
        y.grad.fill(1)
        run.grad()
        derivative = x.grad.to_numpy()
    np.testing.assert_allclose(y.to_numpy(), 0.25**4)
    np.testing.assert_allclose(derivative, 4 * 0.25**3)


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
