"""Loaded-entry Graph composition without recovering Python numerical code."""

import array
import gc

import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.lang import kernel_impl
from taichi_forge_runtime.aot import AotError, Runtime
from tests import test_utils


def _export(tmp_path):
    @ti.kernel
    def advance(state: ti.types.ndarray(ti.i32, 1), delta: ti.i32):
        for i in state:
            state[i] = state[i] * 2 + delta

    count = ti.field(ti.i32, shape=())

    @ti.kernel
    def next_count(out: ti.types.ndarray(ti.i32, 1)):
        count[None] += 1
        out[0] = count[None]

    @ti.kernel
    def returns_value() -> ti.i32:
        return 7

    @ti.kernel
    def tick():
        count[None] += 10

    @ti.kernel
    def aliases(left: ti.types.ndarray(ti.i32, 1), right: ti.types.ndarray(ti.i32, 1)):
        for i in left:
            left[i] = right[i] + 7

    @ti.kernel
    def tensor(values: ti.types.ndarray(ti.types.vector(2, ti.i32), 1)):
        for i in values:
            values[i] += ti.Vector([1, 3])

    @ti.kernel
    def scalar64(out: ti.types.ndarray(ti.i64, 1), value: ti.i64):
        out[0] = value

    module = ti.aot.Module()
    module.add_field("count", count)
    module.add_kernel(advance)
    module.add_kernel(next_count)
    module.add_kernel(returns_value)
    module.add_kernel(tick)
    module.add_kernel(aliases)
    module.add_kernel(tensor)
    if ti.lang.impl.current_cfg().arch != ti.vulkan:
        module.add_kernel(scalar64)
    artifact = tmp_path / "loaded.tcm"
    module.archive(str(artifact))
    # The consumer has no live numerical definitions or producer Field roots.
    arch = ti.lang.impl.current_cfg().arch
    ti.reset()
    ti.init(arch=arch, enable_fallback=False, offline_cache=False, device_memory_GB=0.3)
    return artifact


def _symbol(name="state", dtype=None, ndim=1):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype or ti.i32, ndim=ndim)


def _read(runtime, state, dtype="i"):
    imported = runtime.import_ndarray(state)
    try:
        values = array.array(dtype)
        values.frombytes(imported.read())
        return values.tolist()
    finally:
        imported.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_loaded_graph_order_bindings_and_retirement(tmp_path, monkeypatch):
    if not hasattr(core, "_borrow_aot_runtime"):
        pytest.skip("Requires a split runtime containing the C API")
    artifact = _export(tmp_path)
    runtime = ti.aot.borrow_runtime()
    module = runtime.load_module(artifact)
    entry = module.kernel("advance")
    state = ti.ndarray(ti.i32, 19)
    state.fill(0)
    state_arg = _symbol()
    delta = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "delta", ti.i32)

    @ti.kernel
    def shift(values: ti.types.ndarray(ti.i32, 1)):
        for i in values:
            values[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(shift, state_arg)
    builder.dispatch_loaded(entry, state_arg, delta)
    builder.dispatch(shift, state_arg)
    original = builder.compile()
    graph = original.definition.compile()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Loaded Graph replay must not compile Python numerical kernels or wait on Runtime")

    prepares = []
    prepare = entry.prepare

    def counted(*args):
        prepares.append(None)  # Count without retaining imported storage views.
        return prepare(*args)

    monkeypatch.setattr(entry, "prepare", counted)
    with monkeypatch.context() as guard:
        guard.setattr(kernel_impl.Kernel, "ensure_compiled", forbidden)
        binding = graph.bind({"state": state, "delta": 3})
        # bind prepares but does not execute; _read is a native storage read.
        assert _read(runtime, state) == [0] * 19
        with guard.context() as hot:
            hot.setattr(runtime, "wait", forbidden)
            tickets = [graph.submit(binding), graph.submit(binding)]
            tickets[-1].wait()
        assert _read(runtime, state) == [18] * 19
        assert len(prepares) == 1
        rebound = graph.bind({"state": state, "delta": 5})
        graph.submit(rebound).wait()
        assert _read(runtime, state) == [44] * 19
        # Raw binding preparation must precede the submission lock; its packet
        # remains alive through completion rather than synchronizing in submit.
        with guard.context() as hot:
            hot.setattr(runtime, "wait", forbidden)
            ticket = graph.submit({"state": state, "delta": 3})
            ticket.wait()
        assert _read(runtime, state) == [94] * 19
        assert len(prepares) == 3
        assert not runtime._pending

    # Transient frames must retire after completion rather than accumulating in
    # the standalone AOT pending registry. Test reclamation outside hot submit.
    del ticket
    gc.collect()
    live = len(runtime._resources)
    for _ in range(4):
        transient = graph.submit({"state": state, "delta": 3})
        transient.wait()
        del transient
    gc.collect()
    assert len(runtime._resources) == live
    assert not runtime._pending

    # Closing code while queued retires it. A replacement with the same name
    # cannot silently rebind the old Graph, even if a native address is reused.
    pending = graph.submit(binding)
    module.close()
    pending.wait()
    replacement = runtime.load_module(artifact)
    before = _read(runtime, state)
    with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="closed"):
        graph.submit(binding)
    assert _read(runtime, state) == before  # no prefix side effects
    fresh = ti.graph.GraphBuilder().dispatch_loaded(replacement.kernel("next_count"), state_arg).compile()
    fresh.submit({"state": state}).wait()
    assert _read(runtime, state)[0] == 1
    pending = fresh.submit({"state": state})
    fresh.close()
    pending.wait()
    assert _read(runtime, state)[0] == 2
    graph.close()
    original.close()
    runtime.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_loaded_graph_rejects_wrong_owner_and_abi(tmp_path):
    if not hasattr(core, "_borrow_aot_runtime"):
        pytest.skip("Requires a split runtime containing the C API")
    artifact = _export(tmp_path)
    runtime = ti.aot.borrow_runtime()
    module = runtime.load_module(artifact)
    entry = module.kernel("advance")
    symbol = _symbol()
    delta = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "delta", ti.i32)
    with pytest.raises(ti.TaichiCompilationError, match="decorated Taichi kernel"):
        ti.graph.GraphBuilder().dispatch(entry, symbol, delta)
    with pytest.raises(TypeError, match="loaded AOT Kernel"):
        ti.graph.GraphBuilder().dispatch_loaded(lambda: None)
    with pytest.raises(ValueError, match="count"):
        ti.graph.GraphBuilder().dispatch_loaded(entry, symbol)
    private_builder = ti.graph.GraphBuilder()
    private = private_builder.private_ndarray("private", ti.i32, (4,))
    with pytest.raises(ti.TaichiRuntimeError, match="public Graph arguments"):
        private_builder.dispatch_loaded(entry, private, delta)
    bad_scalar = (
        ti.graph.GraphBuilder()
        .dispatch_loaded(entry, symbol, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "delta", ti.f32))
        .compile()
    )
    with pytest.raises(AotError, match="(?i)(type|dtype|argument)"):
        bad_scalar.bind({"state": ti.ndarray(ti.i32, 4), "delta": 1.0})
    bad_scalar.close()
    with pytest.raises(ti.TaichiRuntimeError, match="void"):
        ti.graph.GraphBuilder().dispatch_loaded(module.kernel("returns_value"))
    graph = ti.graph.GraphBuilder().dispatch_loaded(entry, symbol, delta).compile()
    wrong = ti.ndarray(ti.f32, 4)
    with pytest.raises(ti.TaichiRuntimeError, match="symbol ABI"):
        graph.bind({"state": wrong, "delta": 1})
    # A false declaration also reaches the native artifact ABI validator.
    forged = ti.graph.GraphBuilder().dispatch_loaded(entry, _symbol(dtype=ti.f32), delta).compile()
    with pytest.raises(AotError, match="(?i)(type|dtype|argument)"):
        forged.bind({"state": wrong, "delta": 1})
    arch = str(runtime.arch)
    with Runtime(arch) as independent:
        foreign = independent.load_module(artifact)
        with pytest.raises(ti.TaichiRuntimeError, match="borrowed from the current Program"):
            ti.graph.GraphBuilder().dispatch_loaded(foreign.kernel("advance"), symbol, delta)
    exporting = ti.aot.Module()
    with pytest.raises((RuntimeError, ValueError, ti.TaichiRuntimeError)):
        exporting.add_graph("live_owner", graph)
    graph.close()
    forged.close()
    runtime.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_loaded_graph_zero_arguments_aliases_and_reset(tmp_path, monkeypatch):
    if not hasattr(core, "_borrow_aot_runtime"):
        pytest.skip("Requires a split runtime containing the C API")
    artifact = _export(tmp_path)
    runtime = ti.aot.borrow_runtime()
    module = runtime.load_module(artifact)
    state = ti.ndarray(ti.i32, 5)
    vector = ti.Vector.ndarray(2, ti.i32, 5)
    state.fill(0)
    vector.fill(0)
    tick = ti.graph.GraphBuilder().dispatch_loaded(module.kernel("tick")).compile()
    empty = tick.bind({})
    tick.submit(empty)
    tick.submit({}).wait()
    read = ti.graph.GraphBuilder().dispatch_loaded(module.kernel("next_count"), _symbol()).compile()
    read.run({"state": state})
    assert _read(runtime, state) == [21, 0, 0, 0, 0]
    alias_entry = module.kernel("aliases")
    alias = ti.graph.GraphBuilder().dispatch_loaded(alias_entry, _symbol("left"), _symbol("right")).compile()
    imported = []
    import_array = runtime.import_ndarray

    def count_import(value):
        imported.append(value)
        return import_array(value)

    with monkeypatch.context() as patch:
        patch.setattr(runtime, "import_ndarray", count_import)
        binding = alias.bind({"left": state, "right": state})
        assert len(imported) == 1
        alias.run(binding)
    assert _read(runtime, state) == [28, 7, 7, 7, 7]
    # The same Graph symbol may fill multiple exported argument positions too.
    repeat = ti.graph.GraphBuilder().dispatch_loaded(alias_entry, _symbol(), _symbol()).compile()
    repeat.run({"state": state})
    assert _read(runtime, state) == [35, 14, 14, 14, 14]
    tensor_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "v", ti.types.vector(2, ti.i32), ndim=1)
    tensor = ti.graph.GraphBuilder().dispatch_loaded(module.kernel("tensor"), tensor_arg).compile()
    tensor.submit({"v": vector}).wait()
    assert _read(runtime, vector) == [1, 3] * 5
    if ti.lang.impl.current_cfg().arch != ti.vulkan:
        wide = ti.ndarray(ti.i64, 1)
        scalar64 = (
            ti.graph.GraphBuilder()
            .dispatch_loaded(
                module.kernel("scalar64"), _symbol(dtype=ti.i64), ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i64)
            )
            .compile()
        )
        scalar64.submit({"state": wide, "value": 2**40 + 9}).wait()
        assert _read(runtime, wide, "q") == [2**40 + 9]
    entry = module.kernel("advance")
    ti.reset()
    with pytest.raises(ti.TaichiRuntimeError):
        alias.submit(binding)
    ti.init(arch=ti.cpu, offline_cache=False)
    with pytest.raises(ti.TaichiRuntimeError, match="current Program"):
        ti.graph.GraphBuilder().dispatch_loaded(entry, _symbol(), ti.graph.Arg(ti.graph.ArgKind.SCALAR, "d", ti.i32))
