import gc
import threading

import pytest

import taichi_forge as ti
from tests import test_utils
from taichi_forge.lang import impl


@test_utils.test()
def test_argpack_basic():
    pack_type = ti.types.argpack(a=ti.i32, b=bool, c=ti.f32)
    pack1 = pack_type(a=1, b=False, c=2.1)
    pack2 = pack_type(a=2, b=True, c=2.1)

    @ti.kernel
    def foo(pack: pack_type) -> ti.f32:
        tmp = 0.0
        if pack.b:
            tmp = pack.a + pack.c
        else:
            tmp = pack.a * pack.c
        return tmp

    assert foo(pack1) == test_utils.approx(1 * 2.1, rel=1e-3)
    assert foo(pack2) == test_utils.approx(2 + 2.1, rel=1e-3)


@test_utils.test()
def test_argpack_with_struct():
    struct_type = ti.types.struct(a=ti.i32, c=ti.f32)
    pack_type = ti.types.argpack(d=ti.f32, element=struct_type)

    @ti.kernel
    def foo(pack: pack_type) -> ti.f32:
        tmp = pack.element.a + pack.element.c
        return tmp + pack.d

    pack = pack_type(d=2.1, element=struct_type(a=2, c=2.1))
    assert foo(pack) == test_utils.approx(2 + 2.1 + 2.1, rel=1e-3)


@test_utils.test()
def test_argpack_with_vector():
    pack_type = ti.types.argpack(a=ti.i32, b=ti.types.vector(3, ti.f32), c=ti.f32)
    pack = pack_type(a=1, b=[1.0, 2.0, 3.0], c=2.1)

    @ti.kernel
    def foo(pack: pack_type) -> ti.f32:
        tmp = pack.a * pack.c
        return tmp + pack.b[1]

    assert foo(pack) == test_utils.approx(1 * 2.1 + 2.0, rel=1e-3)


@test_utils.test()
def test_argpack_multiple():
    arr = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4))
    arr.fill([1.0, 2.0, 3.0])

    pack_type1 = ti.types.argpack(a=ti.i32, c=ti.f32)
    pack_type2 = ti.types.argpack(a=ti.types.ndarray(dtype=ti.math.vec3, ndim=2))
    pack1 = pack_type1(a=1, c=2.1)
    pack2 = pack_type2(a=arr)

    @ti.kernel
    def foo(p1: pack_type1, p2: pack_type2) -> ti.f32:
        tmp = p1.a * p1.c
        return tmp + p2.a[1, 2][1]

    assert foo(pack1, pack2) == test_utils.approx(1 * 2.1 + 2.0, rel=1e-3)


@test_utils.test()
def test_argpack_nested():
    arr = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4))
    arr.fill([1.0, 2.0, 3.0])

    pack_type_inner = ti.types.argpack(a=ti.i32, b=ti.i32)
    pack_type = ti.types.argpack(a=ti.types.ndarray(dtype=ti.math.vec3, ndim=2), b=ti.i32, c=pack_type_inner)
    pack_inner = pack_type_inner(a=123, b=456)
    pack = pack_type(a=arr, b=233, c=pack_inner)

    @ti.kernel
    def p(x: pack_type) -> ti.math.vec3:
        return x.a[2, 3]

    @ti.kernel
    def q(x: pack_type) -> int:
        return x.c.a + x.c.b

    @ti.kernel
    def h(x: pack_type) -> int:
        return x.b

    assert p(pack) == [1.0, 2.0, 3.0]
    assert q(pack) == 123 + 456
    assert h(pack) == 233


@test_utils.test()
def test_argpack_as_return():
    pack_type = ti.types.argpack(a=ti.i32, b=bool)

    with pytest.raises(ti.TaichiSyntaxError):

        @ti.kernel
        def foo(pack: pack_type) -> pack_type:
            return pack

        foo()


@test_utils.test()
def test_argpack_as_struct_type_element():
    with pytest.raises(ValueError, match="Invalid data type <ti.ArgPackType a=i32, b=u1>"):
        pack_type = ti.types.argpack(a=ti.i32, b=bool)
        struct_with_argpack_inside = ti.types.struct(element=pack_type)
        print(struct_with_argpack_inside)


@test_utils.test()
def test_argpack_with_ndarray():
    arr = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4))
    arr.fill([1.0, 2.0, 3.0])

    pack_type = ti.types.argpack(element=ti.types.ndarray(dtype=ti.math.vec3, ndim=2))
    pack = pack_type(element=arr)

    @ti.kernel
    def foo(pack: pack_type) -> ti.math.vec3:
        return pack.element[0, 0]

    assert foo(pack) == [1.0, 2.0, 3.0]


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_argpack_registry_retire_waits_for_gpu_sync():
    pack_type = ti.types.argpack(value=ti.i32)
    sink = ti.field(ti.i32, shape=())

    @ti.kernel
    def consume(pack: pack_type):
        sink[None] += pack.value

    # Compile before taking the baseline.  Kernel compilation creates an
    # internal ArgPack proxy whose lifetime is tied to the compiled kernel;
    # this test measures only user ArgPacks submitted at runtime.
    consume(pack_type(value=0))
    ti.sync()
    gc.collect()
    prog = impl.get_runtime().prog
    baseline = prog._debug_argpack_resource_stats()
    pack = pack_type(value=7)
    created = prog._debug_argpack_resource_stats()
    # A public ArgPackType construction may create short-lived normalization
    # packs before returning the final pack.  Keep this test independent of
    # that Python implementation detail and assert the native lifecycle is
    # balanced instead of assuming exactly one native allocation.
    native_created = created["created_total"] - baseline["created_total"]
    assert native_created >= 1
    assert created["live"] == baseline["live"] + 1
    assert created["views"] == baseline["views"] + 1
    assert created["leases"] == baseline["leases"] + 1
    assert created["retired_total"] - baseline["retired_total"] == native_created - 1
    assert created["released_total"] - baseline["released_total"] == native_created - 1

    consume(pack)
    launched = prog._debug_argpack_resource_stats()
    if impl.current_cfg().arch == ti.cpu:
        assert launched["inflight"] == baseline["inflight"]
        assert launched["leases"] == baseline["leases"] + 1
    else:
        assert launched["inflight"] == baseline["inflight"] + 1
        assert launched["leases"] == baseline["leases"] + 2

    del pack
    gc.collect()
    retired = prog._debug_argpack_resource_stats()
    assert retired["views"] == baseline["views"]
    assert retired["retired_total"] == baseline["retired_total"] + native_created
    if impl.current_cfg().arch == ti.cpu:
        assert retired["released_total"] == baseline["released_total"] + native_created
    else:
        assert retired["retiring"] == baseline["retiring"] + 1
        assert retired["released_total"] == baseline["released_total"] + native_created - 1

    ti.sync()
    completed = prog._debug_argpack_resource_stats()
    assert completed["live"] == baseline["live"]
    assert completed["retiring"] == baseline["retiring"]
    assert completed["leases"] == baseline["leases"]
    assert completed["views"] == baseline["views"]
    assert completed["inflight"] == baseline["inflight"]
    assert completed["created_total"] == baseline["created_total"] + native_created
    assert completed["retired_total"] == baseline["retired_total"] + native_created
    assert completed["released_total"] == baseline["released_total"] + native_created
    assert sink[None] == 7


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_argpack_registry_concurrent_submit_and_gc():
    pack_type = ti.types.argpack(value=ti.i32)
    sink = ti.field(ti.i32, shape=())

    @ti.kernel
    def consume(pack: pack_type):
        sink[None] += pack.value

    consume(pack_type(value=0))
    ti.sync()
    gc.collect()
    prog = impl.get_runtime().prog
    baseline = prog._debug_argpack_resource_stats()
    errors = []
    thread_count = 4
    iterations = 8

    def worker(worker_id):
        try:
            for i in range(iterations):
                pack = pack_type(value=worker_id * iterations + i + 1)
                consume(pack)
                del pack
                if i % 2 == 0:
                    gc.collect()
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert not errors

    gc.collect()
    ti.sync()
    completed = prog._debug_argpack_resource_stats()
    expected = thread_count * iterations
    created_delta = completed["created_total"] - baseline["created_total"]
    retired_delta = completed["retired_total"] - baseline["retired_total"]
    released_delta = completed["released_total"] - baseline["released_total"]
    assert completed["live"] == baseline["live"]
    assert completed["retiring"] == baseline["retiring"]
    assert completed["leases"] == baseline["leases"]
    assert completed["views"] == baseline["views"]
    assert completed["inflight"] == baseline["inflight"]
    assert created_delta >= expected
    assert created_delta % expected == 0
    assert retired_delta == created_delta
    assert released_delta == created_delta
    assert sink[None] == expected * (expected + 1) // 2


@test_utils.test(arch=[ti.cuda, ti.vulkan])
def test_argpack_ndarray_inflight_keys_are_type_isolated():
    pack_type = ti.types.argpack(value=ti.i32)
    sink = ti.field(ti.i32, shape=())

    @ti.kernel
    def consume_pack(pack: pack_type):
        sink[None] += pack.value

    @ti.kernel
    def consume_array(arr: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        sink[None] += arr[0]

    # Compile before constructing the colliding user resources so compiler
    # proxy ArgPacks do not perturb the slot identities below.
    consume_pack(pack_type(value=0))
    warmup = ti.ndarray(ti.i32, shape=1)
    warmup.fill(0)
    consume_array(warmup)
    del warmup
    gc.collect()
    ti.sync()

    prog = impl.get_runtime().prog
    pack = pack_type(value=7)
    pack_native = pack._ArgPack__argpack
    pack_identity = prog._debug_argpack_resource_identity(pack_native)

    # Registry domains and kinds deliberately isolate identities. Force the
    # same slot index/generation in the two registries to catch accidental use
    # of a key from the other in-flight map.
    arrays = [
        ti.ndarray(ti.i32, shape=1) for _ in range(pack_identity["index"] + 1)
    ]
    target = arrays[-1]
    target_identity = prog._debug_ndarray_resource_identity(target.arr)
    while target_identity["generation"] < pack_identity["generation"]:
        arrays.pop()
        del target
        gc.collect()
        target = ti.ndarray(ti.i32, shape=1)
        arrays.append(target)
        target_identity = prog._debug_ndarray_resource_identity(target.arr)

    assert target_identity["index"] == pack_identity["index"]
    assert target_identity["generation"] == pack_identity["generation"]
    assert target_identity["domain"] != pack_identity["domain"]
    assert target_identity["kind"] != pack_identity["kind"]

    target.fill(5)
    consume_array(target)
    ndarray_launched = prog._debug_ndarray_resource_stats()
    assert ndarray_launched["inflight"] >= 1

    argpack_before = prog._debug_argpack_resource_stats()
    consume_pack(pack)
    argpack_launched = prog._debug_argpack_resource_stats()
    assert argpack_launched["inflight"] == argpack_before["inflight"] + 1
    assert argpack_launched["leases"] == argpack_before["leases"] + 1

    del pack
    del pack_native
    del target
    arrays.clear()
    gc.collect()
    ti.sync()
    assert sink[None] == 12


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_argpack_registry_more_than_inline_launch_leases():
    pack_type = ti.types.argpack(value=ti.i32)
    sink = ti.field(ti.i32, shape=())

    @ti.kernel
    def consume(
        p0: pack_type,
        p1: pack_type,
        p2: pack_type,
        p3: pack_type,
        p4: pack_type,
        p5: pack_type,
        p6: pack_type,
        p7: pack_type,
        p8: pack_type,
    ):
        sink[None] = (
            p0.value
            + p1.value
            + p2.value
            + p3.value
            + p4.value
            + p5.value
            + p6.value
            + p7.value
            + p8.value
        )

    packs = [pack_type(value=i) for i in range(1, 10)]
    consume(*packs)
    prog = impl.get_runtime().prog
    launched = prog._debug_argpack_resource_stats()
    if impl.current_cfg().arch == ti.cpu:
        assert launched["inflight"] == 0
    else:
        assert launched["inflight"] == len(packs)
    # First compilation creates one persistent _IntermediateArgPack per
    # argument. A live ABI view owns exactly one registry lease, so account
    # for these compiler proxies instead of treating them as user resources.
    expected_inflight_leases = (
        0 if impl.current_cfg().arch == ti.cpu else len(packs)
    )
    assert launched["leases"] == launched["views"] + expected_inflight_leases
    persistent_leases = launched["views"] - len(packs)
    assert persistent_leases >= 0

    del packs
    gc.collect()
    ti.sync()
    completed = prog._debug_argpack_resource_stats()
    assert completed["retiring"] == 0
    assert completed["leases"] == persistent_leases
    assert completed["views"] == persistent_leases
    assert completed["inflight"] == 0
    assert sink[None] == sum(range(1, 10))


@test_utils.test(arch=ti.cpu)
def test_argpack_launch_context_rejects_mismatched_resource_generation():
    pack_type = ti.types.argpack(value=ti.i32)

    @ti.kernel
    def read(pack: pack_type) -> ti.i32:
        return pack.value

    old = pack_type(value=3)
    primal = read._primal
    key = primal.ensure_compiled(old)
    kernel_cpp = primal.compiled_kernels[key]
    prog = impl.get_runtime().prog
    compiled = prog.compile_kernel(
        prog.config(), prog.get_device_caps(), kernel_cpp
    )
    old_native = old._ArgPack__argpack
    old_identity = prog._debug_argpack_resource_identity(old_native)
    old._invalidate_runtime()
    prog.delete_argpack(old_native)

    replacement = pack_type(value=17)
    replacement_native = replacement._ArgPack__argpack
    replacement_identity = prog._debug_argpack_resource_identity(
        replacement_native
    )
    assert replacement_identity != old_identity

    launch_ctx = kernel_cpp.make_launch_context()
    launch_ctx.set_arg_argpack([0], replacement_native)
    launch_ctx._debug_set_argpack_resource_handle(
        [0],
        replacement_identity["domain"],
        replacement_identity["kind"],
        replacement_identity["index"],
        replacement_identity["generation"] + 1,
    )
    with pytest.raises(RuntimeError, match="stale or retired ArgPack"):
        prog.launch_kernel(compiled, launch_ctx)
