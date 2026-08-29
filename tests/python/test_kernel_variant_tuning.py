import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._kernel_variant_tuning import _KernelVariantSession
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_kernel_variant_session_bounds_and_deduplicates_launch_only_axes():
    count = 4096
    values = ti.ndarray(ti.i32, shape=count)
    result = ti.field(ti.i32, shape=())

    @ti.kernel
    def reduce(inp: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            result[None] += inp[i]

    session = _KernelVariantSession(reduce, (values,))
    assert not session.rejections
    assert len(session.variant_ids()) == 384
    assert len(session.compilation_variant_ids()) == 24
    assert all(len(group.variant_ids) == 16 for group in session.compilation_groups)
    assert (
        len(
            {
                session.variant(variant_id).logical_task_id
                for variant_id in session.variant_ids()
            }
        )
        == 1
    )

    group = session.compilation_groups[0]
    bindings = tuple(session.bind(variant_id) for variant_id in group.variant_ids)
    reports = tuple(binding.report(values) for binding in bindings)
    compilation_key = session.variant(
        group.representative_variant_id
    ).spec.compilation_specialization_key
    compiled = tuple(
        kernel_cpp
        for key, kernel_cpp in bindings[0]._kernel.compiled_kernels.items()
        if len(key) == 5 and key[3] == compilation_key
    )
    assert len(compiled) == 1
    assert (
        len({tuple(task.task_id for task in report.tasks) for report in reports}) == 1
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_kernel_variant_session_drops_ineffective_tls_axis():
    count = 4096
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def transform(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i * 3 + 1

    session = _KernelVariantSession(transform, (values,))
    assert not session.rejections
    assert len(session.variant_ids()) == 320
    assert len(session.compilation_variant_ids()) == 20
    assert all(
        session.variant(variant_id).spec.ir.thread_local == "auto"
        for variant_id in session.variant_ids()
    )

    refinement = session.refinement(128)
    assert refinement.structural_mode == "cartesian"
    assert len(refinement.compilation_groups) == 9
    assert len(refinement.variant_ids()) == 144
    combined = tuple(
        variant_id
        for variant_id in refinement.variant_ids()
        if refinement.variant(variant_id).spec.backend.cuda_min_blocks_per_sm == 1
        and refinement.variant(variant_id).spec.artifact.cuda_max_registers == 24
        and refinement.variant(variant_id).spec.launch.range_work_per_thread_target == 4
        and refinement.variant(variant_id).spec.launch.grid_residency_waves == 2
    )
    assert len(combined) == 1


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_kernel_variant_session_reuses_compilation_and_native_memory():
    count = 1 << 14
    values = ti.ndarray(ti.i32, shape=count)
    result = ti.field(ti.i32, shape=())

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i % 17

    @ti.kernel
    def reduce(inp: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            result[None] += inp[i]

    fill(values)
    expected = sum(i % 17 for i in range(count))
    session = _KernelVariantSession(reduce, (values,))
    bindings = tuple(session.bind(variant_id) for variant_id in session.variant_ids())
    reports = tuple(binding.report(values) for binding in bindings)
    compilation_keys = {
        session.variant(variant_id).spec.compilation_specialization_key
        for variant_id in session.variant_ids()
    }
    compiled_variants = {
        key[3]: kernel_cpp
        for key, kernel_cpp in bindings[0]._kernel.compiled_kernels.items()
        if len(key) == 5 and key[3] in compilation_keys
    }
    assert set(compiled_variants) == compilation_keys
    assert len({id(kernel_cpp) for kernel_cpp in compiled_variants.values()}) == 24
    assert all(report.status == "applied" for report in reports)

    for binding in bindings:
        result[None] = 0
        binding(values)
        ti.sync()
        assert result[None] == expected

    program = impl.get_runtime().prog
    compiled_count = len(bindings[0]._kernel.compiled_kernels)
    runtime_memory = program._runtime_statistics_snapshot()["memory"]
    host_memory = dict(ti_core.get_host_memory_pool_stats())
    device_memory = dict(ti_core.get_device_memory_pool_stats())
    for index in range(100):
        replay = _KernelVariantSession(reduce, (values,))
        variant_id = replay.variant_ids()[index % len(replay.variant_ids())]
        replay.bind(variant_id).report(values)

    assert len(bindings[0]._kernel.compiled_kernels) == compiled_count
    assert program._runtime_statistics_snapshot()["memory"] == runtime_memory
    assert dict(ti_core.get_host_memory_pool_stats()) == host_memory
    assert dict(ti_core.get_device_memory_pool_stats()) == device_memory
