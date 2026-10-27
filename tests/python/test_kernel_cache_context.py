import tempfile

import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_native_compile_requests_do_not_reuse_another_config():
    @ti.kernel
    def calculate(x: ti.i32) -> ti.i32:
        return x * 7 + 2

    runtime = ti.lang.impl.get_runtime()
    program = runtime.prog
    specialization = calculate._primal.ensure_compiled(3)
    kernel = calculate._primal.compiled_kernels[specialization]
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)
    config = ti.lang.impl.current_cfg()
    original = config.fast_math
    try:
        program.compile_kernel(config, program.get_device_caps(), kernel)
        first = program._kernel_cache_key_no_compile(kernel)
        config.fast_math = not original
        program.compile_kernel(config, program.get_device_caps(), kernel)
        second = program._kernel_cache_key_no_compile(kernel)
        assert first != second
        config.fast_math = original
        program.compile_kernel(config, program.get_device_caps(), kernel)
        assert program._kernel_cache_key_no_compile(kernel) == first
        stats = runtime.debug_kernel_executable_lifecycle_stats()
        assert stats["compiler_invocations"] == 2
        assert stats["memory_cache_hits"] >= 1
    finally:
        config.fast_math = original
        runtime.set_kernel_executable_lifecycle_telemetry_enabled(False)
    assert calculate(3) == 23


@pytest.mark.parametrize("cap_source", ["frontend_default", "direct_assignment"])
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_precompile_key_query_resolves_kernel_override_and_full_cap(cap_source):
    @ti.kernel(opt_level="full")
    def calculate(x: ti.i32) -> ti.i32:
        return x + 1

    program = ti.lang.impl.get_runtime().prog
    config = ti.lang.impl.current_cfg()
    original_tier = config.compile_tier
    original_cap = config.full_simplify_global_iter_cap
    try:
        config.compile_tier = "fast"
        config.full_simplify_global_iter_cap = 1
        if cap_source == "frontend_default":
            core._set_compile_option_source(config, "full_simplify_global_iter_cap", cap_source)
        specialization = calculate._primal.ensure_compiled(3)
        kernel = calculate._primal.compiled_kernels[specialization]
        queried = program._kernel_cache_key_no_compile(kernel)
        # The kernel overrides the base tier; changing only that tier leaves
        # the effective request unchanged, including an explicit cap of one.
        config.compile_tier = "full"
        assert program._kernel_cache_key_no_compile(kernel) == queried
        assert program._kernel_gpu_semantics_snapshot(kernel)["kernel_identity"] == queried
        # Only a default cap expands to a full fixed point. A direct assignment
        # is an explicit budget and must remain distinct from an unlimited cap.
        config.full_simplify_global_iter_cap = 0
        unlimited = program._kernel_cache_key_no_compile(kernel)
        assert (unlimited == queried) == (cap_source == "frontend_default")
        assert program._kernel_gpu_semantics_snapshot(kernel)["kernel_identity"] == unlimited
        config.full_simplify_global_iter_cap = 3
        changed = program._kernel_cache_key_no_compile(kernel)
        assert changed != queried
        assert changed != unlimited
        assert program._kernel_gpu_semantics_snapshot(kernel)["kernel_identity"] == changed
    finally:
        config.compile_tier = original_tier
        config.full_simplify_global_iter_cap = original_cap
    assert calculate(3) == 4


@pytest.mark.parametrize("aot_first", [False, True])
@test_utils.test(
    arch=[ti.cuda, ti.vulkan], offline_cache=False, compile_tier="fast", advanced_optimization=False
)
def test_jit_and_aot_requests_do_not_share_different_capabilities(aot_first):
    value = ti.field(ti.i32, shape=())

    @ti.kernel
    def write_value():
        value[None] = 17

    runtime = ti.lang.impl.get_runtime()
    program = runtime.prog
    key = write_value._primal.ensure_compiled()
    kernel = write_value._primal.compiled_kernels[key]
    config = ti.lang.impl.current_cfg()
    # Vulkan's explicit empty list means portable baseline capabilities;
    # CUDA uses an explicit supported target distinct from the test device.
    caps = []
    if config.arch == ti.cuda:
        target = 70 if core.query_int64("cuda_compute_capability") == 60 else 60
        caps = [ti.DeviceCapability.cuda_compute_capability(target)]
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)
    try:
        module = ti.aot.Module(caps=caps)
        if not aot_first:
            program.compile_kernel(config, program.get_device_caps(), kernel)
        module.add_kernel(write_value)
        if aot_first:
            program.compile_kernel(config, program.get_device_caps(), kernel)
        assert runtime.debug_kernel_executable_lifecycle_stats()["compiler_invocations"] == 2
        # Returning to either context reuses its own already compiled artifact.
        module.add_kernel(write_value, name="again")
        program.compile_kernel(config, program.get_device_caps(), kernel)
        assert runtime.debug_kernel_executable_lifecycle_stats()["compiler_invocations"] == 2
        with tempfile.TemporaryDirectory() as directory:
            module.save(directory)
    finally:
        runtime.set_kernel_executable_lifecycle_telemetry_enabled(False)
    write_value()
    assert value[None] == 17
