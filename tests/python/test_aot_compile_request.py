import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("kernel_tier", ["fast", "full"])
@pytest.mark.parametrize("use_template", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, fast_math=False)
def test_aot_resolves_kernel_tier_before_codegen_and_cache_lookup(tmp_path, kernel_tier, use_template):
    value_type = ti.template() if use_template else ti.i32

    @ti.kernel(opt_level=kernel_tier)
    def calculate(value: value_type) -> ti.i32:
        return value * 3 + 1

    def add(module):
        if use_template:
            with module.add_kernel_template(calculate) as template:
                template.instantiate(value=7)
        else:
            module.add_kernel(calculate, name="calculate")

    runtime = ti.lang.impl.get_runtime()
    config = ti.lang.impl.current_cfg()
    original_tier = config.compile_tier
    original_cap = config.full_simplify_global_iter_cap
    config.compile_tier = "full" if kernel_tier == "fast" else "fast"
    config.full_simplify_global_iter_cap = 1
    # Materialize source/layout once so both modules share the same ABI.
    calculate._primal.ensure_compiled(7)
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    try:
        first = ti.aot.Module()
        add(first)
        first.save(str(tmp_path / "first"))
        # Equivalent effective request with a different origin: the Program
        # now supplies the already selected kernel tier. The explicitly set
        # simplification budget remains 1, including for the full tier.
        config.compile_tier = kernel_tier
        second = ti.aot.Module()
        add(second)
        second.save(str(tmp_path / "second"))
        assert runtime.debug_kernel_executable_lifecycle_stats()["compiler_invocations"] == 1
        if config.arch in (ti.cpu, ti.cuda):
            # Deployment emits BC, not redundant diagnostic .ll files.
            # LLVMModuleOptions' native round-trip test inspects the frozen
            # optimization values in the serialized compiled payload.
            assert list((tmp_path / "first").glob("calculate*.bc"))
    finally:
        config.compile_tier = original_tier
        config.full_simplify_global_iter_cap = original_cap
        runtime.set_kernel_executable_lifecycle_telemetry_enabled(False)
    assert calculate(7) == 22


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_sparse_helpers_use_declared_cuda_target(tmp_path):
    values = ti.field(ti.i32)
    ti.root.pointer(ti.i, 2).dense(ti.i, 4).place(values)

    @ti.kernel
    def activate():
        for i in range(8):
            values[i] = i + 1

    # Sparse locking uses acquire atomics; LLVM NVPTX requires sm_70 for those
    # orderings. Pre-Volta helper specialization is covered in the native test.
    module = ti.aot.Module(artifact_format="ptx", caps=[ti.DeviceCapability.cuda_compute_capability(70)])
    module.add_field("values", values)
    module.add_kernel(activate)
    module.save(str(tmp_path / "first"))
    # A sparse kernel, its helper requirements, and repeated exports all keep
    # the declared deployment target even on a newer producer device.
    (ptx,) = tuple((tmp_path / "first").glob("activate*.ptx"))
    text = ptx.read_text(encoding="utf8")
    assert ".target sm_70" in text
    module.save(str(tmp_path / "second"))
    assert ".target sm_70" in next((tmp_path / "second").glob("activate*.ptx")).read_text(encoding="utf8")
    activate()
    assert values[7] == 8
