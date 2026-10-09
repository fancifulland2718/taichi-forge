import re

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
        # now supplies the already selected kernel tier and normalized cap.
        config.compile_tier = kernel_tier
        config.full_simplify_global_iter_cap = 0 if kernel_tier == "full" else 1
        second = ti.aot.Module()
        add(second)
        second.save(str(tmp_path / "second"))
        assert runtime.debug_kernel_executable_lifecycle_stats()["compiler_invocations"] == 1
        if config.arch in (ti.cpu, ti.cuda):
            (module_path,) = tuple((tmp_path / "first").glob("calculate*.ll"))
            text = module_path.read_text(encoding="utf-8")
            metadata_id = re.search(r"!taichi\.jit\.options\.v1 = !\{!(\d+)\}", text).group(1)
            level, fast_math = re.search(
                rf"!{metadata_id} = !\{{i32 (\d+), i32 (\d+)\}}", text
            ).groups()
            expected = 3 if kernel_tier == "full" else (1 if config.arch == ti.cuda else 0)
            assert (int(level), int(fast_math)) == (expected, 0)
    finally:
        config.compile_tier = original_tier
        config.full_simplify_global_iter_cap = original_cap
        runtime.set_kernel_executable_lifecycle_telemetry_enabled(False)
    assert calculate(7) == 22
