import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("optimized_first", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_config_variants_can_eliminate_field_dependencies(optimized_first):
    source = ti.field(ti.i32)
    source_builder = ti.FieldsBuilder()
    source_builder.place(source)
    source_tree = source_builder.finalize()
    output = ti.field(ti.i32)
    output_builder = ti.FieldsBuilder()
    output_builder.place(output)
    output_tree = output_builder.finalize()

    @ti.kernel
    def calculate():
        value = source[None]
        value = 7
        output[None] = value

    config = ti.lang.impl.current_cfg()
    original = config.cfg_optimization
    graphs = []
    try:
        for enabled in (optimized_first, not optimized_first, optimized_first):
            config.cfg_optimization = enabled
            calculate()
            assert output[None] == 7
            builder = ti.graph.GraphBuilder()
            builder.dispatch(calculate)
            graph = builder.compile()
            expected = {(output_tree.id, output_tree.generation)}
            if not enabled:
                expected.add((source_tree.id, source_tree.generation))
            assert graph._spec.snode_tree_dependencies == expected
            graph.run({})
            graphs.append((enabled, graph))
        # Previously compiled graphs keep their own executable after the
        # kernel has switched request contexts twice.
        for _, graph in graphs:
            output[None] = 0
            graph.run({})
            assert output[None] == 7
        key = calculate._primal.ensure_compiled()
        kernel = calculate._primal.compiled_kernels[key]
        source_tree.destroy()
        assert kernel.definition_retired()
        for enabled, graph in graphs:
            if enabled:
                output[None] = 0
                graph.run({})
                assert output[None] == 7
            else:
                with pytest.raises(ti.TaichiRuntimeError, match="destroyed SNodeTree"):
                    graph.run({})
        output_tree.destroy()
        for _, graph in graphs:
            with pytest.raises(ti.TaichiRuntimeError, match="destroyed SNodeTree"):
                graph.run({})
    finally:
        config.cfg_optimization = original


@pytest.mark.parametrize("optimized_first", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_config_variants_can_remove_all_field_dependencies(optimized_first):
    source = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    builder.place(source)
    tree = builder.finalize()

    @ti.kernel
    def calculate() -> ti.i32:
        value = source[None]
        value = 7
        return value

    config = ti.lang.impl.current_cfg()
    original = config.cfg_optimization
    program = ti.lang.impl.get_runtime().prog
    key = calculate._primal.ensure_compiled()
    kernel = calculate._primal.compiled_kernels[key]
    try:
        for enabled in (optimized_first, not optimized_first, optimized_first):
            config.cfg_optimization = enabled
            program.compile_kernel(config, program.get_device_caps(), kernel)
            assert calculate() == 7
        tree.destroy()
        assert kernel.definition_retired()
    finally:
        config.cfg_optimization = original


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_field_free_variant_does_not_elide_changed_request_guard(monkeypatch):
    arch = ti.lang.impl.current_cfg().arch
    ti.reset()
    monkeypatch.setenv("TI_DEBUG_ORDINARY_LAUNCH_ATTRIBUTION", "1")
    ti.init(arch=arch, offline_cache=False, compile_tier="fast")
    source = ti.field(ti.i32, shape=())

    @ti.kernel
    def calculate() -> ti.i32:
        value = source[None]
        value = 7
        return value

    program = ti.lang.impl.get_runtime().prog
    config = ti.lang.impl.current_cfg()
    key = calculate._primal.ensure_compiled()
    kernel = calculate._primal.compiled_kernels[key]
    caps = program.get_device_caps()
    config.cfg_optimization = True
    program.compile_and_launch_kernel(config, caps, kernel, kernel.make_launch_context())
    program._debug_reset_ordinary_launch_attribution()
    program.compile_and_launch_kernel(config, caps, kernel, kernel.make_launch_context())
    stats = dict(program._debug_ordinary_launch_attribution())
    assert stats["snode_guard_acquisitions"] == 0
    assert stats["snode_guard_elisions"] >= 1
    config.cfg_optimization = False
    program._debug_reset_ordinary_launch_attribution()
    program.compile_and_launch_kernel(config, caps, kernel, kernel.make_launch_context())
    stats = dict(program._debug_ordinary_launch_attribution())
    assert stats["snode_guard_acquisitions"] >= 1
    assert stats["snode_guard_elisions"] == 0
