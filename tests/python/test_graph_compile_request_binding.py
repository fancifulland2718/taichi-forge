import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("retire_definition", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_cold_graph_keeps_build_request_and_callable(cached, retire_definition):
    source = ti.field(ti.i32)
    source_builder = ti.FieldsBuilder()
    source_builder.place(source)
    source_tree = source_builder.finalize()
    output = ti.field(ti.i32)
    output_builder = ti.FieldsBuilder()
    output_builder.place(output)
    output_tree = output_builder.finalize()

    @ti.kernel
    def calculate(value: ti.i32):
        local = source[None]
        local = value
        output[None] = local

    runtime = ti.lang.impl.get_runtime()
    program = runtime.prog
    config = ti.lang.impl.current_cfg()
    original = config.cfg_optimization
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    try:
        config.cfg_optimization = True
        builder = ti.graph.GraphBuilder()
        builder.dispatch(calculate, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32))
        graph = builder.compile()
        assert graph._spec.snode_tree_dependencies == {(output_tree.id, output_tree.generation)}
        # This private accessor lazily builds the AOT-compatible native Graph;
        # materialize it under the intended build configuration as well.
        native_graph = None if cached else graph._compiled_graph
        config.cfg_optimization = False
        if retire_definition:
            key = calculate._primal.ensure_compiled(1)
            kernel = calculate._primal.compiled_kernels[key]
            program.compile_kernel(config, program.get_device_caps(), kernel)
            source_tree.destroy()
            assert kernel.definition_retired()
        # Readback's own compilation must not enter the measured interval.
        assert output[None] == 0
        runtime.debug_kernel_executable_lifecycle_stats(True)
        for value in (7, 19):
            if cached:
                graph.run({"value": value})
            else:
                native_graph.jit_run(config, {"value": value})
            assert output[None] == value
        assert runtime.debug_kernel_executable_lifecycle_stats()["compiler_invocations"] == 0
        output_tree.destroy()
        with pytest.raises(RuntimeError, match="destroyed SNodeTree"):
            if cached:
                graph.run({"value": 3})
            else:
                native_graph.jit_run(config, {"value": 3})
    finally:
        config.cfg_optimization = original
        runtime.set_kernel_executable_lifecycle_telemetry_enabled(False)
