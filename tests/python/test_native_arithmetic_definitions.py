"""Native arithmetic creation, replay and export consume the same semantics."""

import json

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_arithmetic_definition_reuse_without_warm_conversion(monkeypatch):
    from taichi_forge.algorithms import _algorithms as algorithms
    from taichi_forge.algorithms._primitive_definitions import ReduceDefinition, TransformDefinition
    from taichi_forge.graph._native_definition import NativeDefinition, NativeOperationBinding

    src, dst, output = (ti.ndarray(ti.f32, n) for n in (17, 17, 3))
    host = (np.arange(17, dtype=np.float32) - 8) / 4
    src.from_numpy(host)
    output.fill(37)
    transform = ti.algorithms.TransformWorkspace()
    reduce = ti.algorithms.ReduceWorkspace()
    ti.algorithms.experimental_transform(src, dst, scale=1.25, bias=-0.5, workspace=transform)
    ti.algorithms.experimental_reduce(dst, output, op="sum", workspace=reduce)
    definitions = (transform._native_transform_plan.definition, reduce._native_reduce_plan.definition)
    sequence = (
        ti.algorithms.primitive_sequence()
        .transform(src, dst, scale=1.25, bias=-0.5, workspace=transform)
        .reduce(dst, output, op="sum", workspace=reduce)
    )
    sequence.run()
    assert all(call.plan.definition is definition for call, definition in zip(sequence._calls, definitions))

    def no_conversion(*args, **kwargs):
        raise AssertionError("warm arithmetic must not create or lower definitions")

    with monkeypatch.context() as patch:
        patch.setattr(algorithms, "TransformDefinition", no_conversion)
        patch.setattr(algorithms, "ReduceDefinition", no_conversion)
        patch.setattr(NativeDefinition, "parameter_dict", no_conversion)
        patch.setattr(NativeOperationBinding, "lower", no_conversion)
        ti.algorithms.experimental_transform(src, dst, scale=1.25, bias=-0.5, workspace=transform)
        ti.algorithms.experimental_reduce(dst, output, op="sum", workspace=reduce)
        sequence.run()
    np.testing.assert_array_equal(dst.to_numpy(), host * 1.25 - 0.5)
    np.testing.assert_array_equal(output.to_numpy(), [np.sum(host * 1.25 - 0.5), 37, 37])
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    module = ti.aot.Module()
    selected = []

    def record_bind(method):
        def bind(definition, *args, **kwargs):
            selected.append(definition)
            return method(definition, *args, **kwargs)

        return bind

    with monkeypatch.context() as patch:
        patch.setattr(TransformDefinition, "bind", record_bind(TransformDefinition.bind))
        patch.setattr(ReduceDefinition, "bind", record_bind(ReduceDefinition.bind))
        plan = module.add_execution_plan("arithmetic", graph, source_bindings=dict(src=src, dst=dst, output=output))
    assert len(selected) == 2 and all(a is b for a, b in zip(selected, definitions))
    operations = json.loads(plan.to_json())["operations"]
    assert [op["operation"] for op in operations] == ["transform", "reduce"]
    assert operations[0]["parameters"]["scale_bits"] == "3ff4000000000000"
    assert operations[0]["parameters"]["bias_bits"] == "bfe0000000000000"
    assert operations[1]["parameters"]["op"] == "sum"
    graph.close()
