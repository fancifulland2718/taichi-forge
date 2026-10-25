"""Native workload observations share frozen definitions, never execution owners."""

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph._native_definition import NativeDefinition, NativeOperationBinding, NativeWorkload
from tests import test_utils


def source_contracts(graph):
    return [json.loads(action.source_contract_json)
            for stage in graph.definition._runtime_spec.pipeline_definition
            for action in stage.get("native_actions", ()) if action.source_contract_json]


def test_native_workload_is_frozen_and_does_not_authorize_scheduling():
    dimensions = [2, 1 << 40]
    work = NativeWorkload(dimensions, "logical_item", "provider", "rejected")
    dimensions[0] = 5
    assert work.to_dict()["logical_items"] == 1 << 41
    assert work.to_dict()["transformation_permissions"] == []
    with pytest.raises(FrozenInstanceError):
        work.dimensions = (1,)
    definition = NativeDefinition("vendor", "operation", dict(strategy="opaque"))
    binding = NativeOperationBinding(definition, (), workload=work)
    result = binding.source_contract()
    result["operation"]["parameters"]["strategy"] = "changed"
    result["workload"]["dimensions"][0] = 9
    assert binding.source_contract()["operation"]["parameters"]["strategy"] == "opaque"
    assert binding.source_contract()["workload"]["dimensions"] == [2, 1 << 40]
    assert NativeOperationBinding(definition, ()).source_contract() is None
    assert NativeWorkload((0,), "record", "forge", "no_op").to_dict()["logical_items"] == 0
    for shape in ((-1,), (True,), (0,)):
        with pytest.raises(ValueError):
            NativeWorkload(shape, "record", "forge", "rejected")


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_sort_workload_shared_by_direct_recorded_and_exported_paths(tmp_path, monkeypatch):
    keys, values = (ti.ndarray(ti.i32, 65) for _ in range(2))
    data = np.arange(65, dtype=np.int32) * 7 % 11
    payload = np.arange(65, dtype=np.int32)
    order = np.argsort(data, kind="stable")
    plan = ti.algorithms.prepare_sort(keys, values)
    work = plan.report()["workload"]
    assert work["dimensions"] == [65]
    assert work["scheduling_owner"] == "forge"
    assert plan.report()["workspace_bytes_last_observed"] is None
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record())
    graph = builder.compile()
    source, = source_contracts(graph)
    assert source["workload"] == work
    assert source["operation"] == plan._definition.to_dict()

    def forbidden(*args, **kwargs):
        raise AssertionError("Warm execution rebuilt the native workload description")

    with monkeypatch.context() as patch:
        patch.setattr(NativeOperationBinding, "source_contract", forbidden)
        patch.setattr(NativeWorkload, "to_dict", forbidden)
        for invoke in (plan.run, lambda: graph.run({}), lambda: graph.run({})):
            keys.from_numpy(data)
            values.from_numpy(payload)
            invoke()
            np.testing.assert_array_equal(keys.to_numpy(), data[order])
            np.testing.assert_array_equal(values.to_numpy(), payload[order])

    module = ti.aot.Module()
    exported = module.add_execution_plan("sort", graph, source_bindings=dict(keys=keys, values=values))
    operation = json.loads(exported.to_json())["operations"][0]
    assert operation["parameters"] == source["operation"]["parameters"]
    artifact = tmp_path / "sort.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        buffers = {name: runtime.ndarray("i32", 65) for name in ("keys", "values")}
        try:
            buffers["keys"].write(data.tobytes())
            buffers["values"].write(payload.tobytes())
            with loaded.prepare_execution("sort", buffers) as call:
                call()
            np.testing.assert_array_equal(np.frombuffer(buffers["keys"].read(), np.int32), data[order])
            np.testing.assert_array_equal(np.frombuffer(buffers["values"].read(), np.int32), payload[order])
        finally:
            for buffer in buffers.values():
                buffer.close()
    if ti.lang.impl.current_cfg().arch != ti.cpu:
        recording = plan.record(persistent=True)
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording)
        persistent = builder.compile()
        persistent_source, = source_contracts(persistent)
        assert persistent_source == source
        plan.close()
        keys.from_numpy(data)
        values.from_numpy(payload)
        persistent.run({})
        np.testing.assert_array_equal(values.to_numpy(), payload[order])
        persistent.close()
    graph.close()
    plan.close()
