"""One frozen multi-column definition across direct, recorded and AOT use."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prepared_compact_definition_and_offset_reconstruction(tmp_path, monkeypatch):
    from taichi_forge.graph._native_definition import NativeDefinition, NativeOperationBinding

    n = 17
    inputs = [ti.ndarray(dtype, n + 2) for dtype in (ti.i32, ti.u64)]
    outputs = [ti.ndarray(dtype, n + 3) for dtype in (ti.i32, ti.u64)]
    flags, count = ti.ndarray(ti.i32, n + 2), ti.ndarray(ti.i32, 3)
    view = lambda value: ti.experimental.ndarray_view(value, slices=slice(1, n + 1))
    plan = ti.algorithms.prepare_compact_columns(
        tuple((view(source), view(output)) for source, output in zip(inputs, outputs)),
        view(flags),
        ti.experimental.ndarray_view(count, slices=slice(1, 2)),
    )
    plan._description["columns"] = ()  # Reporting is not the reconstruction authority.
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record())
    graph = builder.compile()
    bindings = dict(flags=flags, count=count)
    bindings.update({f"values_{i}": value for i, value in enumerate(inputs)})
    bindings.update({f"output_{i}": value for i, value in enumerate(outputs)})
    module = ti.aot.Module()
    exported = module.add_execution_plan("select", graph, source_bindings=bindings)
    operation = json.loads(exported.to_json())["operations"][0]
    assert operation["operation"] == "compact"
    assert operation["parameters"] == {"column_count": 2}
    assert set(operation["bindings"]) == set(bindings)

    def no_cold_conversion(*args, **kwargs):
        raise AssertionError("Prepared execution must not rebuild its definition")

    for value, dtype in zip(inputs, (np.int32, np.uint64)):
        value.from_numpy(np.arange(n + 2, dtype=dtype))
    flags.fill(1)
    with monkeypatch.context() as patch:
        patch.setattr(NativeDefinition, "parameter_dict", no_cold_conversion)
        patch.setattr(NativeOperationBinding, "lower", no_cold_conversion)
        plan.run()
        graph.run({})
    assert count.to_numpy()[1] == n
    artifact = tmp_path / "prepared-compact.tcm"
    module.archive(str(artifact))
    graph.close()
    plan.close()
    assert plan._native_binding is None
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_prepared_compact_probe.py")),
                mode,
                "cuda" if ti.cfg.arch == ti.cuda else "vulkan",
                str(artifact),
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert not any(x in result.stdout + result.stderr for x in ("VUID-", "Validation Error", "SYNC-HAZARD"))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_prepared_sort_bitwise_stays_a_jit_only_policy():
    keys = ti.ndarray(ti.f32, 7)
    plan = ti.algorithms.prepare_sort(keys, nan_policy="bitwise")
    plan._description["nan_policy"] = "last"
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record())
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="NaNs-last"):
        module.add_execution_plan("sort", graph, source_bindings=dict(keys=keys))
    assert not module._content and not module._execution_plans
    graph.close()
    plan.close()
