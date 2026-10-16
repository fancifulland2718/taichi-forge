"""Compact native control budgets do not require expanded command capacity."""

import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils
from tests.python.aot_nested_capacity_probe import cases, expected


def _check_nested_capacity(budget, regions, lowering, route, tmp_path):
    from taichi_forge.graph._recipes import GraphFamilySelection

    arch = "vulkan" if ti.lang.impl.current_cfg().arch == ti.vulkan else "cuda"

    @ti.kernel
    def zero(value: ti.types.ndarray(ti.i32, 1)):
        value[0] = 0

    @ti.kernel
    def seed(dx: ti.types.ndarray(ti.f32, 1), dy: ti.types.ndarray(ti.f32, 1)):
        for i in dx:
            dx[i], dy[i] = 0, 1

    @ti.kernel
    def condition(
        count: ti.types.ndarray(ti.i32, 1),
        pred: ti.types.ndarray(ti.i32, 1),
        limits: ti.types.ndarray(ti.i32, 1),
        slot: ti.i32,
    ):
        pred[0] = int(count[0] < limits[slot])

    @ti.kernel
    def advance(count: ti.types.ndarray(ti.i32, 1)):
        count[0] += 1

    @ti.kernel
    def primal(
        x: ti.types.ndarray(ti.f32, 1),
        y: ti.types.ndarray(ti.f32, 1),
        total: ti.types.ndarray(ti.i32, 1),
    ):
        for i in x:
            y[i] = x[i] * x[i] * (total[0] + 1)

    @ti.kernel
    def forward(
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        total: ti.types.ndarray(ti.i32, 1),
    ):
        for i in x:
            y[i] = x[i] * x[i] * (total[0] + 1)

    @ti.kernel
    def publish(
        count: ti.types.ndarray(ti.i32, 1),
        total: ti.types.ndarray(ti.i32, 1),
        terminal: ti.types.ndarray(ti.i32, 1),
    ):
        terminal[0], terminal[1] = count[0], total[0]

    def build(reverse):
        builder = ti.graph.GraphBuilder()
        public = {n: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.f32, ndim=1) for n in ("x", "y", "dx", "dy")}
        public.update({n: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.i32, ndim=1) for n in ("limits", "terminal")})
        outer = builder.private_ndarray("outer", ti.i32, (1,))
        op = builder.private_ndarray("op", ti.i32, (1,))
        total = builder.private_ndarray("total", ti.i32, (1,))
        builder.dispatch(zero, outer)
        builder.dispatch(zero, total)
        builder.dispatch(seed, public["dx"], public["dy"])
        oc, ob = builder.create_sequential(), builder.create_sequential()
        oc.dispatch(
            condition,
            outer,
            op,
            public["limits"],
            oc._bind_internal_scalar("outer_slot", ti.i32, 0),
        )
        for index in range(regions):
            count = builder.private_ndarray(f"inner{index}", ti.i32, (1,))
            pred = builder.private_ndarray(f"predicate{index}", ti.i32, (1,))
            ic, ib = builder.create_sequential(), builder.create_sequential()
            ob.dispatch(zero, count)
            ic.dispatch(
                condition,
                count,
                pred,
                public["limits"],
                ic._bind_internal_scalar(f"slot{index}", ti.i32, 1),
            )
            ib.dispatch(advance, count)
            ib.dispatch(advance, total)
            ob.while_loop(
                ic,
                ib,
                predicate=pred,
                counter=count,
                max_iterations=budget,
                name=f"inner{index}",
            )
        ob.dispatch(advance, outer)
        builder.while_loop(
            oc,
            ob,
            predicate=op,
            counter=outer,
            max_iterations=budget,
            lowering_mode=lowering,
            name="outer",
        )
        builder.dispatch(forward if reverse else primal, public["x"], public["y"], total)
        if reverse:
            builder.dispatch(forward.grad, public["x"], public["y"], total)
        builder.dispatch(publish, outer, total, public["terminal"])
        definition = builder.freeze()
        if route == "device_update":
            from taichi_forge.aot._graph_export import _cuda_nested_while

            source = definition._runtime_spec.structured_control_nodes[0]
            assert source._cuda_nested_submission_options["execution_route"] == "auto"
            # The source still allows fallback. Only materialization freezes
            # its route, even when the user retains the baseline recipe.
            with pytest.raises(ti.TaichiRuntimeError, match="frozen device-update recipe"):
                _cuda_nested_while(None, source)
        recipe = None
        if lowering == "auto" and route == "conditional":
            catalog = definition.recipe_catalog()
            fragment = next(
                f
                for f in catalog.fragments
                if f.provider_namespace == "taichi_forge.graph.structured_control"
                and GraphFamilySelection.from_fragment(f).choice_id == "control:cuda_nested_conditional:v1"
            )
            recipe = catalog.compose((fragment.fragment_id,), stage="single-region").recipe
        owner = definition.materialize(recipe)
        if route == "device_update":
            assert owner.executor._spec.structured_control_nodes[0]._cuda_nested_submission_options == dict(
                execution_route="device_update"
            )
        return owner

    timings = {}
    module = ti.aot.Module()
    begin = time.perf_counter()
    with build(False) as owner:
        timings["build_ms"] = (time.perf_counter() - begin) * 1000
        arrays = {n: ti.ndarray(ti.f32, 4) for n in ("x", "y", "dx", "dy")}
        arrays.update({n: ti.ndarray(ti.i32, 2) for n in ("limits", "terminal")})
        arrays["x"].from_numpy(np.array([0.5, 1, 2, 3], dtype=np.float32))
        frames = []
        for limits in cases(budget):
            arrays["limits"].from_numpy(np.array(limits, dtype=np.int32))
            begin = time.perf_counter()
            owner.executor.submit(arrays).wait()
            frames.append((time.perf_counter() - begin) * 1000)
            y, _, terminal = expected([0.5, 1, 2, 3], limits, budget, regions)
            np.testing.assert_array_equal(arrays["y"].to_numpy(), y)
            np.testing.assert_array_equal(arrays["terminal"].to_numpy(), terminal)
        timings["jit_complete_ms"] = frames

        def stats(nodes):
            for node in nodes:
                if isinstance(node, dict):
                    yield node
                else:
                    yield from stats(node)

        prefix = {
            "conditional": "cuda_conditional_nested_",
            "device_update": "cuda_device_update_nested_",
            "masked": "cuda_masked_",
            "vulkan": "vulkan_",
        }[route]
        nested = owner.executor._spec.structured_control_nodes[0]
        assert nested._native_upgrade_eligible
        nested_stats = list(stats(nested.debug_graph_stats))
        assert any(s["last_path"].startswith(prefix) for s in nested_stats), nested_stats
        if route == "conditional":
            assert nested_stats[0]["known_bounded_control_bytes"] == (regions + 1) * 32
        if route in ("masked", "device_update"):
            arrays["limits"].from_numpy(np.array([2, 1], dtype=np.int32))
            owner.executor.submit(arrays).wait()
            samples = []
            for _ in range(16):
                begin = time.perf_counter()
                owner.executor.submit(arrays).wait()
                samples.append((time.perf_counter() - begin) * 1e6)
            timings["jit_primal_warm_complete_us"] = statistics.median(samples)
        if route == "masked":
            # Masked payloads need a separate persistence contract.
            module = ti.aot.Module()
            with pytest.raises(ti.TaichiRuntimeError, match="selected nested CUDA"):
                module.add_execution_plan("step", owner.executor)
            assert not module._content and not module._graph_export_failed
            (tmp_path / "costs.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
            return
        if route == "device_update":
            begin = time.perf_counter()
            module.add_execution_plan("primal", owner.executor)
            timings["primal_export_ms"] = (time.perf_counter() - begin) * 1000
    begin = time.perf_counter()
    with build(True) as owner:
        timings["gradient_build_ms"] = (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        plan = module.add_execution_plan("step", owner.executor, gradients=dict(x="dx", y="dy"))
        timings["gradient_export_ms"] = (time.perf_counter() - begin) * 1000
        count = len(module._exported_kernels)
        module.add_execution_plan("again", owner.executor, gradients=dict(x="dx", y="dy"))
        assert len(module._exported_kernels) == count
        document = json.loads(plan.to_json())
        outer = next(op for op in document["operations"] if op["kind"] == "while")
        assert outer["max_iterations"] == budget
        inners = (
            outer["lowering"]["inners"]
            if arch == "vulkan" or route == "device_update"
            else [op for op in outer["body"] if op["kind"] == "while"]
        )
        assert len(inners) == regions and all(op["max_iterations"] == budget for op in inners)
        if route == "device_update":
            assert document["version"] == 13
            assert outer["lowering"]["kind"] == "cuda_nested_device_update"
            nested = owner.executor._spec.structured_control_nodes[0]
            assert len(outer["condition"]) + len(outer["body"]) == nested._cuda_nested.physical_dispatch_count
            assert all(op["kind"] == "kernel" for op in outer["condition"] + outer["body"])
            assert document["constraints"]  # Preserve JIT control scalar/disjoint binding contracts.
    timings["kernel_entries"] = len(module._exported_kernels)
    artifact = tmp_path / "nested.tcm"
    begin = time.perf_counter()
    module.archive(str(artifact))
    timings["archive_ms"] = (time.perf_counter() - begin) * 1000
    timings["artifact_bytes"] = artifact.stat().st_size
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_nested_capacity_probe.py")),
                mode,
                str(artifact),
                str(budget),
                str(regions),
                arch,
                "primal" if route == "device_update" else "gradient",
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        timings[mode] = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"load_ms"')))
    (tmp_path / "costs.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")


@pytest.mark.parametrize("budget,regions,lowering", ((65, 1, "native_required"), (1 << 40, 9, "auto")))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_compact_nested_capacity_and_aot_gradient(budget, regions, lowering, tmp_path, monkeypatch):
    if not dict(core.cuda_conditional_graph_capabilities()).get("nested_conditional_graph_available", False):
        pytest.skip("nested CUDA conditional Graph unavailable")
    monkeypatch.delenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", raising=False)
    limits = ti.graph.structured_control_capabilities()["device_control"]["nested_cuda_conditional_limits"]
    assert limits == dict(
        max_iterations=(1 << 63) - 1,
        max_ordered_inner_regions=None,
        max_static_dispatches=None,
        expands_iterations=False,
    )
    _check_nested_capacity(budget, regions, lowering, "conditional", tmp_path)


@pytest.mark.parametrize("route", ("device_update", "masked"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_expanded_cuda_nested_region_count(route, tmp_path, monkeypatch):
    monkeypatch.setenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "1" if route == "masked" else "0")
    capabilities = ti.graph.structured_control_capabilities()["device_control"]
    if route == "device_update" and not (
        capabilities["nested_cuda_device_update_qualified"] or capabilities["nested_cuda_device_update_candidate"]
    ):
        pytest.skip("CUDA nested device update unavailable")
    _check_nested_capacity(2, 9, "native_required", route, tmp_path)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_nested_region_count_and_aot_gradient(tmp_path):
    if not ti.graph.structured_control_capabilities()["nested_native_lowering"]:
        pytest.skip("nested Vulkan control unavailable")
    _check_nested_capacity(2, 9, "native_required", "vulkan", tmp_path)
