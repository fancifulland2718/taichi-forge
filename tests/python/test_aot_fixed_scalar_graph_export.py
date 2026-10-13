"""Export Graph's final fixed scalar frame using each physical argument type."""

import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_fixed_scalar_graph_preserves_values_types_and_reverse(tmp_path):
    @ti.kernel
    def forward(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        n: ti.i32,
        gain: ti.f32,
    ):
        for i in range(n):
            y[i] = gain * x[i] * x[i]

    @ti.kernel
    def seed(dx: ti.types.ndarray(ti.f32, ndim=1), dy: ti.types.ndarray(ti.f32, ndim=1), n: ti.i32):
        for i in range(n):
            dx[i] = 0
            dy[i] = 1

    @ti.kernel
    def encode(
        out: ti.types.ndarray(ti.u64, ndim=1), wide: ti.u64, signed: ti.i64, zero: ti.f32, count: ti.u32, offset: ti.i32
    ):
        out[offset] = wide
        out[offset + 1] = ti.bit_cast(signed, ti.u64)
        out[offset + 2] = ti.cast(ti.bit_cast(zero, ti.u32), ti.u64)
        out[offset + 3] = count

    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    module = ti.aot.Module(caps=["spirv_has_int64"] if arch == "vulkan" else [])
    forward_label = "solver/" + "阶段/" * 64 + "前向"
    for name, gain_value in (("negative", -1.75), ("positive", 0.5)):
        builder = ti.graph.GraphBuilder(_map_recipe="map2")
        public = {key: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, key, ti.f32, ndim=1) for key in ("x", "y", "dx", "dy")}
        encoded = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "encoded", ti.u64, ndim=1)
        first = builder.create_sequential()
        n = first._bind_internal_scalar("n", ti.i32, 17)
        gain = first._bind_internal_scalar("gain", ti.f32, gain_value)
        same_gain = first._bind_internal_scalar("gain_reverse", ti.f32, float(str(gain_value)))
        wide = first._bind_internal_scalar("wide", ti.u64, (1 << 64) - 1)
        signed = first._bind_internal_scalar("signed", ti.i64, -1)
        count = first._bind_internal_scalar("count", ti.u32, 17)
        zero = first._bind_internal_scalar("zero", ti.f32, 0.0)
        offset = first._bind_internal_scalar("first_offset", ti.i32, 0)
        first.dispatch(seed, public["dx"], public["dy"], n, label="solver/seed")
        first.dispatch(forward, public["x"], public["y"], n, gain, label=forward_label)
        first.dispatch(forward.grad, public["x"], public["y"], n, same_gain, label="solver/reverse")
        first.dispatch(encode, encoded, wide, signed, zero, count, offset, label="encode/first")
        builder.append(first)
        last = builder.create_sequential()
        # Equal values merge, with the final fixed frame retaining negative zero.
        zero = last._bind_internal_scalar("zero", ti.f32, -0.0)
        offset = last._bind_internal_scalar("last_offset", ti.i32, 4)
        last.dispatch(encode, encoded, wide, signed, zero, count, offset, label="encode/last")
        builder.append(last)
        graph = builder.compile()
        assert math.copysign(1, graph._spec.fixed_runtime_args["zero"]) == -1
        arrays = {key: ti.ndarray(ti.f32, 17, needs_grad=True) for key in ("x", "y")}
        arrays.update(dx=arrays["x"].grad, dy=arrays["y"].grad, encoded=ti.ndarray(ti.u64, 8))
        values = (np.arange(17, dtype=np.float32) - 8) / 8
        arrays["x"].from_numpy(values)
        # Python JIT Graph explicitly rejects ndarray AD dispatches; ordinary
        # JIT kernel calls provide the reference, AOT supplies explicit bindings.
        seed(arrays["dx"], arrays["dy"], 17)
        forward(arrays["x"], arrays["y"], 17, gain_value)
        forward.grad(arrays["x"], arrays["y"], 17, gain_value)
        for offset_value in (0, 4):
            encode(arrays["encoded"], (1 << 64) - 1, -1, graph._spec.fixed_runtime_args["zero"], 17, offset_value)
        np.testing.assert_array_equal(arrays["y"].to_numpy(), gain_value * values * values)
        np.testing.assert_array_equal(arrays["dx"].to_numpy(), 2 * gain_value * values)
        expected = np.array([(1 << 64) - 1, (1 << 64) - 1, 1 << 31, 17] * 2, dtype=np.uint64)
        np.testing.assert_array_equal(arrays["encoded"].to_numpy(), expected)
        before = len(module._exported_kernels)
        with pytest.raises(ti.TaichiRuntimeError, match="ndarrays"):
            module.add_execution_plan("bad_gradient", graph, gradients=dict(gain="dx"))
        assert len(module._exported_kernels) == before
        plan = module.add_execution_plan(name, graph, gradients=dict(x="dx", y="dy"))
        assert plan.argument_names == ("dx", "dy", "encoded", "x", "y")
        document = json.loads(plan.to_json())
        steps = document["operations"]
        assert [step["label"] for step in steps] == [
            "solver/seed",
            forward_label,
            "solver/reverse",
            "encode/first",
            "encode/last",
        ]
        assert steps[-1]["entry"] == steps[-2]["entry"], "Labels must not duplicate compiled code"
        resources = document["resources"]
        assert all(r["kind"] == "scalar" for r in resources)
        payloads = [(r["dtype"], r["bits"]) for r in resources]
        assert len(set(payloads)) == len(payloads), "Identical typed values share one description"
        assert ("i64", "ffffffffffffffff") in payloads and ("u64", "ffffffffffffffff") in payloads
        assert ("i32", "00000011") in payloads and ("u32", "00000011") in payloads
        assert ("f32", "80000000") in payloads and ("f32", "00000000") not in payloads
        graph.close()
        del graph, builder, first, last, arrays
        gc.collect()
    artifact = tmp_path / "fixed-scalars.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_fixed_scalar_graph_probe.py")), mode, arch, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_labeled_reduction_preserves_memory_contracts():
    builder = ti.graph.GraphBuilder()
    values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder.reduce(values, output, count=17)
    graph = builder.compile()
    assert graph._spec.memory_layout_requirements
    assert graph._spec.memory_disjoint_pairs
    module = ti.aot.Module()
    plan = module.add_execution_plan("reduction", graph)
    constraints = json.loads(plan.to_json())["constraints"]
    assert len(constraints) == 3
    assert {value["array"] for value in constraints if value["kind"] == "array_layout"} == {"values", "output"}
    assert dict(kind="disjoint", left="output", right="values") in constraints
    assert all(operation.get("label") for operation in json.loads(plan.to_json())["operations"])
    graph.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_aot_fixed_scalar_matches_jit_graph_merged_zero(tmp_path):
    @ti.kernel
    def bits(out: ti.types.ndarray(ti.u32, ndim=1), value: ti.f32, index: ti.i32):
        out[index] = ti.bit_cast(value, ti.u32)

    builder = ti.graph.GraphBuilder()
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.u32, ndim=1)
    for index, value in enumerate((0.0, -0.0)):
        sequence = builder.create_sequential()
        scalar = sequence._bind_internal_scalar("value", ti.f32, value)
        index_arg = sequence._bind_internal_scalar(f"index_{index}", ti.i32, index)
        sequence.dispatch(bits, output, scalar, index_arg)
        builder.append(sequence)
    sequence = builder.create_sequential()
    overflow = sequence._bind_internal_scalar("overflow", ti.f32, 1e100)
    index_arg = sequence._bind_internal_scalar("overflow_index", ti.i32, 2)
    sequence.dispatch(bits, output, overflow, index_arg)
    builder.append(sequence)
    graph = builder.compile()
    array = ti.ndarray(ti.u32, 3)
    graph.run(dict(out=array))
    np.testing.assert_array_equal(array.to_numpy(), np.array([1 << 31, 1 << 31, 0x7F800000], dtype=np.uint32))
    module = ti.aot.Module()
    module.add_execution_plan("bits", graph)
    artifact = tmp_path / "merged-zero.tcm"
    module.archive(str(artifact))
    graph.close()
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        result = runtime.ndarray("u32", (3,))
        prepared = loaded.prepare_execution("bits", dict(out=result))
        prepared()
        assert result.read() == b"\0\0\0\x80" * 2 + b"\0\0\x80\x7f"


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_aot_fixed_scalar_graph_rejects_unrepresentable_or_dynamic_values():
    @ti.kernel
    def write(out: ti.types.ndarray(ti.i32, ndim=1), value: ti.i32):
        out[0] = value

    class DynamicInt(int):
        pass

    for value in ((1 << 40), DynamicInt(17)):
        builder = ti.graph.GraphBuilder()
        sequence = builder.create_sequential()
        fixed = sequence._bind_internal_scalar("value", ti.i32, value)
        sequence.dispatch(write, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1), fixed)
        builder.append(sequence)
        graph = builder.compile()
        module = ti.aot.Module()
        with pytest.raises(ti.TaichiRuntimeError, match="cannot be encoded|unsupported values"):
            module.add_execution_plan("invalid", graph)
        assert not module._content and not module._exported_kernels and not module._execution_plans
        graph.close()
