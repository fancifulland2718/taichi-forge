"""Explicit CUDA nested plans preserve physical kernels and owned controls."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import AotError, ExecutionPlan, Scalar
from tests import test_utils


def _kernel(entry, *arguments):
    return dict(kind="kernel", entry=entry, arguments=list(arguments))


@pytest.mark.parametrize("budget", (0, 2))
@pytest.mark.parametrize("encoding", ("device_update", "masked"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_nested_persistent_forward_reverse_and_owners(budget, encoding, tmp_path):
    caps = ti.graph.structured_control_capabilities()["device_control"]
    if encoding == "device_update" and not (
        caps["nested_cuda_device_update_qualified"] or caps["nested_cuda_device_update_candidate"]
    ):
        pytest.skip("CUDA device-update support unavailable")

    @ti.kernel
    def zero(value: ti.types.ndarray(ti.i32, 1)):
        value[0] = 0

    @ti.kernel
    def condition(
        count: ti.types.ndarray(ti.i32, 1),
        predicate: ti.types.ndarray(ti.i32, 1),
        limits: ti.types.ndarray(ti.i32, 1),
        slot: ti.i32,
    ):
        predicate[0] = int(count[0] < limits[slot])

    @ti.kernel
    def advance(value: ti.types.ndarray(ti.i32, 1)):
        value[0] += 1

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

    begin = time.perf_counter()
    module = ti.aot.Module()
    for kernel in (zero, condition, advance, forward, publish):
        module.add_kernel(kernel)
    module.add_kernel(forward.grad, name="reverse")
    timings = dict(compile_ms=(time.perf_counter() - begin) * 1000)
    begin = time.perf_counter()
    if encoding == "masked":
        # Private code-variant producer; automatic Graph export is separate.
        for kernel in (zero, condition, advance):
            name = kernel.__name__
            compiled = module._exported_kernels[name]
            entry = "gated_" + name
            module._aot_builder._add_with_launch_abi(entry, compiled, "cuda_graph_gate_v1")
            module._exported_kernels[entry] = compiled
            module._content.append("kernel:" + entry)
    timings["variant_generation_ms"] = (time.perf_counter() - begin) * 1000
    regions = 2
    private = ["outer", "op", "total"] + [f"{prefix}{i}" for i in range(regions) for prefix in ("count", "ip")]
    resources = [dict(kind="ndarray", name=name, dtype="i32", shape=[1]) for name in private]
    resources += [Scalar("i32", i).resource(f"slot{i}") for i in range(2)]
    outer_condition = _kernel("condition", "outer", "op", "limits", "slot0")
    body, inners = [], []
    for i in range(regions):
        count, predicate = f"count{i}", f"ip{i}"
        check = _kernel("condition", count, predicate, "limits", "slot1")
        body.append(_kernel("zero", count))
        condition_begin = 1 + len(body)
        body.append(check)
        body_begin = 1 + len(body)
        body.extend([_kernel("advance", count), _kernel("advance", "total"), check])
        inners.append(
            dict(
                predicate=predicate,
                condition_begin=condition_begin,
                body_begin=body_begin,
                end=1 + len(body),
                max_iterations=budget,
            )
        )
    body.extend([_kernel("advance", "outer"), outer_condition])
    if encoding == "masked":
        body = [dict(op, entry="gated_" + op["entry"]) for op in body]
    loop = dict(
        kind="while",
        predicate="op",
        max_iterations=budget,
        condition=[outer_condition],
        body=body,
        lowering=dict(kind="cuda_nested_" + encoding, version=1, inners=inners),
    )
    plan = ExecutionPlan(
        [
            _kernel("zero", "outer"),
            _kernel("zero", "total"),
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="fill32", destination="dy", value=0x3F800000),
            loop,
            _kernel("forward", "x", "y", "total"),
            _kernel("reverse", "x", "y", "total"),
            _kernel("publish", "outer", "total", "terminal"),
        ],
        resources=resources,
        gradients=dict(x="dx", y="dy"),
    )
    assert json.loads(plan.to_json())["version"] == (14 if encoding == "masked" else 13)
    count = len(module._exported_kernels)
    module.add_execution_plan("step", plan)
    module.add_execution_plan("again", plan)
    assert len(module._exported_kernels) == count == (9 if encoding == "masked" else 6)
    artifact = tmp_path / "nested.tcm"
    begin = time.perf_counter()
    module.archive(str(artifact))
    timings["save_ms"] = (time.perf_counter() - begin) * 1000
    timings["artifact_bytes"] = artifact.stat().st_size
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cuda_nested_plan_probe.py")),
                mode,
                str(artifact),
                str(budget),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        timings[mode] = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"load_ms"')))
    (tmp_path / "costs.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")


@test_utils.test(arch=[ti.cpu, ti.vulkan], offline_cache=False)
def test_cuda_nested_plan_rejects_other_backends_before_loading_payload(tmp_path):
    @ti.kernel
    def noop():
        pass

    producer = ti.aot.Module()
    producer.add_kernel(noop)
    artifact = tmp_path / "empty.tcm"
    producer.archive(str(artifact))
    payload = _kernel("missing_payload")
    plan = ExecutionPlan(
        [
            dict(
                kind="while",
                predicate="p",
                max_iterations=0,
                condition=[payload],
                body=[payload] * 3,
                lowering=dict(
                    kind="cuda_nested_device_update",
                    version=1,
                    inners=[dict(predicate="ip", condition_begin=1, body_begin=2, end=3, max_iterations=0)],
                ),
            )
        ]
    )
    with ti.aot.borrow_runtime() as runtime:
        with runtime.load_module(artifact) as module:
            p, ip = runtime.ndarray("i32", 1), runtime.ndarray("i32", 1)
            for encoding in ("device_update", "masked"):
                document = json.loads(plan.to_json())
                document["version"] = 14
                document["operations"][0]["lowering"]["kind"] = "cuda_nested_" + encoding
                selected = ExecutionPlan.from_json(json.dumps(document))
                with pytest.raises(AotError, match="requires a CUDA runtime"):
                    module.prepare_execution(selected, dict(p=p, ip=ip))
