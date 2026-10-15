"""Persist helper binding requirements independently of live Graph owners."""

import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_binding_constraints_forward_reverse_and_import_aliases(tmp_path):
    @ti.kernel
    def forward(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in range(17):
            y[i] = x[i] * x[i] + 2 * x[i]

    @ti.kernel
    def inplace(x: ti.types.ndarray(ti.f32, ndim=1), y: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(17):
            y[i] = 2 * x[i] + 3

    module = ti.aot.Module()
    module.add_execution_plan(
        "typed_fill",
        ti.aot.ExecutionPlan(
            [dict(kind="fill32", destination="x", value=0x3F800000)],
            constraints=[
                dict(kind="array_layout", array="x", dtype="f32", exact_shape=[17], record_bytes=4, alignment=4)
            ],
        ),
    )
    module.add_kernel(forward)
    module.add_kernel(forward.grad, name="backward")
    module.add_kernel(inplace)
    constraints = [
        dict(kind="array_layout", array=name, exact_shape=[17], record_bytes=4, alignment=4)
        for name in ("x", "y", "dx", "dy")
    ]
    constraints.extend(
        dict(kind="disjoint", left=a, right=b) for a, b in (("x", "y"), ("x", "dx"), ("y", "dx"), ("dy", "y"))
    )
    module.add_execution_plan(
        "differentiate",
        ti.aot.ExecutionPlan(
            [
                dict(kind="fill32", destination="dx", value=0),
                dict(kind="fill32", destination="dy", value=0x3F800000),
                dict(kind="kernel", entry="forward", arguments=["x", "y"]),
                dict(kind="kernel", entry="backward", arguments=["x", "y"]),
            ],
            resources=[dict(kind="ndarray", name="dy", dtype="f32", shape=[17])],
            gradients=dict(x="dx", y="dy"),
            constraints=constraints,
        ),
    )
    module.add_execution_plan(
        "tensor",
        ti.aot.ExecutionPlan(
            [dict(kind="copy", source="x", destination="y")],
            constraints=[
                dict(
                    kind="array_layout",
                    array=name,
                    minimum_shape=[3],
                    element_shape=[2, 2],
                    record_bytes=16,
                    alignment=4,
                )
                for name in ("x", "y")
            ]
            + [dict(kind="disjoint", left="x", right="y")],
        ),
    )
    same_view = dict(kind="disjoint", left="x", right="y", allow_same_view=True)
    module.add_execution_plan(
        "same_or_disjoint",
        ti.aot.ExecutionPlan(
            [dict(kind="kernel", entry="inplace", arguments=["x", "y"])],
            constraints=[
                dict(kind="array_layout", array=name, exact_shape=[17], record_bytes=4, alignment=4)
                for name in ("x", "y")
            ]
            + [same_view],
        ),
    )
    module.add_execution_plan(
        "alias_only",
        ti.aot.ExecutionPlan(
            [dict(kind="fill32", destination=name, value=0) for name in ("x", "y")], constraints=[same_view]
        ),
    )
    for name, minimum, record_bytes, alignment in (
        ("aligned", 17, 4, 8),
        ("wrong_record", 3, 8, 4),
        ("empty", 0, 4, 4),
    ):
        module.add_execution_plan(
            name,
            ti.aot.ExecutionPlan(
                [dict(kind="copy", source="x", destination="y")],
                constraints=[
                    dict(
                        kind="array_layout",
                        array="x",
                        minimum_shape=[minimum],
                        element_shape=[2, 2] if name == "wrong_record" else [],
                        record_bytes=record_bytes,
                        alignment=alignment,
                    ),
                    dict(kind="disjoint", left="x", right="y"),
                ],
            ),
        )
    artifact = tmp_path / "constraints.tcm"
    module.archive(str(artifact))
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_binding_constraints_probe.py")), mode, arch, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
