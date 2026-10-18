"""Sparse topology, lists and final-state gradients inside native control."""

import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_sparse_retirement import _sparse_module


def export_control(module):
    @ti.kernel
    def condition(
        counter: ti.types.ndarray(ti.i32, 1),
        predicate: ti.types.ndarray(ti.i32, 1),
        checks: ti.types.ndarray(ti.i32, 1),
        target: ti.i32,
    ):
        predicate[0] = ti.cast(counter[0] < target, ti.i32)
        checks[0] += 1

    @ti.kernel
    def advance(counter: ti.types.ndarray(ti.i32, 1)):
        counter[0] += 1

    module.add_kernel(condition)
    module.add_kernel(advance)
    resources = [dict(kind="ndarray", name=name, dtype="i32", shape=[1]) for name in ("counter", "predicate", "checks")]
    resources.append(dict(kind="ndarray", name="dloss", dtype="f32", shape=[1]))
    prefix = [
        dict(kind="fill32", destination=name, value=0) for name in ("counter", "checks", "out", "loss", "gradient")
    ]
    prefix += [
        dict(kind="kernel", entry="deactivate", arguments=[]),
        dict(kind="fill32", destination="dloss", value=0x3F800000),
    ]
    loop = dict(
        kind="while",
        predicate="predicate",
        max_iterations=4,
        condition=[dict(kind="kernel", entry="condition", arguments=["counter", "predicate", "checks", "target"])],
        body=[
            dict(kind="kernel", entry="deactivate", arguments=[]),
            dict(kind="kernel", entry="activate", arguments=["scale"]),
            dict(kind="fill32", destination="out", value=0),
            dict(kind="kernel", entry="gather", arguments=["out"]),
            dict(kind="kernel", entry="advance", arguments=["counter"]),
        ],
    )
    suffix = [
        dict(kind="kernel", entry="sparse_energy", arguments=["loss"]),
        dict(kind="kernel", entry="sparse_backward", arguments=["loss"]),
        dict(kind="kernel", entry="gather_grad", arguments=["gradient"]),
        dict(kind="copy", source="counter", destination="state", size_bytes=4),
        dict(kind="copy", source="checks", destination="state", destination_offset=4, size_bytes=4),
    ]
    for name, budget in (("loop", 4), ("zero", 0)):
        module.add_execution_plan(
            name,
            ti.aot.ExecutionPlan(
                prefix + [dict(loop, max_iterations=budget)] + suffix, resources=resources, gradients={"loss": "dloss"}
            ),
        )


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_sparse_native_control(tmp_path):
    _sparse_module(
        tmp_path,
        "ptx" if ti.cfg.arch == ti.cuda else "bc",
        export_extra=export_control,
        consumer="aot_sparse_control_probe.py",
    )
