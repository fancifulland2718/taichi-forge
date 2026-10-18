"""Sparse topology, lists and final-state gradients inside native control."""

import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_sparse_retirement import _make_list_growth_export, _sparse_module


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

    @ti.kernel
    def clear_output(out: ti.types.ndarray(ti.f32, 1)):
        for i in range(3):
            out[i] = 0

    @ti.kernel
    def reset_counter(counter: ti.types.ndarray(ti.i32, 1)):
        counter[0] = 0

    @ti.kernel
    def outer_condition(counter: ti.types.ndarray(ti.i32, 1), predicate: ti.types.ndarray(ti.i32, 1), target: ti.i32):
        predicate[0] = ti.cast(counter[0] < target, ti.i32)

    module.add_kernel(condition)
    module.add_kernel(advance)
    if ti.cfg.arch == ti.vulkan:
        module.add_kernel(clear_output)
        module.add_kernel(reset_counter)
        module.add_kernel(outer_condition)
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
    cases = [("loop", 4, "compact"), ("zero", 0, "compact")]
    if ti.cfg.arch == ti.vulkan:
        loop["body"][2] = dict(kind="kernel", entry="clear_output", arguments=["out"])
        cases += [(strategy, 4, strategy) for strategy in ("chained", "conditional", "coarse_conditional")]
    for name, budget, strategy in cases:
        selected = dict(loop, max_iterations=budget)
        if ti.cfg.arch == ti.vulkan:
            selected["lowering"] = dict(
                kind="vulkan_structured",
                version=1,
                counter="counter",
                chunks=[dict(iterations=budget, strategy=strategy)] if budget else [],
            )
        module.add_execution_plan(
            name,
            ti.aot.ExecutionPlan(prefix + [selected] + suffix, resources=resources, gradients={"loss": "dloss"}),
        )
    if ti.cfg.arch == ti.vulkan:
        outer_test = dict(
            kind="kernel", entry="outer_condition", arguments=["outer_counter", "outer_predicate", "outer_target"]
        )
        inner_start = 2  # outer condition, then reset the inner counter
        body_start = inner_start + len(loop["condition"])
        inner_end = body_start + len(loop["body"]) + len(loop["condition"])
        nested = dict(
            kind="while",
            predicate="outer_predicate",
            max_iterations=2,
            condition=[outer_test],
            body=[dict(kind="kernel", entry="reset_counter", arguments=["counter"])]
            + loop["condition"]
            + loop["body"]
            + loop["condition"]
            + [dict(kind="kernel", entry="advance", arguments=["outer_counter"]), outer_test],
            lowering=dict(
                kind="vulkan_nested",
                version=1,
                counter="outer_counter",
                inners=[
                    dict(
                        predicate="predicate",
                        counter="counter",
                        condition_begin=inner_start,
                        body_begin=body_start,
                        end=inner_end,
                        max_iterations=4,
                        chunk_size=4,
                    )
                ],
            ),
        )
        module.add_execution_plan(
            "nested",
            ti.aot.ExecutionPlan(
                prefix + [dict(kind="fill32", destination="outer_counter", value=0), nested] + suffix,
                resources=resources
                + [dict(kind="ndarray", name=n, dtype="i32", shape=[1]) for n in ("outer_counter", "outer_predicate")],
                gradients={"loss": "dloss"},
            ),
        )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_sparse_native_control(tmp_path):
    growth = _make_list_growth_export(tmp_path) if ti.cfg.arch == ti.vulkan else None

    def extra(module):
        export_control(module)
        if growth is not None:
            growth(module)

    _sparse_module(
        tmp_path,
        {ti.cpu: "bc", ti.cuda: "ptx", ti.vulkan: None}[ti.cfg.arch],
        export_extra=extra,
        consumer="aot_sparse_control_probe.py",
    )
