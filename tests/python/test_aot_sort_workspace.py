import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_sort_workspace_regions_and_frames(tmp_path):
    arch = (
        "cpu"
        if ti.cfg.arch == ti.cpu
        else "cuda" if ti.cfg.arch == ti.cuda else "vulkan"
    )
    module = ti.aot.Module()
    operations = [
        dict(kind="sort", keys=f"keys{i}", values=f"values{i}") for i in range(2)
    ]
    module.add_execution_plan("one", ti.aot.ExecutionPlan(operations[:1]))
    module.add_execution_plan("pair", ti.aot.ExecutionPlan(operations))
    module.add_execution_plan(
        "initialized", ti.aot.ExecutionPlan(operations, initialization=operations[:1])
    )
    module.add_execution_plan(
        "regions",
        ti.aot.ExecutionPlan(
            [
                dict(
                    kind="branch",
                    selector="selector",
                    selection="nonzero",
                    has_default=True,
                    condition=[],
                    bodies=[operations, list(reversed(operations))],
                    lowering=dict(
                        kind="cpu" if arch == "cpu" else arch + "_host", version=1
                    ),
                )
            ]
        ),
    )
    if arch == "cuda":
        module.add_execution_plan(
            "family",
            ti.aot.ExecutionPlan(
                operations, binding_strategy="cuda_immutable_argument_frames"
            ),
        )

        @ti.kernel
        def read_first(
            keys: ti.types.ndarray(ti.i32, 1), out: ti.types.ndarray(ti.i32, 1)
        ):
            out[0] = keys[0]

        module.add_kernel(read_first)
        parallel = dict(
            kind="cuda_parallel",
            branches=[
                [
                    dict(
                        kind="kernel",
                        entry="read_first",
                        arguments=[f"keys{i}", f"out{i}"],
                    )
                ]
                for i in range(2)
            ],
        )
        module.add_execution_plan(
            "parallel", ti.aot.ExecutionPlan([operations[0], parallel, operations[1]])
        )
    artifact = tmp_path / "sort-workspace.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_sort_workspace_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
