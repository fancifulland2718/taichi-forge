"""Shared native bucket ownership and source-free AOT reconstruction."""

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import taichi_forge as ti
from tests import test_utils


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4
)
def test_bucket_native_registry_runtime_only_and_borrowed(tmp_path):
    arch = ti.lang.impl.current_cfg().arch
    module = ti.aot.Module()
    configurations = [
        (dtype, [], 17, 7, False)
        for dtype in ("i32", "u32", "f32", "i64", "u64", "f64")
    ]
    configurations += [
        ("f32", [3], 2051, 7, True),
        ("i32", [], 17, 4097, True),
        ("i32", [], 0, 7, True),
    ]
    if arch == ti.cpu:
        # Exercise both existing parallel policies at their actual activation
        # boundary. A one-lane tensor must retain the scalar copy strategy.
        configurations += [
            ("i32", [1], 262147, 7, False),
            ("f32", [3], 262147, 7, False),
        ]
    cases = []
    for index, (dtype, element_shape, n, bins, private) in enumerate(configurations):
        name = f"bucket_{index}"
        parameters = dict(
            size=n,
            num_bins=bins,
            dtype=dtype,
            element_shape=element_shape,
            cpu_max_threads=4,
            vulkan_policy=dict(
                private_enabled=private,
                private_min_items=0,
                private_max_bins=8192,
                private_max_bytes=1 << 24,
            ),
        )
        bindings = {
            name: name + "_view" for name in ("keys", "values", "offsets", "output")
        }
        module.add_execution_plan(
            name,
            ti.aot.ExecutionPlan(
                [
                    dict(
                        kind="native",
                        provider="builtin",
                        operation="bucket_builder",
                        version=1,
                        parameters=parameters,
                        bindings=bindings,
                    )
                ],
                resources=[
                    dict(
                        kind="ndarray_view",
                        name=target,
                        array=source,
                        offset=1,
                        shape=[bins + 2 if source == "offsets" else n + 1],
                    )
                    for source, target in bindings.items()
                ],
            ),
        )
        cases.append(dict(name=name, **parameters))
    artifact = tmp_path / "bucket.tcm"
    module.archive(str(artifact))
    assert not module._exported_kernels
    with zipfile.ZipFile(artifact) as archive:
        assert not any(
            name.endswith((".bc", ".ptx", ".spv", ".cubin"))
            for name in archive.namelist()
        )
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf8")
    backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_bucket_builder_probe.py")),
                mode,
                backend,
                str(artifact),
            ],
            env=dict(os.environ, TI_VULKAN_BUCKET_BUILDER_PRIVATE="0"),
            capture_output=True,
            text=True,
            timeout=90,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "Validation Error" not in output and "SYNC-HAZARD" not in output, output
