"""Cached kernels must not depend on unrelated producer SNode trees."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize(
    "arch",
    [a for a in (ti.cpu, ti.cuda, ti.vulkan) if a in test_utils.expected_archs()],
)
@pytest.mark.parametrize("uses_field", [False, True])
def test_cached_kernel_ignores_unreferenced_roots(tmp_path, arch, uses_field):
    script = tmp_path / "cached_roots.py"
    script.write_text(
        textwrap.dedent("""
            import json
            import sys
            import taichi_forge as ti

            ti.init(arch=getattr(ti, sys.argv[1]), enable_fallback=False,
                    offline_cache=True, offline_cache_file_path=sys.argv[2])
            uses_field = sys.argv[3] == '1'
            extra_count = int(sys.argv[4])
            owners = []
            field = None
            if uses_field:
                field = ti.field(ti.i32)
                builder = ti.FieldsBuilder()
                builder.dense(ti.i, 8).place(field)
                owners.append((field, builder.finalize()))
            for _ in range(extra_count):
                extra = ti.field(ti.i32)
                builder = ti.FieldsBuilder()
                builder.dense(ti.i, 4).place(extra)
                owners.append((extra, builder.finalize()))
            runtime = ti.lang.impl.get_runtime()
            runtime.materialize()
            runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)

            @ti.kernel
            def fill(data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
                for i in range(8):
                    if ti.static(uses_field):
                        field[i] = i * 2
                        data[i] = field[i] + 3
                    else:
                        data[i] = i * 2 + 3

            data = ti.ndarray(ti.i32, shape=8)
            fill(data)
            ti.sync()
            print('ROOT_CACHE=' + json.dumps(runtime.debug_kernel_executable_lifecycle_stats()))
            assert data.to_numpy().tolist() == list(range(3, 19, 2))
            if uses_field:
                assert field.to_numpy().tolist() == list(range(0, 16, 2))
            ti.reset()
            """),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["TI_ENABLE_VERSION_CHECK"] = "0"
    for extra_count in (3, 0):
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                ti._lib.core.arch_name(arch),
                str(tmp_path / "cache"),
                str(int(uses_field)),
                str(extra_count),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        stats = json.loads(
            next(
                line.removeprefix("ROOT_CACHE=")
                for line in completed.stdout.splitlines()
                if line.startswith("ROOT_CACHE=")
            )
        )
        (tmp_path / f"stats-{extra_count}.json").write_text(
            json.dumps(stats), encoding="utf-8"
        )
        if extra_count:
            assert stats["compiler_invocations"] == 1
        else:
            assert stats["disk_loads"] == 1
            assert stats["compiler_invocations"] == 0
