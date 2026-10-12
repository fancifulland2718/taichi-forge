import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge._lib import core, utils as runtime_utils
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_registered_aot_primitives_and_unique_gradient(tmp_path, monkeypatch):
    native = runtime_utils._loaded_native_runtime_path
    if native is None or not hasattr(core, "native_operation_capabilities"):
        pytest.skip("Requires the shared native operation registry")
    arch = ti.lang.impl.current_cfg().arch
    module = ti.aot.Module(caps=["spirv_has_int64"] if arch == ti.vulkan else [])
    assert core.native_operation_capabilities(arch, "compact", 1)
    assert not core.native_operation_capabilities(arch, "compact", 2)

    for kind, options in (
        ("missing", {}),
        ("unique", {"key_dtype": "f32"}),
        ("unique", {"values": "v"}),
        ("unique", {"indices": "__forge_primitive:invalid:heads"}),
        ("compact", {"columns": []}),
        ("sort", {"unknown": "value"}),
    ):
        with pytest.raises(ValueError):
            module.add_primitive("invalid", kind, **options)
    assert not module._execution_plans and not module._exported_kernels
    with monkeypatch.context() as patch:
        patch.setattr(core, "native_operation_capabilities", lambda *_: 0)
        with pytest.raises(ValueError, match="requires compact"):
            module.add_primitive("unavailable", "unique")
    assert not module._exported_kernels
    for dtype in (ti.i32, ti.u32, ti.i64, ti.u64):
        name = dtype.to_string()
        module.add_primitive(name, "unique", key_dtype=dtype)
    helper_count = len(module._exported_kernels)
    duplicate = module.add_primitive("i32_again", "unique")
    assert len(module._exported_kernels) == helper_count == 4
    assert duplicate.argument_names == ("count", "keys", "output")
    with pytest.raises(ValueError, match="Duplicate"):
        module.add_primitive("i32_again", "unique")
    for version in (0, -1, True, 2):
        with pytest.raises(ValueError):
            module.add_primitive("invalid", "unique", version=version)
    module.add_primitive("sort", "sort", keys="keys")
    module.add_primitive("compact", "compact")
    module.add_primitive("payload", "unique", values="values", unique_values="unique_values")
    assert len(module._exported_kernels) == helper_count
    indices = module.add_primitive("indices", "unique", indices="selected")
    assert len(module._exported_kernels) == helper_count + 1

    @ti.kernel
    def evaluate(
        x: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
        selected: ti.types.ndarray(dtype=ti.i32, ndim=1),
        count: ti.types.ndarray(dtype=ti.i32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
    ):
        for i in range(count[0]):
            y[i] = x[selected[i]] * x[selected[i]]

    module.add_kernel(evaluate)
    module.add_kernel(evaluate.grad, name="backward")
    document = json.loads(indices.to_json())
    document["operations"] += [
        {"kind": "fill32", "destination": "dx", "value": 0},
        {"kind": "fill32", "destination": "y", "value": 0},
        {"kind": "kernel", "entry": "evaluate", "arguments": ["x", "selected", "count", "y"]},
        {"kind": "kernel", "entry": "backward", "arguments": ["x", "selected", "count", "y"]},
    ]
    document["gradients"] = {"x": "dx", "y": "dy"}
    module.add_execution_plan("gradient", ti.aot.ExecutionPlan.from_json(json.dumps(document)))
    artifact = tmp_path / "primitives.tcm"
    module.archive(str(artifact))
    backend = core.arch_name(arch)
    if backend in ("x64", "arm64"):
        backend = "cpu"
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(native).parent),
        TAICHI_RUNTIME_DIR=runtime_utils._runtime_bitcode_dir(),
        TI_ENABLE_VERSION_CHECK="0",
    )
    result = subprocess.run(
        [sys.executable, "-S", str(Path(__file__).with_name("aot_primitives_probe.py")), backend, str(artifact)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Unable to release execution resource" not in result.stdout + result.stderr
