"""Public pack identities survive template removal and physical resource lifting."""
import json
import shutil

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_argument_origins_export_aliases_and_validate_compiled_abi(tmp_path):
    child = ti.types.argpack(values=ti.types.ndarray(ti.f32, 1))
    pack = ti.types.argpack(child=child)
    x, y, out = (ti.ndarray(ti.f32, 1) for _ in range(3))
    x.fill(3)
    y.fill(7)
    left, right = pack(child=child(values=x)), pack(child=child(values=y))

    @ti.kernel
    def evaluate(factor: ti.template(), left: pack, bias: ti.i32,
                 extra: ti.template(), right: pack, out: ti.types.ndarray(ti.f32, 1)):
        out[0] = factor * left.child.values[0] + right.child.values[0] + bias + extra

    evaluate(10, left, 2, 1, right, out)
    assert out[0] == 40
    evaluate(10, right, 2, 1, left, out)
    assert out[0] == 76
    module = ti.aot.Module()
    for name in ("entry", "alias"):
        module.add_kernel(evaluate, name=name, template_args={"factor": 10, "extra": 1, "out": out})
    good = tmp_path / "good"
    module.save(str(good))
    metadata = json.loads((good / "argument_origins.json").read_text())
    assert metadata["version"] == 1 and len(metadata["layouts"]) == 1
    kernels = {item["key"]: item["value"] for item in metadata["kernels"]}
    assert kernels["entry"] == kernels["alias"]
    origins = {tuple(item["logical"]): item for item in metadata["layouts"][0]}
    assert [origins[(i,)]["name"] for i in range(4)] == ["left", "bias", "right", "out"]
    assert origins[(0, 0, 0)]["physical"] == [4]
    assert origins[(2, 0, 0)]["physical"] == [5]
    assert origins[(0, 0, 0)]["name"] == origins[(2, 0, 0)]["name"] == "values"
    assert origins[(0, 0)]["physical"] == [0, 0]
    assert origins[(2, 0)]["physical"] == [2, 0]
    with ti.aot.borrow_runtime() as runtime:
        with runtime.load_module(good) as loaded:
            assert loaded.kernel("entry") and loaded.kernel("alias")
        # Keep the public tree well formed but swap a scalar and an ndarray's
        # physical slots. The loaded compiled ABI must reject this mapping.
        bad = tmp_path / "bad"
        shutil.copytree(good, bad)
        origins[(0, 0, 0)]["physical"], origins[(1,)]["physical"] = (
            origins[(1,)]["physical"], origins[(0, 0, 0)]["physical"]
        )
        (bad / "argument_origins.json").write_text(json.dumps(metadata))
        with runtime.load_module(bad) as loaded:
            with pytest.raises(RuntimeError, match="origin.*compiled kind|value origin names a resource"):
                loaded.kernel("entry")
        with runtime.load_module(good) as loaded:
            assert loaded.kernel("entry")  # Rejection does not poison another module.

    @ti.kernel
    def plain(out: ti.types.ndarray(ti.f32, 1)):
        out[0] = 1

    ordinary = ti.aot.Module()
    ordinary.add_kernel(plain)
    ordinary.save(str(tmp_path / "plain"))
    assert not (tmp_path / "plain/argument_origins.json").exists()
