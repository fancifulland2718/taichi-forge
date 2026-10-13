"""Exported GFX root layouts follow used resources, not builder creation time."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.vulkan], offline_cache=False)
def test_gfx_aot_incremental_layouts_preserve_primal_reverse_and_compact_roots(tmp_path):
    unused = ti.field(ti.f32, shape=19)
    unused.fill(7)

    @ti.kernel
    def square(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True), y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)
    ):
        for i in x:
            y[i] = x[i] * x[i]

    module = ti.aot.Module()
    module.add_kernel(square)
    module.add_kernel(square.grad, name="square_grad")
    # An explicitly exported later field belongs to a new producer SNode tree.
    scale = ti.field(ti.f32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 5).place(scale)
    scale_tree = fields.finalize()
    scale.fill(3)

    @ti.kernel
    def seed(value: ti.f32):
        for i in scale:
            scale[i] = value

    @ti.kernel
    def apply_field(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True), y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)
    ):
        for i in x:
            y[i] = scale[i] * x[i] * x[i]

    module.add_kernel(seed)
    module.add_field("scale", scale)
    module.add_kernel(apply_field)
    module.add_kernel(apply_field.grad, name="field_grad")
    module.add_kernel(square, name="square_alias")
    output = tmp_path / "mixed"
    module.save(str(output))
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["root_buffer_sizes"] == [20]
    assert metadata["fields"][0]["snode_tree_id"] == 0
    assert len(metadata["kernels"]) == 5
    assert metadata["kernel_aliases"] == [{"name": "square_alias", "kernel_index": 0}]
    roots = dict(zip((k["name"] for k in metadata["kernels"]), metadata["kernel_metadata"]))
    assert all(value["num_snode_trees"] == 1 for value in roots.values())
    assert roots["square"]["used_snode_tree_ids"] == roots["square_grad"]["used_snode_tree_ids"] == []
    assert all(roots[name]["used_snode_tree_ids"] == [0] for name in ("seed", "apply_field", "field_grad"))
    # A newer Module may reuse a kernel compiled against the older global table.
    arrays_only = ti.aot.Module()
    arrays_only.add_kernel(square)
    arrays_only.add_kernel(square.grad, name="square_grad")
    arrays_only.save(str(tmp_path / "arrays"))
    metadata = json.loads((tmp_path / "arrays/metadata.json").read_text())
    assert metadata["root_buffer_sizes"] == []
    assert all(item == {"num_snode_trees": 0, "used_snode_tree_ids": []} for item in metadata["kernel_metadata"])
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_incremental_layouts_probe.py")),
                mode,
                str(tmp_path),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
    scale_tree.destroy()
    with pytest.raises(RuntimeError, match="inactive SNodeTree"):
        module.save(str(tmp_path / "retired"))
