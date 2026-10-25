"""Preserve storage-image argument metadata even without an image instruction."""

import json

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_rwtexture_format_and_shape_only_mip_metadata(tmp_path):
    @ti.kernel
    def shape(image: ti.types.rw_texture(2, fmt=ti.Format.r32f, lod=1), output: ti.types.ndarray(ti.i32, 1)):
        output[0] = image.shape[0]
        output[1] = image.shape[1]

    @ti.kernel
    def ordinary(output: ti.types.ndarray(ti.i32, 1)):
        output[0] = 7

    module = ti.aot.Module()
    module.add_kernel(shape)
    module.save(str(tmp_path / "image"))
    path = tmp_path / "image" / "metadata.json"
    metadata = json.loads(path.read_text())
    assert metadata["metadata_version"] == 12
    argument = next(
        item["value"] for item in metadata["kernels"][0]["ctx_attribs"]["arg_attribs_vec_"] if item["key"] == [0]
    )
    assert argument["format"] == int(ti.Format.r32f)
    assert argument["texture_lod"] == 1
    assert all(not task["texture_binds"] for task in metadata["kernels"][0]["tasks_attribs"])
    with ti.aot.borrow_runtime() as runtime:
        with runtime.load_module(path.parent):
            pass
        metadata["metadata_version"] = 2
        path.write_text(json.dumps(metadata))
        with pytest.raises(RuntimeError):
            runtime.load_module(path.parent)
    plain = ti.aot.Module()
    plain.add_kernel(ordinary)
    plain.save(str(tmp_path / "ordinary"))
    assert json.loads((tmp_path / "ordinary" / "metadata.json").read_text())["metadata_version"] == 12
