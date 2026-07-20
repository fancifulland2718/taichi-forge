import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_sparse_opt_out_replaces_previous_program_gate():
    enabled = ti.field(ti.f32)
    ti.root.pointer(ti.i, 8).place(enabled)

    ti.reset()
    ti.init(
        arch=ti.vulkan,
        vulkan_sparse_experimental=False,
        offline_cache=False,
    )

    disabled = ti.field(ti.f32)
    with pytest.raises(
        TaichiRuntimeError,
        match="Pointer SNode is not supported on this backend",
    ):
        ti.root.pointer(ti.i, 8).place(disabled)

    ti.reset()
    ti.init(
        arch=ti.vulkan,
        vulkan_sparse_experimental=True,
        offline_cache=False,
    )

    reenabled = ti.field(ti.f32)
    ti.root.pointer(ti.i, 8).place(reenabled)
