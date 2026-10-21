"""Debug launch synchronization must not execute inside stream capture."""

import array

import pytest
import taichi_forge as ti
from taichi_forge_runtime import aot
from tests import test_utils


@test_utils.test(arch=ti.cuda, debug=True, offline_cache=False)
def test_borrowed_debug_preparation_and_assertion(tmp_path):
    @ti.kernel
    def advance(value: ti.types.ndarray(ti.i32, ndim=1)):
        assert value[0] >= 0, "negative prepared input"
        value[0] += 1

    module = ti.aot.Module()
    module.add_kernel(advance)
    module.save(str(tmp_path))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(tmp_path)
        value = runtime.ndarray("i32", 1)
        value.write(bytes(4))
        prepared = loaded.kernel("advance").prepare(value)
        assert value.read() == bytes(4)
        prepared.run()
        runtime.wait()
        assert value.read() == array.array("i", [1]).tobytes()
        value.write(array.array("i", [-1]).tobytes())
        prepared.run()
        with pytest.raises(aot.AotError, match="negative prepared input"):
            runtime.wait()
        value.write(bytes(4))
        prepared.run()
        assert value.read() == array.array("i", [1]).tobytes()
