"""Public aggregate values share ordinary/prepared storage and resource owners."""

import json
import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
import numpy as np
import pytest
from contextlib import ExitStack, closing
from taichi_forge._lib import utils
from tests import test_utils
from tests.cpp.aot.python_scripts.value_argument_aot import export


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("fold_dimensions", [False, True])
def test_prepared_bitcode_specialization_keeps_shapes_and_instances_independent(tmp_path, fold_dimensions):
    from taichi_forge_runtime.aot import Scalar

    @ti.real_func
    def helper(value: ti.i32) -> ti.i32:
        return 2 * value + 1

    @ti.kernel
    def update(x: ti.types.ndarray(ti.types.vector(2, ti.i32), 2), n: ti.i32, gain: ti.f32):
        for i, j in x:
            if j < n:
                x[i, j][0] = helper(x[i, j][0]) + n
                x[i, j][1] += ti.cast(gain, ti.i32)

    module = ti.aot.Module()
    module.add_kernel(update)
    artifact = tmp_path / "specialized.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, ExitStack() as stack:
        loaded = stack.enter_context(runtime.load_module(artifact))
        entry = loaded.kernel("update")
        records = []
        # Same shape shares code, not the pointer or contents. Other shapes and
        # explicit scalar constants select distinct, immutable preparations.
        for shape, n, initial, gain in [((3, 7), 5, 1, 3), ((3, 7), 5, 11, 7),
                                       ((4, 9), 8, 21, -2), ((3, 7), 2, 31, 9)]:
            array = stack.enter_context(closing(runtime.ndarray("i32", shape, element_shape=(2,))))
            array.write(np.full((*shape, 2), initial, dtype=np.int32))
            bound = entry.bind(array, Scalar("i32", n), Scalar("f32", float(gain)))
            ready = stack.enter_context(closing(bound.prepare(
                specialize=True, constant_args=(1,) if fold_dimensions else ())))
            records.append((array, ready, shape, n, initial, gain))
        for array, ready, shape, n, initial, gain in reversed(records):
            ready.run()
            expected = np.full((*shape, 2), initial, dtype=np.int32)
            expected[:, :n] += [initial + n + 1, gain]
            np.testing.assert_array_equal(np.frombuffer(array.read(), np.int32).reshape(*shape, 2), expected)
            array.write(np.zeros((*shape, 2), dtype=np.int32))
            ready.run()
            expected.fill(0)
            expected[:, :n] += [n + 1, gain]
            np.testing.assert_array_equal(np.frombuffer(array.read(), np.int32).reshape(*shape, 2), expected)
        with pytest.raises(Exception, match="integer scalar"):
            bound.prepare(specialize=True, constant_args=(2,))
        with pytest.raises(ValueError, match="requires specialize"):
            bound.prepare(constant_args=(1,))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_value_arguments_forward_adjoint_and_prepared_retirement(tmp_path):
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    seconds = export(arch, tmp_path)
    print(json.dumps(dict(arch=arch, export_seconds=seconds,
                          artifact_bytes=(tmp_path / "values.tcm").stat().st_size)))
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join([str(root / "packaging/runtime/src"), *sys.path]),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        TI_ENABLE_VERSION_CHECK="0",
    )
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(__file__).with_name("aot_value_argument_probe.py")),
             mode, arch, str(tmp_path / "values.tcm")],
            env=environment, capture_output=True, text=True, timeout=120,
        )
        output = result.stdout + result.stderr
        print(output)
        assert result.returncode == 0, output
        assert "VALUE_ARGUMENTS_PASSED" in output
        assert not any(item in output for item in ("VUID-", "Validation Error", "SYNC-HAZARD")), output
