"""Identical value layouts share a Plan input despite neighbour alignment."""
import ctypes as C
import struct

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime, ExecutionPlan, AotError
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_plan_value_ignores_only_trailing_argument_padding(tmp_path):
    vec3 = ti.types.vector(3, ti.f32)
    ints = ti.types.vector(3, ti.i32)

    @ti.kernel
    def narrow(v: vec3, marker: ti.f32, output: ti.types.ndarray(ti.f32, 1)):
        output[0] = v.sum() + marker

    @ti.kernel
    def padded(v: vec3, output: ti.types.ndarray(ti.f32, 1)):
        output[1] = v.sum()

    @ti.kernel
    def incompatible(v: ints, output: ti.types.ndarray(ti.f32, 1)):
        output[1] = ti.cast(v.sum(), ti.f32)

    backend = 'cpu' if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    module = ti.aot.Module()
    for kernel in (narrow, padded, incompatible):
        module.add_kernel(kernel)
    artifact = tmp_path/'padding.tcm'
    module.archive(str(artifact))
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        output = runtime.ndarray('f32', 2)
        operations = [dict(kind='kernel', entry='narrow', arguments=['v', 'marker', 'out']),
                      dict(kind='kernel', entry='padded', arguments=['v', 'out'])]
        plan = ExecutionPlan(operations)
        value = plan.value(loaded, 'v', [2., 3., 7.])
        with plan.prepare(loaded, dict(v=value, marker=5., out=output)) as call:
            call.run()
        assert struct.unpack('<2f', output.read()) == (17., 12.)
        invalid = ExecutionPlan([operations[0], dict(operations[1], entry='incompatible')])
        with pytest.raises(ValueError, match='incompatible native ABI'):
            invalid.value(loaded, 'v', [2., 3., 7.])
        # Keep the old C window contract while rejecting truncated/noncanonical
        # spans. LLVM aligns the following ndarray to eight bytes: 12 -> 16.
        if backend in ('cpu', 'cuda'):
            from taichi_forge_runtime.aot import _c_api as abi

            entry = loaded.kernel('padded')
            bound = entry.bind([2., 3., 7.], output)
            assert bound._arguments[0].value.value.size == 12
            arguments = (abi.Argument * 2)(*bound._arguments)
            buffer = C.create_string_buffer(struct.pack('<3fI', 2., 3., 7., 0), 16)
            arguments[0].value.value = abi.ValueArgument(16, C.cast(buffer, C.c_void_p))
            runtime._api.call(runtime._api.ti_launch_kernel_with_bindings_ext,
                runtime._handle, entry._handle, 2, arguments, 0, None)
            assert struct.unpack('<2f', output.read()) == (17., 12.)
            for size in (11, 13, 17):
                arguments[0].value.value.size = size
                with pytest.raises(AotError, match='compiled argument size'):
                    runtime._api.call(runtime._api.ti_prepare_kernel_ext,
                        runtime._handle, entry._handle, 2, arguments, 0, None)
