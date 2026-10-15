"""Exercise physical imported aliases, cold failures and actual reverse kernels."""

import ctypes as C
import sys

from taichi_forge_runtime.aot import AotError, Runtime
from taichi_forge_runtime.aot._runtime import NdArray, _Memory
from tests.python.aot_graph_recording_probe import read, write
from tests.python.aot_cutensor_probe import near


def failure(operation, message=None):
    try:
        operation()
    except (AotError, ValueError, RuntimeError) as error:
        if message is not None:
            assert message in str(error), str(error)
        return
    raise AssertionError("Expected a cold binding error")


class PointerInfo(C.Structure):
    _fields_ = [("ptr", C.c_void_p), ("size", C.c_uint64)]


class VulkanInfo(C.Structure):
    _fields_ = [
        ("buffer", C.c_void_p),
        ("size", C.c_uint64),
        ("usage", C.c_uint32),
        ("memory", C.c_void_p),
        ("offset", C.c_uint64),
    ]


def imported(runtime, source, arch, *, offset=0, size=None, known_memory=True):
    """Import the real exported storage under a distinct native allocation ID."""
    size = source.nbytes if size is None else size
    api = runtime._api
    info_type = VulkanInfo if arch == "vulkan" else PointerInfo
    info = info_type()
    export = getattr(api.library, f"ti_export_{arch}_memory")
    export.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(info_type)]
    export.restype = None
    api.call(export, runtime._handle, source._memory._handle, C.byref(info))
    load = getattr(api.library, f"ti_import_{arch}_memory")
    load.restype = C.c_void_p
    if arch == "vulkan":
        assert offset == 0  # The public ABI exposes whole VkBuffers, not subviews.
        load.argtypes = [C.c_void_p, C.POINTER(info_type)]
        info.size = size
        if not known_memory:
            info.memory = None
        handle = api.call(load, runtime._handle, C.byref(info))
    else:
        load.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
        handle = api.call(load, runtime._handle, info.ptr + offset, size)
    memory = _Memory._from_import(runtime, handle, size)
    return NdArray(memory, "f32", source._code, (size // 4,), ())


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        module = runtime.load_module(artifact)
        typed = runtime.ndarray("f32", 17)
        same_width = runtime.ndarray("i32", 17)
        failure(lambda: module.prepare_execution("typed_fill", {"x": same_width}), "dtype")
        with module.prepare_execution("typed_fill", {"x": typed}) as fill:
            fill()
            near(read(typed), [1.0] * 17)
        typed.close()
        same_width.close()
        frames = [{name: runtime.ndarray("f32", (17,)) for name in ("x", "y", "dx")} for _ in range(2)]
        calls = [module.prepare_execution("differentiate", frame) for frame in frames]
        for step in range(3):
            for index, (frame, call) in enumerate(zip(frames, calls)):
                values = [(i - 8 + step + index) / 8 for i in range(17)]
                write(frame["x"], values)
                call()
                near(read(frame["y"]), [v * v + 2 * v for v in values])
                near(read(frame["dx"]), [2 * v + 2 for v in values])

        frame = frames[0]
        write(frame["y"], [99] * 17)
        short = runtime.ndarray("f32", (16,))
        failure(lambda: module.prepare_execution("differentiate", dict(frame, x=short)), "exact shape")
        long = runtime.ndarray("f32", (18,))
        failure(lambda: module.prepare_execution("differentiate", dict(frame, x=long)), "exact shape")
        long.close()
        failure(lambda: module.prepare_execution("differentiate", dict(frame, y=frame["x"])), "disjoint")
        alias = imported(runtime, frame["x"], arch)
        assert alias._memory._handle != frame["x"]._memory._handle
        failure(lambda: module.prepare_execution("differentiate", dict(frame, y=alias)), "disjoint")
        alias.close()
        assert read(frame["y"]) == [99] * 17, "Failed preparation must execute no kernels"

        # Identity is physical storage plus typed shape, not allocation handles.
        alias = imported(runtime, frame["x"], arch, known_memory=arch != "vulkan")
        for output in (frame["x"], alias, frame["y"]):
            write(frame["x"], [2] * 17)
            with module.prepare_execution("same_or_disjoint", dict(x=frame["x"], y=output)) as in_place:
                in_place()
                assert read(output) == [7] * 17
                in_place()
                assert read(output) == ([17] if output is not frame["y"] else [7]) * 17
        reshaped = NdArray(alias._memory, "f32", alias._code, (1, 17), ())
        failure(lambda: module.prepare_execution("alias_only", dict(x=frame["x"], y=reshaped)), "same view")
        tensor_view = NdArray(alias._memory, "f32", alias._code, (1,), (17,))
        failure(lambda: module.prepare_execution("alias_only", dict(x=frame["x"], y=tensor_view)), "same view")
        integer = runtime.ndarray("i32", (17,))
        retyped = NdArray(alias._memory, "i32", integer._code, (17,), ())
        failure(lambda: module.prepare_execution("alias_only", dict(x=frame["x"], y=retyped)), "same view")
        integer.close()
        alias.close()

        if arch == "vulkan":
            # Missing VkDeviceMemory identity is not evidence of disjointness.
            unknown = imported(runtime, frame["y"], arch, known_memory=False)
            failure(lambda: module.prepare_execution("differentiate", dict(frame, y=unknown)), "disjoint")
            unknown.close()
        else:
            parent = runtime.ndarray("f32", (34,))
            left = imported(runtime, parent, arch, size=68)
            overlapping = imported(runtime, parent, arch, offset=4, size=68)
            right = imported(runtime, parent, arch, offset=68, size=68)
            failure(lambda: module.prepare_execution("aligned", dict(x=overlapping, y=frame["y"])), "alignment")
            failure(lambda: module.prepare_execution("differentiate", dict(frame, x=left, y=overlapping)), "disjoint")
            failure(lambda: module.prepare_execution("same_or_disjoint", dict(x=left, y=overlapping)), "same view")
            with module.prepare_execution("same_or_disjoint", dict(x=left, y=right)) as adjacent_alias:
                write(left, [2] * 17)
                adjacent_alias()
                assert read(right) == [7] * 17
            adjacent = module.prepare_execution("differentiate", dict(frame, x=left, y=right))
            write(left, [2] * 17)
            adjacent()
            assert read(right) == [8] * 17
            assert read(frame["dx"]) == [6] * 17
            adjacent.close()
            left.close()
            overlapping.close()
            right.close()
            parent.close()

        # A genuine tensor shape is required, even when the byte count matches.
        tensor = {name: runtime.ndarray("f32", (3,), element_shape=(2, 2)) for name in ("x", "y")}
        write(tensor["x"], list(range(12)))
        tensor_call = module.prepare_execution("tensor", tensor)
        tensor_call()
        assert read(tensor["y"]) == list(range(12))
        flat = runtime.ndarray("f32", (12,))
        failure(lambda: module.prepare_execution("tensor", dict(tensor, x=flat)), "required shape")
        failure(lambda: module.prepare_execution("wrong_record", tensor), "record size")
        empty = runtime.ndarray("f32", (0,))
        with module.prepare_execution("empty", dict(x=empty, y=empty)) as empty_call:
            empty_call()
            runtime.wait()
        empty.close()
        with Runtime(arch) as other:
            foreign = other.ndarray("f32", (17,))
            failure(lambda: module.prepare_execution("differentiate", dict(frame, x=foreign)), "different AOT runtime")
        short.close()
        failure(lambda: module.prepare_execution("differentiate", dict(frame, x=short)), "closed")

        # Cold failures leave existing prepared owners usable and independently retired.
        write(frame["x"], [3] * 17)
        calls[0]()
        assert read(frame["y"]) == [15] * 17
        calls[0].close()
        calls[1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        failure(calls[1])
        calls[1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
