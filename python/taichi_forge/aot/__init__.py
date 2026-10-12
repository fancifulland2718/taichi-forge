"""Taichi's AOT (ahead of time) module.

Users can use Taichi as a GPU compute shader/kernel compiler by compiling their
Taichi kernels into an AOT module.
"""

import taichi_forge.aot.conventions
from taichi_forge.aot._export import export, export_as
from taichi_forge.aot.conventions.gfxruntime140 import GfxRuntime140
from taichi_forge.aot.module import Module


def borrow_runtime():
    """Load AOT code on the current CPU/CUDA/Vulkan Program's backend.

    Requires a matching split runtime with the C API. The returned consumer
    owns its AOT resources; closing it leaves the JIT Program alive. Resetting
    the Program retires all its borrowers. JIT ndarray imports are separate
    from this owner bridge and are not accepted as raw pointer bindings.
    """
    from taichi_forge._lib import core, utils
    from taichi_forge.lang import impl
    from taichi_forge_runtime.aot import Runtime

    factory = getattr(core, "_borrow_aot_runtime", None)
    if factory is None or utils._loaded_native_runtime_path is None:
        raise RuntimeError("AOT borrowing requires a split runtime built with the C API")
    runtime = impl.get_runtime()
    runtime.materialize()
    client = factory(runtime.prog)
    result = Runtime._from_borrowed(client, utils._loaded_native_runtime_path)
    runtime.register_runtime_object(result)
    return result


def __getattr__(name):
    if name == "ExecutionPlan":
        from taichi_forge_runtime.aot import ExecutionPlan

        return ExecutionPlan
    raise AttributeError(name)
