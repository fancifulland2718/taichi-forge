"""Taichi's AOT (ahead of time) module.

Users can use Taichi as a GPU compute shader/kernel compiler by compiling their
Taichi kernels into an AOT module.
"""

import taichi_forge.aot.conventions
from taichi_forge.aot._export import export, export_as
from taichi_forge.aot.conventions.gfxruntime140 import GfxRuntime140
from taichi_forge.aot.module import Module


# Load capture support only when requested. Ordinary JIT/Module imports do not
# need the declaration compiler or variant builder, and application producers
# should not depend on their private implementation module paths.
_component_exports = {
    "Component": "._component",
    "StaticValue": "._component",
    "RootOption": "._component",
    "BuildBudget": "._component_variants",
    "ComponentBuildError": "._component_variants",
    "Variant": "._component_variants",
    "build_variants": "._component_variants",
}


def borrow_runtime():
    """Load AOT code on the current CPU/CUDA/Vulkan Program's backend.

    Requires a matching split runtime with the C API. The returned consumer
    owns its AOT resources; closing it leaves the JIT Program alive. Resetting
    the Program retires all its borrowers. ``runtime.import_ndarray(array)``
    retains an explicit zero-copy storage lease on an existing JIT ndarray.
    CUDA/Vulkan ``runtime.import_texture(texture)`` retains a sampled Texture's
    existing storage and sampler through the same owner boundary.
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
    result = Runtime._from_borrowed(
        client, utils._loaded_native_runtime_path,
        ndarray_importer=_import_ndarray, texture_importer=_import_texture,
        dense_view_importer=_import_dense_view,
        acceleration_structure_importer=_import_acceleration_structure,
    )
    # The frontend Graph bridge must distinguish Programs, not just devices or
    # backend names. The native client still owns retirement and storage checks.
    result._frontend_program = runtime.prog
    runtime.register_runtime_object(result)
    return result


def _import_ndarray(client, array):
    # Keep frontend type knowledge on the producer side. The standalone AOT
    # consumer has no dependency on the DSL or on numpy.
    from taichi_forge.lang._ndarray import Ndarray, StructNdarray

    if not isinstance(array, Ndarray) or isinstance(array, StructNdarray):
        raise TypeError("AOT import requires a scalar or tensor JIT ndarray")
    if array.arr is None:
        raise RuntimeError("Cannot import an ndarray after its runtime was reset")
    return client.import_ndarray(array.arr)


def _import_dense_view(client, view):
    from taichi_forge.lang import impl
    from taichi_forge.lang._storage_view import DenseNdarrayView

    if not isinstance(view, DenseNdarrayView):
        raise TypeError("AOT dense import requires an explicit ndarray_view")
    if view._runtime_prog is not impl.get_runtime().prog:
        raise RuntimeError("Cannot import a dense view from a retired Program")
    return client.import_dense_view(view.descriptor)


def _import_texture(client, texture):
    from taichi_forge.lang._texture import Texture

    if not isinstance(texture, Texture):
        raise TypeError("AOT import requires a JIT Texture")
    if texture.tex is None:
        raise RuntimeError("Cannot import a Texture after its runtime was reset")
    return client.import_texture(texture.tex)


def _import_acceleration_structure(client, resource):
    from taichi_forge.hardware._ray import TriangleScene, TriangleBLAS, InstanceTLAS

    if type(resource) not in (TriangleScene, TriangleBLAS, InstanceTLAS):
        raise TypeError("AOT AS import requires a qualified Vulkan triangle resource")
    resource._validate_lifetime()
    return client.import_acceleration_structure(
        resource._runtime_prog, resource._handle, type(resource) is TriangleScene
    )


def __getattr__(name):
    if name == "ExecutionPlan":
        from taichi_forge_runtime.aot import ExecutionPlan

        return ExecutionPlan
    if name in _component_exports:
        from importlib import import_module

        value = getattr(import_module(_component_exports[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()) | set(_component_exports) | {"ExecutionPlan"})
