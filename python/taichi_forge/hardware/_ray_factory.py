"""Cold backend selection and ownership for the existing triangle batch APIs."""

from taichi_forge.hardware._runtime import active_backend
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


class NativeTriangleScene:
    """Owner returned by :func:`triangle_scene`, with the existing batch methods.

    ``native_scene`` is the actual backend object; it is not a copy. ``record``,
    ``record_typed``, ``record_refit``, ``trace``, ``trace_typed``, ``refit`` and
    ``memory_report`` are its bound methods, with no per-call backend selection.
    ``refit`` returns that native object. Close this owner after its Graph users.
    A caller-supplied OptixProvider remains caller-owned; a factory-created one
    is closed with the scene. Runtime reset still uses the existing native owners.
    """

    def __init__(self, scene, backend, owned_provider=None):
        self._scene = scene
        self._owned_provider = owned_provider
        self._backend = backend
        for name in ("record", "record_typed", "record_refit", "trace", "trace_typed", "refit", "memory_report"):
            setattr(self, name, getattr(scene, name))

    @property
    def native_scene(self):
        return self._scene

    @property
    def backend(self):
        return self._backend

    @property
    def closed(self):
        return self._scene.closed and (self._owned_provider is None or self._owned_provider.closed)

    def close(self):
        self._scene.close()
        if self._owned_provider is not None:
            self._owned_provider.close()
            self._owned_provider = None

    def __enter__(self):
        self._scene._validate_lifetime()
        return self

    def __exit__(self, *args):
        self.close()


def triangle_scene(vertices, indices, *, backend="auto", provider=None, library_path=None, provider_path=None):
    """Create an updatable native triangle scene on the current runtime backend.

    Vulkan uses AS/ray-query; CUDA lazily loads the compatible OptiX adapter and
    user/system vendor runtime. ``backend`` may be auto, vulkan or cuda and never
    switches the runtime or moves resources between devices. Missing hardware or
    dependencies raise at creation; this API has no software-renderer fallback.

    Geometry is compact f32 (N,3) vertices and i32 (M,3) indices, or equivalent
    AOS vector-3 storage. Ray and hit layouts are those of the existing scene APIs.
    ``library_path`` names the OptiX vendor runtime; ``provider_path`` names a Forge
    adapter, not a CUDA Toolkit. Both are CUDA-only and cannot accompany an already
    constructed provider. Typed hits require a compatible adapter when requested.
    This factory does not enable retained execution, OMM or alternate precision.
    """
    if backend not in ("auto", "vulkan", "cuda"):
        raise ValueError("backend must be auto, vulkan or cuda")
    current = active_backend()
    if impl.get_runtime().prog is None:
        raise TaichiRuntimeError("triangle_scene requires an initialized runtime")
    if backend != "auto" and backend != current:
        raise TaichiRuntimeError("triangle_scene cannot switch the active runtime backend")
    if current == "vulkan":
        if provider is not None or library_path is not None or provider_path is not None:
            raise ValueError("OptiX provider and library paths are only valid on CUDA")
        from taichi_forge.hardware._ray import TriangleScene

        return NativeTriangleScene(TriangleScene(vertices, indices), current)
    if current != "cuda":
        raise TaichiRuntimeError(f"triangle_scene has no native batch route for backend {current}")
    from taichi_forge.hardware._optix import OptixProvider

    if provider is not None:
        if not isinstance(provider, OptixProvider):
            raise TypeError("provider must be an OptixProvider")
        if library_path is not None or provider_path is not None:
            raise ValueError("paths cannot be combined with a caller-owned provider")
        return NativeTriangleScene(provider.triangle_scene(vertices, indices), current)
    owned = OptixProvider(library_path=library_path, provider_path=provider_path)
    try:
        scene = owned.triangle_scene(vertices, indices)
    except BaseException:
        owned.close()
        raise
    return NativeTriangleScene(scene, current, owned)
