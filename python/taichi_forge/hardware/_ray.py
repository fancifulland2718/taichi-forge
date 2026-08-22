"""Explicit Vulkan acceleration-structure and batch ray-query provider."""

from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect, RuntimeBinding
from taichi_forge.graph._native import (
    BackendCommandGraphAction,
    BackendCommandRecording,
    NativeGraphExecutable,
    NativeGraphNode,
)
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32, i32


def _active_backend():
    arch = _ti_core.arch_name(impl.current_cfg().arch)
    return "cpu" if arch in ("x64", "arm64") else arch


def _item_count(value, width, dtype, name):
    if not isinstance(value, Ndarray):
        raise TaichiRuntimeError(f"Vulkan ray {name} must be a Taichi ndarray")
    shape = tuple(value.shape)
    element_shape = tuple(value.element_shape)
    if value.dtype != dtype:
        raise TaichiRuntimeError(f"Vulkan ray {name} must use dtype {dtype}")
    if element_shape == () and len(shape) == 2 and shape[1] == width:
        count = shape[0]
    elif element_shape == (width,) and len(shape) == 1:
        count = shape[0]
    else:
        raise TaichiRuntimeError(
            f"Vulkan ray {name} must have scalar shape (N, {width}) or "
            f"AOS vector-{width} shape (N,)"
        )
    if count <= 0:
        raise TaichiRuntimeError(f"Vulkan ray {name} must not be empty")
    return count


class VulkanRayQueryRecording(BackendCommandRecording):
    """One batch query against a fixed :class:`TriangleScene` generation."""

    def __init__(self, scene, ray_count, *, rays="rays", hits="hits"):
        if not isinstance(scene, TriangleScene):
            raise TypeError("Vulkan ray query recording requires a TriangleScene")
        if (
            isinstance(ray_count, bool)
            or not isinstance(ray_count, int)
            or ray_count <= 0
            or ray_count > 0xFFFFFFFF
        ):
            raise ValueError("Vulkan ray query count must be in [1, UINT32_MAX]")
        if any(not isinstance(name, str) or not name for name in (rays, hits)):
            raise ValueError("Vulkan ray query binding names must be nonempty strings")
        if rays == hits:
            raise ValueError("Vulkan ray query binding names must be unique")
        super().__init__(
            backend="vulkan",
            binding_names=(rays, hits),
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "scene", scene)
        object.__setattr__(self, "ray_count", ray_count)
        object.__setattr__(self, "rays", rays)
        object.__setattr__(self, "hits", hits)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.rays, GraphAccess.READ),
            ResourceEffect(self.hits, GraphAccess.WRITE),
        )

    def execute(self, bindings):
        required = frozenset(self.binding_names)
        provided = frozenset(bindings)
        if provided != required:
            missing = sorted(required.difference(provided))
            unexpected = sorted(provided.difference(required))
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise TaichiRuntimeError(
                "Vulkan ray query bindings do not match the recording: "
                + "; ".join(details)
            )
        self.validate_graph_lifetime()
        rays = bindings[self.rays]
        hits = bindings[self.hits]
        if _item_count(rays, 8, f32, self.rays) != self.ray_count:
            raise TaichiRuntimeError(
                f"Vulkan ray binding {self.rays!r} has the wrong ray count"
            )
        if _item_count(hits, 4, f32, self.hits) != self.ray_count:
            raise TaichiRuntimeError(
                f"Vulkan ray binding {self.hits!r} has the wrong ray count"
            )
        self.scene._execute_query(rays, hits, self.ray_count)

    def validate_graph_lifetime(self):
        self.scene._validate_lifetime()

    def _as_graph_native_node(self):
        return _VulkanRayQueryNode(self)


class _VulkanRayQueryExecutable(NativeGraphExecutable):
    def __init__(self, recording):
        self._recording = recording
        self._action = BackendCommandGraphAction(recording)

    def run(self, runtime_args):
        return self._recording.execute(runtime_args)

    @property
    def runtime_arg_schema(self):
        return tuple(
            RuntimeBinding(name, "ndarray") for name in self._recording.binding_names
        )

    @property
    def resource_effects(self):
        return self._recording.resource_effects

    @property
    def lifetime_leases(self):
        return (self._recording.scene,)

    @property
    def recordable_action(self):
        return self._action

    @property
    def debug_info(self):
        return {
            "kind": "vulkan_triangle_ray_query",
            "ray_count": self._recording.ray_count,
            "scene_kind": "static_triangle_blas_tlas",
        }


class _VulkanRayQueryNode(NativeGraphNode):
    def __init__(self, recording):
        self._recording = recording

    def compile(self):
        return _VulkanRayQueryExecutable(self._recording)


class TriangleScene:
    """One immutable triangle BLAS and one identity-instance TLAS.

    ``vertices`` and ``indices`` accept scalar ``(N, 3)`` ndarrays or AOS
    vector-3 ndarrays with shape ``(N,)``. Indices are signed i32 for parity
    with Forge mesh storage but must all be nonnegative and in range; this
    low-level provider does not perform a host readback to validate them.
    """

    def __init__(self, vertices, indices):
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError(
                "TriangleScene requires an initialized Taichi runtime"
            )
        if _active_backend() != "vulkan":
            raise TaichiRuntimeError(
                "TriangleScene requires the Vulkan backend; the active backend is "
                f"{_active_backend()}"
            )
        if not program.vulkan_ray_query_available():
            raise TaichiRuntimeError(
                "TriangleScene requires VK_KHR_acceleration_structure and "
                "VK_KHR_ray_query"
            )
        vertex_count = _item_count(vertices, 3, f32, "vertices")
        triangle_count = _item_count(indices, 3, i32, "indices")
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._handle = int(
            program._create_vulkan_triangle_ray_scene(
                vertices.arr,
                indices.arr,
                vertex_count,
                triangle_count,
            )
        )
        self.vertex_count = vertex_count
        self.triangle_count = triangle_count

    @property
    def closed(self):
        return self._handle is None

    def record(self, ray_count, *, rays="rays", hits="hits"):
        self._validate_lifetime()
        return VulkanRayQueryRecording(
            self, ray_count, rays=rays, hits=hits
        )

    def trace(self, rays, hits):
        ray_count = _item_count(rays, 8, f32, "rays")
        recording = self.record(ray_count)
        recording.execute({"rays": rays, "hits": hits})
        return hits

    def _execute_query(self, rays, hits, ray_count):
        self._validate_lifetime()
        self._runtime_prog._vulkan_triangle_ray_query(
            self._handle, rays.arr, hits.arr, ray_count
        )

    def _validate_lifetime(self):
        if self._handle is None:
            raise TaichiRuntimeError("TriangleScene has been closed")
        if (
            impl.get_runtime().prog is not self._runtime_prog
            or int(impl.runtime_generation()) != self._runtime_generation
        ):
            raise TaichiRuntimeError(
                "TriangleScene belongs to a previous Taichi runtime generation"
            )

    def validate_graph_lifetime(self):
        self._validate_lifetime()

    def close(self):
        if self._handle is None:
            return None
        handle = self._handle
        self._handle = None
        if (
            impl.get_runtime().prog is self._runtime_prog
            and int(impl.runtime_generation()) == self._runtime_generation
        ):
            self._runtime_prog._destroy_vulkan_triangle_ray_scene(handle)
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def is_available():
    """Return whether the active runtime supports the complete Vulkan slice."""

    program = impl.get_runtime().prog
    return bool(
        program is not None
        and _active_backend() == "vulkan"
        and program.vulkan_ray_query_available()
    )


__all__ = ["TriangleScene", "VulkanRayQueryRecording", "is_available"]
