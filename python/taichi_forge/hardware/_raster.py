"""Explicit Vulkan fixed-function raster passes for simulation rendering."""

from dataclasses import dataclass
from types import MappingProxyType

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import (
    BackendCommandPlan,
    BackendCommandRecording,
    NativeGraphExecutable,
    NativeGraphNode,
)
from taichi_forge.hardware._native_adapter import validate_exact_bindings
from taichi_forge._hardware_telemetry import instrument_hardware_recording
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


_RESOURCE_ARGUMENTS = {
    "lines": ("vertices", "indices", "per_vertex_color"),
    "mesh": ("vertices", "indices", "normals", "per_vertex_color"),
    "mesh_instance": (
        "vertices",
        "indices",
        "normals",
        "per_vertex_color",
        "transforms",
    ),
    "particles": (
        "centers",
        "per_vertex_color",
        "per_vertex_radius",
    ),
}

def _tuple_of_floats(value, size, name):
    value = tuple(float(component) for component in value)
    if len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    return value


@dataclass(frozen=True)
class _RasterDraw:
    kind: str
    arguments: object


@instrument_hardware_recording("raster.draw.vulkan")
class VulkanRasterPassRecording(BackendCommandRecording):
    """Immutable direct-execution recording for one offscreen raster pass.

    The semantic draw list is queued first, then one native offscreen-frame
    entry point records and submits the complete Vulkan graphics command list.
    Its Graph adapter is deliberately opaque and explicit-only: GGUI vertex
    preparation and provider-owned color/depth targets do not expose the
    bindings required for truthful automatic admission into one backend Graph.
    """

    def __init__(self, owner, draws, camera, ambient, point_lights):
        draws = tuple(draws)
        if not draws:
            raise ValueError("Vulkan raster recordings require at least one draw")

        fixed_bindings = {}
        resource_names = {}
        symbolic_draws = []
        for draw in draws:
            arguments = dict(draw.arguments)
            for argument_name in _RESOURCE_ARGUMENTS[draw.kind]:
                resource = arguments.get(argument_name)
                if resource is None:
                    continue
                identity = id(resource)
                binding_name = resource_names.get(identity)
                if binding_name is None:
                    binding_name = f"raster_resource_{len(resource_names)}"
                    resource_names[identity] = binding_name
                    fixed_bindings[binding_name] = resource
                arguments[argument_name] = ("binding", binding_name)
            symbolic_draws.append(_RasterDraw(draw.kind, MappingProxyType(arguments)))

        super().__init__(
            backend="vulkan",
            binding_names=tuple(fixed_bindings),
            command_count=len(draws) + 1,
            queue="graphics",
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_draws", tuple(symbolic_draws))
        object.__setattr__(self, "_camera", camera)
        object.__setattr__(self, "_ambient", ambient)
        object.__setattr__(self, "_point_lights", tuple(point_lights))
        object.__setattr__(self, "_fixed_bindings", MappingProxyType(fixed_bindings))

    @property
    def fixed_bindings(self):
        return self._fixed_bindings

    @property
    def resource_effects(self):
        return tuple(
            ResourceEffect(name, GraphAccess.READ) for name in self.binding_names
        )

    def execute(self, bindings=None):
        bindings = self._fixed_bindings if bindings is None else bindings
        validate_exact_bindings(self, bindings, "Vulkan raster")
        self.validate_graph_lifetime()
        return self._owner._execute_recording(
            self._draws,
            bindings,
            self._camera,
            self._ambient,
            self._point_lights,
        )

    def validate_graph_lifetime(self):
        self._owner._validate_lifetime()

    def memory_report(self):
        return self._owner.memory_report()

    def _as_graph_native_node(self):
        return _VulkanRasterPassNode(self)


class _VulkanRasterPassExecutable(NativeGraphExecutable):
    def __init__(self, recording):
        self._recording = recording

    def run(self, runtime_args=None):
        if runtime_args:
            raise TaichiRuntimeError(
                "Vulkan raster Graph execution has no public runtime bindings"
            )
        return self._recording.execute()

    @property
    def resource_effects(self):
        return self._recording.resource_effects

    @property
    def lifetime_leases(self):
        return (self._recording._owner,)

    @property
    def backend_command_plan(self):
        return BackendCommandPlan(
            backend="vulkan",
            helper_count=None,
            helper_count_exact=False,
            command_count=1,
            command_count_exact=False,
            provider_replay=False,
            no_host_readback=True,
            fragmentation_reason="ggui_helpers_and_hidden_attachments",
        )

    @property
    def debug_info(self):
        return {
            "kind": "vulkan_raster_pass",
            "draw_count": len(self._recording._draws),
            "graph_mode": "explicit_segmented",
        }


class _VulkanRasterPassNode(NativeGraphNode):
    def __init__(self, recording):
        self._recording = recording

    def compile(self):
        return _VulkanRasterPassExecutable(self._recording)


class RasterPass:
    """A reusable offscreen Vulkan hardware raster provider.

    Draw declarations are persistent until :meth:`clear`. :meth:`record`
    freezes them into an immutable recording; :meth:`execute` records and
    submits the current list. Color and depth readback are separate explicit
    host operations and are never performed by execution itself.
    """

    def __init__(
        self,
        resolution,
        *,
        background_color=(0.0, 0.0, 0.0),
        name="Taichi Forge Vulkan RasterPass",
    ):
        resolution = tuple(resolution)
        if (
            len(resolution) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in resolution
            )
            or any(value <= 0 for value in resolution)
        ):
            raise ValueError("RasterPass resolution must contain two positive integers")
        if impl.get_runtime().prog is None:
            raise TaichiRuntimeError(
                "RasterPass requires an initialized Taichi runtime"
            )
        if active_backend() != "vulkan":
            raise TaichiRuntimeError(
                "RasterPass requires the Vulkan backend; the active backend is "
                f"{active_backend()}"
            )

        from taichi_forge.ui.window import Window

        self._runtime_prog = impl.get_runtime().prog
        self._window = Window(name, resolution, show_window=False)
        self._canvas = self._window.get_canvas()
        self._scene = self._window.get_scene()
        self._background_color = _tuple_of_floats(
            background_color, 3, "background_color"
        )
        self._canvas.set_background_color(self._background_color)
        self._camera = None
        self._ambient = None
        self._point_lights = []
        self._draws = []
        self._frame_available = False

    @property
    def resolution(self):
        return tuple(self._window.get_window_shape())

    def set_camera(self, camera):
        self._validate_lifetime()
        if camera is None or getattr(camera, "ptr", None) is None:
            raise TypeError("RasterPass camera must be a ti.ui.Camera")
        self._camera = camera
        return self

    def ambient_light(self, color):
        self._validate_lifetime()
        self._ambient = _tuple_of_floats(color, 3, "ambient light color")
        return self

    def point_light(self, position, color):
        self._validate_lifetime()
        self._point_lights.append(
            (
                _tuple_of_floats(position, 3, "point light position"),
                _tuple_of_floats(color, 3, "point light color"),
            )
        )
        return self

    def _append_draw(self, kind, arguments):
        self._validate_lifetime()
        self._draws.append(_RasterDraw(kind, MappingProxyType(dict(arguments))))
        return self

    def mesh(self, vertices, **options):
        return self._append_draw("mesh", {"vertices": vertices, **options})

    def mesh_instance(self, vertices, **options):
        return self._append_draw("mesh_instance", {"vertices": vertices, **options})

    def particles(self, centers, radius, **options):
        return self._append_draw(
            "particles", {"centers": centers, "radius": radius, **options}
        )

    def lines(self, vertices, width, **options):
        return self._append_draw(
            "lines", {"vertices": vertices, "width": width, **options}
        )

    def clear(self):
        self._validate_lifetime()
        self._draws.clear()
        self._point_lights.clear()
        return self

    def record(self):
        self._validate_lifetime()
        if self._camera is None:
            raise ValueError("RasterPass requires set_camera() before recording")
        return VulkanRasterPassRecording(
            self,
            self._draws,
            self._camera,
            self._ambient,
            self._point_lights,
        )

    def execute(self):
        return self.record().execute()

    def _execute_recording(self, draws, bindings, camera, ambient, point_lights):
        self._validate_lifetime()
        self._window.window._begin_offscreen_frame()
        self._scene.set_camera(camera)
        if ambient is not None:
            self._scene.ambient_light(ambient)
        for position, color in point_lights:
            self._scene.point_light(position, color)
        for draw in draws:
            arguments = dict(draw.arguments)
            for argument_name in _RESOURCE_ARGUMENTS[draw.kind]:
                value = arguments.get(argument_name)
                if (
                    isinstance(value, tuple)
                    and len(value) == 2
                    and value[0] == "binding"
                ):
                    arguments[argument_name] = bindings[value[1]]
            getattr(self._scene, draw.kind)(**arguments)
        self._canvas.scene(self._scene)
        if not self._window.window._render_offscreen_frame():
            raise TaichiRuntimeError("Vulkan RasterPass submission was rejected")
        self._frame_available = True

    def color_numpy(self):
        self._validate_lifetime()
        if not self._frame_available:
            raise TaichiRuntimeError(
                "RasterPass color readback requires a new execute()"
            )
        try:
            return self._window.get_image_buffer_as_numpy()
        finally:
            self._frame_available = False

    def depth_numpy(self):
        self._validate_lifetime()
        if not self._frame_available:
            raise TaichiRuntimeError(
                "RasterPass depth readback requires a new execute()"
            )
        try:
            return self._window.get_depth_buffer_as_numpy()
        finally:
            self._frame_available = False

    def _validate_lifetime(self):
        if self._window is None or self._window.window is None:
            raise TaichiRuntimeError("RasterPass has been destroyed")
        if impl.get_runtime().prog is not self._runtime_prog:
            raise TaichiRuntimeError(
                "RasterPass belongs to a previous Taichi runtime generation"
            )

    def memory_report(self):
        """Report lifecycle while keeping hidden GGUI/driver bytes opaque."""

        window_present = self._window is not None and self._window.window is not None
        runtime_valid = (
            window_present and impl.get_runtime().prog is self._runtime_prog
        )
        return make_memory_report(
            "vulkan_raster_pass",
            "vulkan",
            (
                HardwareMemoryComponent(
                    "ggui_window_attachments_and_pipeline",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=runtime_valid,
                ),
            ),
            lifecycle_state=(
                "ready"
                if runtime_valid
                else "closed"
                if not window_present
                else "runtime_invalid"
            ),
            ownership_scope="raster_pass_generation",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def destroy(self):
        if self._window is None:
            return None
        window = self._window
        self._window = None
        self._frame_available = False
        return window.destroy()

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.destroy()
        return False


__all__ = ["RasterPass", "VulkanRasterPassRecording"]
