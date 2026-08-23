"""Low-level runtime-ordered Vulkan image transfer commands."""

from dataclasses import dataclass

from taichi_forge._hardware_telemetry import instrument_hardware_recording
from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    validate_exact_bindings,
    validate_runtime_generation,
)
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.exception import TaichiRuntimeError


def _active_backend():
    arch = _ti_core.arch_name(impl.current_cfg().arch)
    return "cpu" if arch in ("x64", "arm64") else arch


def _index(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= 0x7FFFFFFF:
        raise ValueError(f"{name} must be in [{minimum}, INT_MAX]")
    return value


def _coordinates(value, name, *, positive):
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a coordinate sequence") from exc
    if not 1 <= len(result) <= 3:
        raise ValueError(f"{name} must have one, two, or three components")
    minimum = 1 if positive else 0
    result = tuple(_index(item, name, minimum=minimum) for item in result)
    padding = (1 if positive else 0,) * (3 - len(result))
    return (*result, *padding)


def _texture_extent(texture):
    return (*tuple(texture.shape), *(1,) * (3 - len(texture.shape)))


@dataclass(frozen=True)
class VulkanImageRegion:
    """One color-image subresource region.

    Current :class:`Texture` resources contain one mip and one layer. The mip
    and layer fields are explicit so unsupported requests fail locally rather
    than being silently treated as the base subresource.
    """

    offset: tuple = (0, 0, 0)
    extent: object = None
    mip_level: int = 0
    base_layer: int = 0
    layer_count: int = 1

    def __post_init__(self):
        object.__setattr__(
            self, "offset", _coordinates(self.offset, "image offset", positive=False)
        )
        if self.extent is not None:
            object.__setattr__(
                self,
                "extent",
                _coordinates(self.extent, "image extent", positive=True),
            )
        object.__setattr__(self, "mip_level", _index(self.mip_level, "image mip_level"))
        object.__setattr__(
            self, "base_layer", _index(self.base_layer, "image base_layer")
        )
        object.__setattr__(
            self,
            "layer_count",
            _index(self.layer_count, "image layer_count", minimum=1),
        )

    def resolved_extent(self, texture):
        if self.extent is not None:
            return self.extent
        size = _texture_extent(texture)
        extent = tuple(size[axis] - self.offset[axis] for axis in range(3))
        if any(value <= 0 for value in extent):
            raise TaichiRuntimeError("Vulkan image region offset exceeds its texture")
        return extent

    def effect_scope(self, extent=None):
        return (
            "image",
            self.mip_level,
            self.base_layer,
            self.layer_count,
            self.offset,
            self.extent if extent is None else extent,
        )


@dataclass(frozen=True)
class VulkanBufferImageLayout:
    """Raw-buffer byte offset and Vulkan texel row/image pitches.

    Offset alignment depends on the bound image format and is therefore
    validated when the recording executes.
    """

    byte_offset: int = 0
    row_length: int = 0
    image_height: int = 0

    def __post_init__(self):
        object.__setattr__(
            self, "byte_offset", _index(self.byte_offset, "buffer byte_offset")
        )
        object.__setattr__(
            self, "row_length", _index(self.row_length, "buffer row_length")
        )
        object.__setattr__(
            self,
            "image_height",
            _index(self.image_height, "buffer image_height"),
        )
    def effect_scope(self):
        return (
            "buffer_image_layout",
            self.byte_offset,
            self.row_length,
            self.image_height,
        )


def _region(value, name):
    if value is None:
        return VulkanImageRegion()
    if not isinstance(value, VulkanImageRegion):
        raise TypeError(f"{name} must be a VulkanImageRegion")
    return value


def _buffer_layout(value):
    if value is None:
        return VulkanBufferImageLayout()
    if not isinstance(value, VulkanBufferImageLayout):
        raise TypeError("buffer_layout must be a VulkanBufferImageLayout")
    return value


@instrument_hardware_recording("image.copy.vulkan")
class _VulkanImageTransferRecording(BackendCommandRecording):
    def __init__(self, binding_kinds, *, queue="compute"):
        binding_kinds = tuple(binding_kinds)
        names = tuple(name for name, _ in binding_kinds)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Vulkan image binding names must be nonempty strings")
        if len(names) != len(set(names)):
            raise ValueError("Vulkan image binding names must differ")
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError(
                "Vulkan image command requires an initialized Taichi runtime"
            )
        backend = _active_backend()
        if backend != "vulkan":
            raise TaichiRuntimeError(
                "Vulkan image command requires the Vulkan backend; the active "
                f"backend is {backend}"
            )
        super().__init__(
            backend="vulkan",
            binding_names=names,
            command_count=1,
            queue=queue,
            stream_binding="runtime_ordered",
            barrier_policy="internal",
            workspace_ownership="none",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "_binding_kinds", binding_kinds)
        object.__setattr__(self, "_runtime_prog", program)
        object.__setattr__(self, "_runtime_generation", int(impl.runtime_generation()))

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "Vulkan image")
        self.validate_graph_lifetime()
        for name, kind in self._binding_kinds:
            expected = Texture if kind == "texture" else Ndarray
            if not isinstance(bindings[name], expected):
                raise TaichiRuntimeError(
                    f"Vulkan image binding {name!r} must be a Taichi {kind}"
                )
        return self._execute(bindings)

    def validate_graph_lifetime(self):
        validate_runtime_generation(
            self,
            "Vulkan image recording belongs to a previous Taichi runtime "
            "generation",
        )

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            runtime_bindings=lambda item: item._binding_kinds,
            debug_info=lambda item: {"kind": item.debug_kind},
        )


class VulkanImageCopyRecording(_VulkanImageTransferRecording):
    """One reusable whole-image or region-to-region color copy."""

    def __init__(
        self,
        *,
        source="source",
        destination="destination",
        source_region=None,
        destination_region=None,
    ):
        super().__init__(((source, "texture"), (destination, "texture")))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(
            self, "source_region", _region(source_region, "source_region")
        )
        object.__setattr__(
            self,
            "destination_region",
            _region(destination_region, "destination_region"),
        )
        object.__setattr__(
            self, "_whole_image", source_region is None and destination_region is None
        )

    @property
    def resource_effects(self):
        if self._whole_image:
            source_scope = destination_scope = ("image", "whole")
        else:
            source_scope = self.source_region.effect_scope()
            destination_scope = self.destination_region.effect_scope(
                self.source_region.extent
            )
        return (
            ResourceEffect(self.source, GraphAccess.READ, subresource=source_scope),
            ResourceEffect(
                self.destination, GraphAccess.WRITE, subresource=destination_scope
            ),
        )

    def _execute(self, bindings):
        source = bindings[self.source]
        destination = bindings[self.destination]
        if self._whole_image:
            self._runtime_prog._vulkan_copy_texture(destination.tex, source.tex)
            return destination
        extent = self.source_region.resolved_extent(source)
        if (
            self.destination_region.extent is not None
            and self.destination_region.extent != extent
        ):
            raise TaichiRuntimeError(
                "Vulkan image copy source and destination region extents must match"
            )
        self._runtime_prog._vulkan_copy_texture_region(
            destination.tex,
            source.tex,
            self.source_region.offset,
            self.destination_region.offset,
            extent,
            self.source_region.mip_level,
            self.destination_region.mip_level,
            self.source_region.base_layer,
            self.destination_region.base_layer,
            self.source_region.layer_count,
        )
        return destination

    @property
    def debug_kind(self):
        return "vulkan_image_copy"


class VulkanBufferToImageRecording(_VulkanImageTransferRecording):
    """One raw ndarray-buffer to color-image region transfer."""

    def __init__(
        self,
        *,
        source="source",
        destination="destination",
        buffer_layout=None,
        image_region=None,
    ):
        super().__init__(((source, "ndarray"), (destination, "texture")))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "buffer_layout", _buffer_layout(buffer_layout))
        object.__setattr__(self, "image_region", _region(image_region, "image_region"))

    @property
    def resource_effects(self):
        return (
            ResourceEffect(
                self.source,
                GraphAccess.READ,
                subresource=self.buffer_layout.effect_scope(),
            ),
            ResourceEffect(
                self.destination,
                GraphAccess.WRITE,
                subresource=self.image_region.effect_scope(),
            ),
        )

    def _execute(self, bindings):
        source = bindings[self.source]
        destination = bindings[self.destination]
        extent = self.image_region.resolved_extent(destination)
        self._runtime_prog._vulkan_copy_ndarray_to_texture(
            destination.tex,
            source.arr,
            self.buffer_layout.byte_offset,
            self.buffer_layout.row_length,
            self.buffer_layout.image_height,
            self.image_region.offset,
            extent,
            self.image_region.mip_level,
            self.image_region.base_layer,
            self.image_region.layer_count,
        )
        return destination

    @property
    def debug_kind(self):
        return "vulkan_buffer_to_image"


class VulkanImageToBufferRecording(_VulkanImageTransferRecording):
    """One color-image region to raw ndarray-buffer transfer."""

    def __init__(
        self,
        *,
        source="source",
        destination="destination",
        buffer_layout=None,
        image_region=None,
    ):
        super().__init__(((source, "texture"), (destination, "ndarray")))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "buffer_layout", _buffer_layout(buffer_layout))
        object.__setattr__(self, "image_region", _region(image_region, "image_region"))

    @property
    def resource_effects(self):
        return (
            ResourceEffect(
                self.source,
                GraphAccess.READ,
                subresource=self.image_region.effect_scope(),
            ),
            ResourceEffect(
                self.destination,
                GraphAccess.WRITE,
                subresource=self.buffer_layout.effect_scope(),
            ),
        )

    def _execute(self, bindings):
        source = bindings[self.source]
        destination = bindings[self.destination]
        extent = self.image_region.resolved_extent(source)
        self._runtime_prog._vulkan_copy_texture_to_ndarray(
            destination.arr,
            source.tex,
            self.buffer_layout.byte_offset,
            self.buffer_layout.row_length,
            self.buffer_layout.image_height,
            self.image_region.offset,
            extent,
            self.image_region.mip_level,
            self.image_region.base_layer,
            self.image_region.layer_count,
        )
        return destination

    @property
    def debug_kind(self):
        return "vulkan_image_to_buffer"


class VulkanImageBlitRecording(_VulkanImageTransferRecording):
    """One feature-gated Vulkan image blit with nearest or linear filtering."""

    def __init__(
        self,
        *,
        source="source",
        destination="destination",
        source_region=None,
        destination_region=None,
        filter="nearest",
    ):
        if filter not in ("nearest", "linear"):
            raise ValueError("Vulkan image blit filter must be 'nearest' or 'linear'")
        super().__init__(
            ((source, "texture"), (destination, "texture")), queue="graphics"
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(
            self, "source_region", _region(source_region, "source_region")
        )
        object.__setattr__(
            self,
            "destination_region",
            _region(destination_region, "destination_region"),
        )
        object.__setattr__(self, "filter", filter)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(
                self.source,
                GraphAccess.READ,
                subresource=self.source_region.effect_scope(),
            ),
            ResourceEffect(
                self.destination,
                GraphAccess.WRITE,
                subresource=self.destination_region.effect_scope(),
            ),
        )

    def _execute(self, bindings):
        source = bindings[self.source]
        destination = bindings[self.destination]
        source_extent = self.source_region.resolved_extent(source)
        destination_extent = self.destination_region.resolved_extent(destination)
        if self.source_region.layer_count != self.destination_region.layer_count:
            raise TaichiRuntimeError(
                "Vulkan image blit source and destination layer counts must match"
            )
        self._runtime_prog._vulkan_blit_texture(
            destination.tex,
            source.tex,
            self.source_region.offset,
            source_extent,
            self.destination_region.offset,
            destination_extent,
            self.source_region.mip_level,
            self.destination_region.mip_level,
            self.source_region.base_layer,
            self.destination_region.base_layer,
            self.source_region.layer_count,
            self.filter == "linear",
        )
        return destination

    @property
    def debug_kind(self):
        return "vulkan_image_blit"


def copy(destination, source, *, source_region=None, destination_region=None):
    """Copy one complete color texture or one explicitly declared region."""

    recording = VulkanImageCopyRecording(
        source_region=source_region, destination_region=destination_region
    )
    return recording.execute({"source": source, "destination": destination})


def copy_buffer_to_image(destination, source, *, buffer_layout=None, image_region=None):
    recording = VulkanBufferToImageRecording(
        buffer_layout=buffer_layout, image_region=image_region
    )
    return recording.execute({"source": source, "destination": destination})


def copy_image_to_buffer(destination, source, *, buffer_layout=None, image_region=None):
    recording = VulkanImageToBufferRecording(
        buffer_layout=buffer_layout, image_region=image_region
    )
    return recording.execute({"source": source, "destination": destination})


def blit(
    destination,
    source,
    *,
    source_region=None,
    destination_region=None,
    filter="nearest",
):
    recording = VulkanImageBlitRecording(
        source_region=source_region,
        destination_region=destination_region,
        filter=filter,
    )
    return recording.execute({"source": source, "destination": destination})


__all__ = [
    "VulkanBufferImageLayout",
    "VulkanBufferToImageRecording",
    "VulkanImageBlitRecording",
    "VulkanImageCopyRecording",
    "VulkanImageRegion",
    "VulkanImageToBufferRecording",
    "blit",
    "copy",
    "copy_buffer_to_image",
    "copy_image_to_buffer",
]
