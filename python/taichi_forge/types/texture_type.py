import operator

from taichi_forge.lang.enums import Format
from taichi_forge.lang.exception import TaichiCompilationError
from taichi_forge.types.primitive_types import f16, f32, i8, i16, i32, u8, u16, u32

FORMAT2TY_CH = {
    Format.r8: (u8, 1),
    Format.r8u: (u8, 1),
    Format.r8i: (i8, 1),
    Format.rg8: (u8, 2),
    Format.rg8u: (u8, 2),
    Format.rg8i: (i8, 2),
    Format.rgba8: (u8, 4),
    Format.rgba8u: (u8, 4),
    Format.rgba8i: (i8, 4),
    Format.r16: (u16, 1),
    Format.r16u: (u16, 1),
    Format.r16i: (i16, 1),
    Format.r16f: (f16, 1),
    Format.rg16: (u16, 2),
    Format.rg16u: (u16, 2),
    Format.rg16i: (i16, 2),
    Format.rg16f: (f16, 2),
    Format.rgb16: (u16, 3),
    Format.rgb16u: (u16, 3),
    Format.rgb16i: (i16, 3),
    Format.rgb16f: (f16, 3),
    Format.rgba16: (u16, 4),
    Format.rgba16u: (u16, 4),
    Format.rgba16i: (i16, 4),
    Format.rgba16f: (f16, 4),
    Format.r32u: (u32, 1),
    Format.r32i: (i32, 1),
    Format.r32f: (f32, 1),
    Format.rg32u: (u32, 2),
    Format.rg32i: (i32, 2),
    Format.rg32f: (f32, 2),
    Format.rgb32u: (u32, 3),
    Format.rgb32i: (i32, 3),
    Format.rgb32f: (f32, 3),
    Format.rgba32u: (u32, 4),
    Format.rgba32i: (i32, 4),
    Format.rgba32f: (f32, 4),
}

# Reverse lookup by (channel_format, num_channels)
TY_CH2FORMAT = {v: k for k, v in FORMAT2TY_CH.items()}

_RW_TEXTURE_SIGNED_INTEGER_FORMATS = frozenset(
    {
        Format.r8i,
        Format.rg8i,
        Format.rgba8i,
        Format.r16i,
        Format.rg16i,
        Format.rgba16i,
        Format.r32i,
        Format.rg32i,
        Format.rgba32i,
    }
)
_RW_TEXTURE_UNSIGNED_INTEGER_FORMATS = frozenset(
    {
        Format.r8u,
        Format.rg8u,
        Format.rgba8u,
        Format.r16u,
        Format.rg16u,
        Format.rgba16u,
        Format.r32u,
        Format.rg32u,
        Format.rgba32u,
    }
)

_FLOAT_SAMPLED_TEXTURE_FORMATS = (
    frozenset(FORMAT2TY_CH)
    - _RW_TEXTURE_SIGNED_INTEGER_FORMATS
    - _RW_TEXTURE_UNSIGNED_INTEGER_FORMATS
) | frozenset(
    {
        Format.rgba8srgb,
        Format.bgra8,
        Format.bgra8srgb,
        Format.depth16,
        Format.depth32f,
    }
)


def is_float_sampled_texture_format(fmt):
    """Whether sampling exposes normalized/floating-point shader values."""
    return fmt in _FLOAT_SAMPLED_TEXTURE_FORMATS


def rw_texture_sampled_type(fmt):
    """Returns the shader-visible scalar type for storage image load/store."""
    if fmt in _RW_TEXTURE_SIGNED_INTEGER_FORMATS:
        return i32
    if fmt in _RW_TEXTURE_UNSIGNED_INTEGER_FORMATS:
        return u32
    return f32


class TextureType:
    """Type annotation for Textures.

    Args:
        num_dimensions (int): Number of dimensions. For examples for a 2D texture this should be `2`.
    """

    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions


class TextureCollectionType:
    """Type annotation for a fixed-capacity sampled texture collection."""

    def __init__(self, ndim, capacity):
        if isinstance(ndim, bool):
            raise TaichiCompilationError("texture_collection ndim must be an integer")
        if isinstance(capacity, bool):
            raise TaichiCompilationError("texture_collection capacity must be an integer")
        try:
            ndim = operator.index(ndim)
        except TypeError as exc:
            raise TaichiCompilationError(
                "texture_collection ndim must be an integer"
            ) from exc
        try:
            capacity = operator.index(capacity)
        except TypeError as exc:
            raise TaichiCompilationError(
                "texture_collection capacity must be an integer"
            ) from exc
        if not 1 <= ndim <= 3:
            raise TaichiCompilationError("texture_collection ndim must be in [1, 3]")
        if not 1 <= capacity <= 0x7FFFFFFF:
            raise TaichiCompilationError(
                "texture_collection capacity must be in [1, 2^31 - 1]"
            )
        self.num_dimensions = ndim
        self.capacity = capacity


class RWTextureType:
    """Type annotation for RW Textures (image load store).

    Args:
        num_dimensions (int): Number of dimensions. For examples for a 2D texture this should be `2`.
        lod (int): Allocated mip level bound as a storage image (2D Vulkan for nonzero levels).
        fmt (ti.Format): Color format of texture
    """

    def __init__(self, num_dimensions, lod=0, fmt=None):
        self.num_dimensions = num_dimensions
        if fmt is None:
            raise TaichiCompilationError("fmt is required for rw_texture type")
        else:
            self.fmt = fmt
        if isinstance(lod, bool):
            raise TaichiCompilationError("rw_texture lod must be an integer mip level")
        try:
            lod = operator.index(lod)
        except TypeError as exc:
            raise TaichiCompilationError("rw_texture lod must be an integer mip level") from exc
        if not 0 <= lod <= 30:
            raise TaichiCompilationError("rw_texture lod must be in [0, 30]")
        if lod and num_dimensions != 2:
            raise TaichiCompilationError("Nonzero rw_texture lod currently requires a 2D texture")
        self.lod = lod


texture = TextureType
texture_collection = TextureCollectionType
rw_texture = RWTextureType
"""Alias for :class:`~taichi_forge.types.ndarray_type.TextureType`.
"""

__all__ = ["texture", "texture_collection", "rw_texture"]
