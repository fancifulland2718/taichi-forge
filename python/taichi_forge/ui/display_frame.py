import numpy as np

from taichi_forge.lang._texture import Texture
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.types.primitive_types import u32


class DisplayFrame:
    """A display-ready frame for the GGUI set_image submission path.

    This is intentionally narrower than Canvas.set_image(): callers use it when
    the image is already in a displayable representation and should not be
    repacked through the generic numpy/field/ndarray path.
    """

    __slots__ = (
        "kind",
        "width",
        "height",
        "row_stride_bytes",
        "transpose",
        "_host_rgba8",
        "_packed_u32",
        "_texture",
    )

    HOST_RGBA8 = "host_rgba8"
    PACKED_U32 = "packed_u32"
    TEXTURE = "texture"

    def __init__(
        self,
        kind,
        width,
        height,
        row_stride_bytes=0,
        transpose=True,
        host_rgba8=None,
        packed_u32=None,
        texture=None,
    ):
        self.kind = kind
        self.width = int(width)
        self.height = int(height)
        self.row_stride_bytes = int(row_stride_bytes)
        self.transpose = bool(transpose)
        self._host_rgba8 = host_rgba8
        self._packed_u32 = packed_u32
        self._texture = texture

    @classmethod
    def from_numpy_rgba8(cls, image, *, copy=False, transpose=True):
        arr = np.asarray(image)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 4:
            raise ValueError("DisplayFrame host input must be a uint8 RGBA image")
        if not arr.flags.c_contiguous:
            if not copy:
                raise ValueError("DisplayFrame host input must be C-contiguous")
            arr = np.ascontiguousarray(arr)
        return cls(
            cls.HOST_RGBA8,
            arr.shape[0],
            arr.shape[1],
            arr.strides[0],
            transpose,
            host_rgba8=arr,
        )

    @classmethod
    def from_texture(cls, texture, *, transpose=False):
        if not isinstance(texture, Texture):
            raise TypeError("DisplayFrame.from_texture() expects a ti.Texture")
        return cls(
            cls.TEXTURE,
            texture.shape[0],
            texture.shape[1],
            0,
            transpose,
            texture=texture,
        )

    @classmethod
    def from_packed_u32_ndarray(cls, image, *, transpose=True):
        if not isinstance(image, Ndarray):
            raise TypeError("DisplayFrame.from_packed_u32_ndarray() expects a ti.ndarray")
        if image.dtype != u32 or len(image.shape) != 2:
            raise ValueError("packed display frame input must be a 2D u32 ndarray")
        return cls(
            cls.PACKED_U32,
            image.shape[0],
            image.shape[1],
            0,
            transpose,
            packed_u32=image,
        )

    @property
    def host_rgba8(self):
        return self._host_rgba8

    @property
    def packed_u32(self):
        return self._packed_u32

    @property
    def texture(self):
        return self._texture
