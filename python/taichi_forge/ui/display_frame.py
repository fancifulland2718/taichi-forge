import numpy as np
import weakref

from taichi_forge.lang._texture import Texture
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.types.primitive_types import u32


class DisplayCompletion:
    """GPU display-consumer completion, not confirmation of on-screen presentation."""

    def __init__(self, native):
        self._native = native

    @property
    def status(self):
        return self._native.status

    def done(self):
        return self._native.done()

    def wait(self):
        self._native.wait()


class WritableDisplayFrame:
    """Canvas-owned packed RGBA8 write lease from Canvas.acquire_frame().

    pixels is a (width, height) u32 dense view, x-major, y=0 at the bottom.
    Use it only before submit/cancel, on Forge's ordered CUDA execution path.
    Do not retain the view for later writes. Cancelling does not publish pixels.
    """

    def __init__(self, canvas, pixels, completion):
        self._canvas = weakref.ref(canvas)
        self._pixels = pixels
        self.width, self.height = pixels.shape
        self.completion = DisplayCompletion(completion)
        self.source_completion = None
        self._state = "writable"

    @property
    def pixels(self):
        canvas = self._canvas()
        if self._state != "writable" or canvas is None:
            raise RuntimeError("Display frame is no longer writable")
        canvas._check_display_owner()
        return self._pixels

    def cancel(self):
        if self._state != "writable":
            return
        canvas = self._canvas()
        if canvas is not None:
            canvas._cancel_write(self)

    def _seal(self, state):
        self._state = state
        self._pixels = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cancel()


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
        "_field_info",
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
        field_info=None,
    ):
        self.kind = kind
        self.width = int(width)
        self.height = int(height)
        self.row_stride_bytes = int(row_stride_bytes)
        self.transpose = bool(transpose)
        self._host_rgba8 = host_rgba8
        self._packed_u32 = packed_u32
        self._texture = texture
        self._field_info = field_info

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
        from .utils import get_field_info  # pylint: disable=import-outside-toplevel

        return cls(
            cls.PACKED_U32,
            image.shape[0],
            image.shape[1],
            0,
            transpose,
            packed_u32=image,
            field_info=get_field_info(image),
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

    @property
    def field_info(self):
        return self._field_info
