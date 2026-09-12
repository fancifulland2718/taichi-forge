"""Portable sampled-image binding state for hardware texture operations."""

from dataclasses import dataclass
import math
from numbers import Real

from taichi_forge._lib import core as _ti_core

_FILTERS = {
    "nearest": _ti_core.ImageFilter.nearest,
    "linear": _ti_core.ImageFilter.linear,
}
_ADDRESS_MODES = {
    "repeat": _ti_core.ImageAddressMode.repeat,
    "mirrored_repeat": _ti_core.ImageAddressMode.mirrored_repeat,
    "clamp_to_edge": _ti_core.ImageAddressMode.clamp_to_edge,
}


def _choice(value, choices, label):
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    try:
        return choices[value]
    except KeyError as exc:
        supported = ", ".join(sorted(choices))
        raise ValueError(f"unsupported {label} {value!r}; choose {supported}") from exc


@dataclass(frozen=True)
class SamplerConfig:
    """Immutable texture sampling state.

    Sampling uses normalized coordinates. Textures default to one mip level;
    managed 2D Vulkan textures may allocate multiple levels explicitly. Vulkan
    defaults to the nearest mip level. ``mip_filter="linear"`` interpolates
    between levels, independently of the min/mag texel filters. LOD bias,
    clamps and anisotropy are currently Vulkan-only. ``max_lod=None`` leaves
    the upper LOD unclamped; ``max_anisotropy=1`` disables anisotropy.
    ``Texture.fetch`` uses exact integer coordinates and ignores sampler state.
    """

    min_filter: str = "linear"
    mag_filter: str = "linear"
    address_mode_u: str = "repeat"
    address_mode_v: str = "repeat"
    address_mode_w: str = "repeat"
    mip_filter: str = "nearest"
    lod_bias: float = 0.0
    min_lod: float = 0.0
    max_lod: float | None = None
    max_anisotropy: float = 1.0

    def __post_init__(self):
        _choice(self.min_filter, _FILTERS, "min_filter")
        _choice(self.mag_filter, _FILTERS, "mag_filter")
        _choice(self.address_mode_u, _ADDRESS_MODES, "address_mode_u")
        _choice(self.address_mode_v, _ADDRESS_MODES, "address_mode_v")
        _choice(self.address_mode_w, _ADDRESS_MODES, "address_mode_w")
        _choice(self.mip_filter, _FILTERS, "mip_filter")
        for label in ("lod_bias", "min_lod", "max_lod", "max_anisotropy"):
            value = getattr(self, label)
            if label == "max_lod" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{label} must be a finite real number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
            object.__setattr__(self, label, value)
        if self.min_lod < 0 or (
            self.max_lod is not None and self.max_lod < self.min_lod
        ):
            raise ValueError("LOD clamps require 0 <= min_lod <= max_lod")
        if self.max_anisotropy < 1:
            raise ValueError("max_anisotropy must be at least 1")

    def _as_core_config(self):
        config = _ti_core.ImageSamplerConfig()
        config.min_filter = _FILTERS[self.min_filter]
        config.mag_filter = _FILTERS[self.mag_filter]
        config.address_mode_u = _ADDRESS_MODES[self.address_mode_u]
        config.address_mode_v = _ADDRESS_MODES[self.address_mode_v]
        config.address_mode_w = _ADDRESS_MODES[self.address_mode_w]
        config.mip_filter = _FILTERS[self.mip_filter]
        config.lod_bias = self.lod_bias
        config.min_lod = self.min_lod
        config.max_lod = -1.0 if self.max_lod is None else self.max_lod
        config.max_anisotropy = self.max_anisotropy
        return config


__all__ = ["SamplerConfig"]
