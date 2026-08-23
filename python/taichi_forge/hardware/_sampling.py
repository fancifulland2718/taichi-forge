"""Portable sampled-image binding state for hardware texture operations."""

from dataclasses import dataclass

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
    """Immutable filter and address state for ``Texture.sample_lod``.

    Taichi textures currently expose one mip level and normalized coordinates.
    ``Texture.fetch`` is an exact integer-coordinate operation and ignores this
    sampler state.
    """

    min_filter: str = "linear"
    mag_filter: str = "linear"
    address_mode_u: str = "repeat"
    address_mode_v: str = "repeat"
    address_mode_w: str = "repeat"

    def __post_init__(self):
        _choice(self.min_filter, _FILTERS, "min_filter")
        _choice(self.mag_filter, _FILTERS, "mag_filter")
        _choice(self.address_mode_u, _ADDRESS_MODES, "address_mode_u")
        _choice(self.address_mode_v, _ADDRESS_MODES, "address_mode_v")
        _choice(self.address_mode_w, _ADDRESS_MODES, "address_mode_w")

    def _as_core_config(self):
        config = _ti_core.ImageSamplerConfig()
        config.min_filter = _FILTERS[self.min_filter]
        config.mag_filter = _FILTERS[self.mag_filter]
        config.address_mode_u = _ADDRESS_MODES[self.address_mode_u]
        config.address_mode_v = _ADDRESS_MODES[self.address_mode_v]
        config.address_mode_w = _ADDRESS_MODES[self.address_mode_w]
        return config


__all__ = ["SamplerConfig"]
