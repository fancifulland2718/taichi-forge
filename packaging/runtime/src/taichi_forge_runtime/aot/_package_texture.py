"""Package texture construction using the native image provider's ABI."""

from dataclasses import asdict, dataclass

from ._package_native import NativeObject


@dataclass(frozen=True)
class SamplerConfig:
    # Values are checked by the native image schema before allocation. Keep the
    # public defaults equal to hardware.sampling.SamplerConfig; no frontend is
    # imported by an independent consumer.
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
    compare_op: str | None = None


class Texture(NativeObject):
    def __init__(self, package, fmt, arr_shape, *, sampler=None, mip_levels=1):
        package._check()
        sampler = SamplerConfig() if sampler is None else sampler
        if not isinstance(sampler, SamplerConfig):
            raise TypeError("sampler must be a hardware.sampling.SamplerConfig")
        shape = tuple(arr_shape)
        # JIT color textures on Vulkan support both sampling and storage;
        # depth textures and CUDA textures are sampled-only.
        storage = package.runtime.arch == "vulkan" and not fmt.startswith("depth")
        resource = package.runtime.native_resource("image", "texture", parameters=dict(
            shape=shape, format=fmt, storage=storage, mip_levels=mip_levels,
            sampler=asdict(sampler)))
        super().__init__(package, resource)
        self.fmt, self.shape, self.num_dims = fmt, shape, len(shape)
        self.mip_levels, self.sampler = mip_levels, sampler
