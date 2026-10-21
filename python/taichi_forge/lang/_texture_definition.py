"""Normalized image creation facts shared with cold resource reconstruction.

Definitions contain no image handles, texels or owners. Initialization and
sampled/storage/attachment qualification stay with the selected graph context.
"""

from dataclasses import asdict, dataclass

from taichi_forge.types.texture_type import is_float_sampled_texture_format


@dataclass(frozen=True)
class TextureDefinition:
    format: object
    shape: tuple
    mip_levels: int
    sampler: object

    @property
    def metadata(self):
        return self.format, self.shape, self.mip_levels, self.sampler

    @property
    def basic_sampler(self):
        from taichi_forge.hardware._sampling import SamplerConfig

        return SamplerConfig(
            min_filter=self.sampler.min_filter,
            mag_filter=self.sampler.min_filter,
            address_mode_u=self.sampler.address_mode_u,
            address_mode_v=self.sampler.address_mode_v,
            address_mode_w=self.sampler.address_mode_w,
        )

    def native_definition(self, *, storage=False, attachment=False):
        from taichi_forge.graph._native_definition import NativeDefinition

        sampler = self.sampler
        sampled = is_float_sampled_texture_format(self.format)
        sampling = (
            dict(
                filter=sampler.min_filter,
                address_modes=[sampler.address_mode_u, sampler.address_mode_v, sampler.address_mode_w][
                    : len(self.shape)
                ],
            )
            if sampler == self.basic_sampler
            else dict(sampler=asdict(sampler))
        )
        return NativeDefinition(
            "image",
            "sampled" if sampled else "storage",
            dict(
                format=self.format.name,
                shape=self.shape,
                **({"mip_levels": self.mip_levels} if self.mip_levels != 1 else {}),
                **(sampling if sampled else {}),
                **({"storage": True} if sampled and storage else {}),
                **({"attachment": True} if attachment else {}),
            ),
        )


@dataclass(frozen=True)
class TextureCollectionDefinition:
    dimensions: int
    image_count: int

    def member_links(self, members):
        # Slot order and repeated members are part of the resource contract.
        if len(members) != self.image_count:
            raise ValueError("TextureCollection definition differs from its member count")
        return {f"image_{i}": member for i, member in enumerate(members)}

    def native_definition(self):
        from taichi_forge.graph._native_definition import NativeDefinition

        return NativeDefinition("image", "collection", dict(dimensions=self.dimensions, image_count=self.image_count))
