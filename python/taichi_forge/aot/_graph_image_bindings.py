"""Cold image metadata and explicit consumer seeds for Graph export."""

from dataclasses import asdict

from taichi_forge._lib import core
from taichi_forge.lang import impl
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.texture_type import FORMAT2TY_CH, is_float_sampled_texture_format


class SourceImageBindings:
    def __init__(self, bindings, fixed_bindings):
        self.owners = {
            name: owner
            for name, owner in fixed_bindings.items()
            if type(owner) is Texture
        }
        self.used = set()
        self._storage_owners = set()
        self._attachment_owners = set()
        for name, owner in bindings.items():
            if not isinstance(name, str) or not name or "\0" in name:
                raise ValueError(
                    "Source binding names must be nonempty strings without NUL"
                )
            if name in fixed_bindings and fixed_bindings[name] is not owner:
                raise ValueError("Source binding conflicts with private Graph storage")
            self.owners[name] = owner
        for owner in self.owners.values():
            self._validate_owner(owner)

    @staticmethod
    def _validate_owner(owner):
        if owner.tex is None or owner._runtime_prog is not impl.get_runtime().prog:
            raise TaichiRuntimeError(
                "Source texture belongs to a retired or different runtime"
            )

    def require(self, name):
        owner = self.owners.get(name)
        if owner is None:
            raise TaichiRuntimeError(
                "AOT image needs its Texture in source_bindings"
            )
        self._validate_owner(owner)
        return owner

    def require_storage(self, name, owner):
        # Qualified binding preflight runs before any resource is reconstructed.
        # Keep one image owner even when sampling precedes a later shader write.
        if self.require(name) is not owner:
            raise TaichiRuntimeError("AOT storage image source differs from its fixed launch binding")
        self._storage_owners.add(id(owner))

    def require_attachment(self, name):
        owner = self.require(name)
        _, shape, mip_levels, _ = owner._creation_metadata
        if len(shape) != 2 or mip_levels != 1:
            raise TaichiRuntimeError("AOT graphics attachments require single-mip 2D textures")
        self._attachment_owners.add(id(owner))

    def validate_argument(self, arch, arg):
        owner = self.require(arg.name)
        fmt, shape, _, _ = owner._creation_metadata
        if len(shape) != len(arg.texture_shape):
            raise TaichiRuntimeError("AOT Texture dimension differs from the Graph kernel argument")
        if arg.tag == core.ArgKind.RWTEXTURE:
            if arch != core.Arch.vulkan:
                raise TaichiRuntimeError("AOT RWTexture Graph export requires Vulkan")
            matches = (
                fmt == arg.texture_format
                if arg.texture_format != core.Format.unknown
                else FORMAT2TY_CH[fmt] == (arg.channel_format(), arg.num_channels)
            )
            if not matches:
                raise TaichiRuntimeError("AOT RWTexture format differs from the Graph kernel argument")
            self.require_storage(arg.name, owner)
        elif arg.tag != core.ArgKind.TEXTURE or not is_float_sampled_texture_format(fmt):
            raise TaichiRuntimeError("AOT sampled Texture requires a floating sampling format")

    def preflight(self, arch, requirements):
        # Reuse the Graph's frozen resource contracts, including aliases whose
        # writes occur after an earlier sampling kernel. No second graph walk.
        for arg in requirements:
            self.validate_argument(arch, arg)

    def resource(self, context, name, *, first_clear=False):
        from taichi_forge.hardware._sampling import SamplerConfig

        owner = self.require(name)
        fmt, shape, mip_levels, sampler = owner._creation_metadata
        sampled = is_float_sampled_texture_format(fmt)
        attachment = id(owner) in self._attachment_owners
        if first_clear and not attachment:
            raise TaichiRuntimeError("AOT image clear requires a qualified attachment")
        if not sampled and (context.arch != core.Arch.vulkan or not (attachment or id(owner) in self._storage_owners)):
            raise TaichiRuntimeError("AOT integer image requires a Vulkan storage-image or attachment use")
        if context.arch not in (core.Arch.cuda, core.Arch.vulkan) or (
            mip_levels != 1 and (context.arch != core.Arch.vulkan or len(shape) != 2)
        ):
            raise TaichiRuntimeError(
                "AOT sampled-image reconstruction requires CUDA/Vulkan; mip chains require 2D Vulkan"
            )
        basic = SamplerConfig(
            min_filter=sampler.min_filter,
            mag_filter=sampler.min_filter,
            address_mode_u=sampler.address_mode_u,
            address_mode_v=sampler.address_mode_v,
            address_mode_w=sampler.address_mode_w,
        )
        if context.arch == core.Arch.cuda and sampler != basic:
            raise TaichiRuntimeError(
                "AOT sampled-image reconstruction requires a basic sampler with matching min/mag filters"
            )

        public = context.public_resource(owner)
        if public is not None:
            self.used.add(name)
            return public

        def build():
            ports = ("pixels",) if mip_levels == 1 else tuple(f"pixels_{i}" for i in range(mip_levels))
            # This only permits allocation without a seed. The shared native
            # recorder proves initialization along the selected control flow;
            # a conditional clear cannot authorize a later unconditional read.
            inputs = (
                context.resource_inputs.require(owner, ports)
                if not first_clear or context.resource_inputs.has_inputs(owner) else None
            )
            sampling = (
                dict(
                    filter=sampler.min_filter,
                    address_modes=[sampler.address_mode_u, sampler.address_mode_v, sampler.address_mode_w][: len(shape)],
                )
                if sampler == basic else dict(sampler=asdict(sampler))
            )
            resource = context.resource(
                owner,
                "image",
                "sampled" if sampled else "storage",
                parameters=dict(
                    format=fmt.name,
                    shape=list(shape),
                    **({"mip_levels": mip_levels} if mip_levels != 1 else {}),
                    **(sampling if sampled else {}),
                    **({"storage": True} if sampled and id(owner) in self._storage_owners else {}),
                    **({"attachment": True} if attachment else {}),
                ),
            )
            if inputs is not None:
                context.initialize_native_once(
                    resource,
                    "image",
                    "upload" if mip_levels == 1 else "upload_mips",
                    parameters={} if mip_levels == 1 else dict(mip_levels=mip_levels),
                    bindings=dict(image=resource, **inputs),
                )
            return resource

        resource = context.resource_once(
            owner, build, policy="image_consumer_seed"
        )
        self.used.add(name)
        return resource

    def kernel_resource(self, context, arg):
        self.validate_argument(context.arch, arg)
        return self.resource(context, arg.name)

    def validate_used(self):
        if self.owners.keys() - self.used:
            raise ValueError(
                "source_bindings contains textures unused by the exported Graph"
            )
